"""
Backend chính của hệ thống -- ĐÃ REFACTOR theo pipeline Tầng 2:

    Query
      -> Query Parser        (query_parser.py)      : trích negation/status/chapters/genre
      -> Retrieval pool rộng (search_bm25/search_semantic)
      -> SQLite enrich       (join lấy đủ metadata)
      -> Constraint Filter   (constraint_filter.py)  : lọc cứng theo các constraint đã parse
      -> Rerank + Top-K      (sort theo score, cắt về đúng top_k)

Lưu ý kỹ thuật (đã bàn với người dùng): về lý thuyết "Constraint Filter" nên
đứng TRƯỚC Retrieval (lọc SQLite trước, chỉ chạy BM25/Semantic trên tập con).
Nhưng bm25.pkl/faiss.index hiện build sẵn trên toàn corpus, không dễ subset
tại query-time mà không sửa sâu search_bm25.py/search_semantic.py. Nên chọn
cách TƯƠNG ĐƯƠNG về kết quả: lấy pool rộng (fetch_k) trước, filter sau, rồi
mới cắt về top_k -- đơn giản hơn, ít rủi ro hơn, cùng hiệu quả với dữ liệu
quy mô hiện tại.
"""

import os
import sys
import sqlite3

import pandas as pd

# ---------------------------------------------------------------------------
# Thiết lập đường dẫn để import được search_bm25 / search_semantic / các
# module cùng cấp (query_parser, constraint_filter), dù retrieval.py được
# chạy/import từ bất kỳ đâu (Streamlit, notebook, CLI...)
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)                     # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                    # .../System_Sematic_Search_Truyen

BM25_DIR = os.path.join(SRC_DIR, "bm25")
SEMANTIC_DIR = os.path.join(SRC_DIR, "semantic")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "stories.db")

for p in (BM25_DIR, SEMANTIC_DIR, CURRENT_DIR):
    if p not in sys.path:
        sys.path.append(p)

import search_bm25                     # noqa: E402
import search_semantic as sem          # noqa: E402
from query_parser import parse_query   # noqa: E402
import constraint_filter as cf         # noqa: E402


# ---------------------------------------------------------------------------
# Cache: model semantic + danh sách genre (đều tốn chi phí load, chỉ load 1 lần)
# ---------------------------------------------------------------------------
_semantic_model = None
_semantic_index = None
_semantic_stories = None
_known_genres = None


def _load_semantic_once():
    global _semantic_model, _semantic_index, _semantic_stories

    if _semantic_model is None:
        _semantic_model = sem.load_model()
        _semantic_index = sem.load_index()
        _semantic_stories = sem.load_stories()

    return _semantic_model, _semantic_index, _semantic_stories


def _load_known_genres_once() -> list:
    """Lấy danh sách genre DUY NHẤT từ SQLite (tách theo dấu phẩy), dùng cho
    Query Parser nhận diện genre_hints đúng theo dữ liệu thật của dự án."""
    global _known_genres

    if _known_genres is None:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT DISTINCT genre FROM stories WHERE genre IS NOT NULL").fetchall()
        conn.close()

        genres = set()
        for (g,) in rows:
            for part in g.split(","):
                part = part.strip()
                if part:
                    genres.add(part)

        _known_genres = sorted(genres)

    return _known_genres


def _enrich_with_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join kết quả retrieval (chỉ có vài cột từ stories.pkl) với bảng `stories`
    trong SQLite để lấy đầy đủ metadata: author, description, tags, status,
    chapters, views, url -- cần đủ các cột này để Constraint Filter hoạt động.

    Tự nhận diện cột khoá join là "id" hoặc "story_id".
    """
    id_col = None
    for candidate in ("id", "story_id"):
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        print(
            "[Cảnh báo] Không tìm thấy cột 'id'/'story_id' trong kết quả retrieval "
            "-> không join được với SQLite, Constraint Filter sẽ thiếu dữ liệu."
        )
        return df

    ids = [int(i) for i in df[id_col].tolist()]
    if not ids:
        return df

    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(ids))
    query = f"SELECT * FROM stories WHERE id IN ({placeholders})"
    db_df = pd.read_sql_query(query, conn, params=ids)
    conn.close()

    score_map = dict(zip(df[id_col], df["score"]))
    db_df["score"] = db_df["id"].map(score_map)
    db_df = db_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    return db_df


def search(query: str, method: str = "semantic", top_k: int = 10, verbose: bool = True) -> pd.DataFrame:
    """
    Hàm backend chính -- pipeline đầy đủ:
    Query -> Parser -> Retrieval (pool rộng) -> SQLite enrich -> Constraint Filter -> Top-K

    Parameters
    ----------
    query : str
    method : "bm25" hoặc "semantic"
    top_k : số kết quả trả về
    verbose : có in log các bước filter hay không (tắt khi chạy batch eval)
    """
    method = method.lower().strip()

    known_genres = _load_known_genres_once()
    parsed = parse_query(query, known_genres)

    if verbose and parsed.has_constraints():
        print(f"[Query Parser] {parsed}")

    # Có constraint nào cần lọc thêm -> lấy pool rộng hơn top_k để không bị
    # thiếu kết quả sau khi filter.
    fetch_k = min(max(top_k * 5, 50), 200) if parsed.has_constraints() else top_k

    if method == "bm25":
        results = search_bm25.search(query, top_k=fetch_k)

    elif method == "semantic":
        model, index, stories = _load_semantic_once()
        results = sem.search(query, model, index, stories, top_k=fetch_k)

    else:
        raise ValueError(f"Method không hợp lệ: '{method}'. Chọn 'bm25' hoặc 'semantic'.")

    results = results.reset_index(drop=True)
    results = _enrich_with_db(results)

    before = len(results)
    results = cf.apply_constraints(results, parsed)
    if verbose and parsed.has_constraints():
        print(f"[Constraint Filter] {before} -> {len(results)} kết quả sau khi lọc.")

    results = results.sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)

    return results


if __name__ == "__main__":
    # Test nhanh backend từ terminal, không cần Streamlit.
    while True:
        query = input("\nNhập truy vấn (hoặc 'exit'): ")
        if query.lower() == "exit":
            break

        method = input("Method (bm25/semantic) [semantic]: ").strip() or "semantic"

        try:
            top_k = int(input("Top K [10]: ").strip() or "10")
        except ValueError:
            top_k = 10

        results = search(query, method=method, top_k=top_k)

        print(f"\n===== KẾT QUẢ ({method.upper()}) =====\n")
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            print(f"{rank}. {row.get('title')}")
            print(f"   Thể loại  : {row.get('genre')}")
            print(f"   Trạng thái: {row.get('status')}")
            print(f"   Số chương : {row.get('chapters')}")
            print(f"   URL       : {row.get('url')}")
            print(f"   Score     : {row.get('score'):.4f}")
            print()