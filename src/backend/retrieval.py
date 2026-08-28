"""
Backend chính của hệ thống: Query -> Retrieval -> Top-K

Luồng xử lý:
1. Nhận query từ người dùng (Streamlit sau này sẽ gọi vào đây).
2. Tuỳ method ("bm25" hoặc "semantic"), gọi lại đúng hàm search() có sẵn
   trong src/bm25/search_bm25.py hoặc src/semantic/search_semantic.py.
3. Lấy id các truyện trong Top-K, join với bảng `stories` trong SQLite
   để lấy đầy đủ metadata (description, tags, chapters, views, url)
   -- vì stories.pkl có thể không có đủ các cột này.
4. Trả về DataFrame Top-K đã sắp theo score giảm dần, sẵn sàng cho UI.
"""

import os
import sys
import sqlite3

import pandas as pd

# Thiết lập đường dẫn để import được search_bm25 / search_semantic
# dù retrieval.py được chạy/import từ bất kỳ đâu (Streamlit, notebook, CLI...)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)                     # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                    # .../System_Sematic_Search_Truyen

BM25_DIR = os.path.join(SRC_DIR, "bm25")
SEMANTIC_DIR = os.path.join(SRC_DIR, "semantic")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "stories.db")

for p in (BM25_DIR, SEMANTIC_DIR):
    if p not in sys.path:
        sys.path.append(p)

import search_bm25                     # noqa: E402  (import sau khi set sys.path)
import search_semantic as sem          # noqa: E402


# Semantic cần load model + FAISS index 1 lần duy nhất (tốn thời gian),
# nên cache lại bằng biến global thay vì load mỗi lần search().

_semantic_model = None
_semantic_index = None
_semantic_stories = None


def _load_semantic_once():
    global _semantic_model, _semantic_index, _semantic_stories

    if _semantic_model is None:
        _semantic_model = sem.load_model()
        _semantic_index = sem.load_index()
        _semantic_stories = sem.load_stories()

    return _semantic_model, _semantic_index, _semantic_stories


def _enrich_with_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join kết quả retrieval (chỉ có vài cột từ stories.pkl) với bảng `stories`
    trong SQLite để lấy đầy đủ metadata: author, description, tags, status,
    chapters, views, url.

    Nếu df không có cột "id" (không xác định được join key), trả về nguyên bản
    kèm cảnh báo -- khi đó cần kiểm tra lại stories.pkl có cột id hay không.
    """
    if "id" not in df.columns:
        print(
            "[Cảnh báo] Không tìm thấy cột 'id' trong kết quả retrieval "
            "-> không join được với SQLite. Kiểm tra lại stories.pkl có cột "
            "'id' trùng với id trong database/stories.db không."
        )
        return df

    ids = [int(i) for i in df["id"].tolist()]
    if not ids:
        return df

    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(ids))
    query = f"SELECT * FROM stories WHERE id IN ({placeholders})"
    db_df = pd.read_sql_query(query, conn, params=ids)
    conn.close()

    # Gắn lại score từ kết quả retrieval (SQLite không có cột score)
    score_map = dict(zip(df["id"], df["score"]))
    db_df["score"] = db_df["id"].map(score_map)

    db_df = db_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    return db_df


def search(query: str, method: str = "semantic", top_k: int = 10) -> pd.DataFrame:
    """
    Hàm backend chính: Query -> Retrieval -> Top-K

    Parameters
    ----------
    query : str
        Câu truy vấn của người dùng.
    method : str
        "bm25" hoặc "semantic".
    top_k : int
        Số kết quả trả về.

    Returns
    -------
    pd.DataFrame
        Top-K kết quả, đã có đầy đủ metadata (nếu join SQLite thành công)
        và cột "score" để sắp xếp / hiển thị.
    """
    method = method.lower().strip()

    if method == "bm25":
        results = search_bm25.search(query, top_k=top_k)

    elif method == "semantic":
        model, index, stories = _load_semantic_once()
        results = sem.search(query, model, index, stories, top_k=top_k)

    else:
        raise ValueError(f"Method không hợp lệ: '{method}'. Chọn 'bm25' hoặc 'semantic'.")

    results = results.reset_index(drop=True)
    results = _enrich_with_db(results)

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
            print(f"   Thể loại : {row.get('genre')}")
            print(f"   Trạng thái: {row.get('status')}")
            print(f"   URL       : {row.get('url')}")
            print(f"   Score     : {row.get('score'):.4f}")
            print()