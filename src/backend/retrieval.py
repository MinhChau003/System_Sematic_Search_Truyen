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
import re
import sys
import sqlite3

import pandas as pd

# ---------------------------------------------------------------------------
# Thiết lập đường dẫn để import được search_bm25 / search_semantic
# dù retrieval.py được chạy/import từ bất kỳ đâu (Streamlit, notebook, CLI...)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Semantic cần load model + FAISS index 1 lần duy nhất (tốn thời gian),
# nên cache lại bằng biến global thay vì load mỗi lần search().
# ---------------------------------------------------------------------------
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

    Tự nhận diện cột khoá join là "id" hoặc "story_id" (2 file stories.pkl của
    BM25 và Semantic có thể đặt tên cột id khác nhau).
    """
    id_col = None
    for candidate in ("id", "story_id"):
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        print(
            "[Cảnh báo] Không tìm thấy cột 'id' hoặc 'story_id' trong kết quả "
            "retrieval -> không join được với SQLite. Kiểm tra lại tên cột khoá "
            "trong stories.pkl."
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

    # Gắn lại score từ kết quả retrieval (SQLite không có cột score)
    score_map = dict(zip(df[id_col], df["score"]))
    db_df["score"] = db_df["id"].map(score_map)

    db_df = db_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    return db_df


# ---------------------------------------------------------------------------
# FIX LỖI E2 (Negative constraint violation): query dạng "không có X" / "không X"
# nhưng kết quả vẫn chứa X. Hướng xử lý: rule-based, KHÔNG sửa BM25/Semantic gốc.
#   1. Trích các cụm bị phủ định từ câu query.
#   2. Loại các truyện có genre/tags chứa cụm đó.
# ---------------------------------------------------------------------------

# Các từ đệm hay đứng ngay sau "không" mà không mang nghĩa cần lọc,
# cần bóc ra để lấy đúng phần "X" thực sự bị phủ định.
_NEGATION_FILLERS = [
    "có yếu tố ", "có ", "thuộc thể loại ", "thuộc ",
    "tập trung vào ", "yếu tố ", "thể loại ",
    "phải là ", "phải ", "mang yếu tố ", "chứa yếu tố ", "chứa ",
]

_NEGATION_PATTERN = re.compile(
    r"không\s+([^,\.]+?)(?=\s+(?:và|nhưng)\b|,|\.|$)",
    flags=re.IGNORECASE,
)


def _strip_negation_fillers(clause: str) -> str:
    clause = clause.strip()
    changed = True
    while changed:
        changed = False
        for filler in _NEGATION_FILLERS:
            if clause.lower().startswith(filler):
                clause = clause[len(filler):].strip()
                changed = True
                break
    return clause.strip()


def extract_negated_phrases(query: str) -> list:
    """Trích các cụm bị phủ định trong câu, VD 'không có yếu tố harem' -> ['harem']."""
    phrases = []
    for m in _NEGATION_PATTERN.finditer(query):
        cleaned = _strip_negation_fillers(m.group(1))
        if cleaned and len(cleaned) >= 2:
            phrases.append(cleaned.lower())
    return phrases


# Từ điển đồng nghĩa cho các khái niệm phủ định hay gặp trong bộ query thực tế
# -- mô tả truyện hay diễn đạt cùng 1 ý bằng nhiều từ khác nhau (VD: harem có
# thể được viết là "nhiều nữ chính", "hậu cung"...). Không đầy đủ tuyệt đối,
# mở rộng dần khi phát hiện thêm case mới qua Error Analysis.
_NEGATION_SYNONYMS = {
    "harem": ["harem", "hậu cung", "nhiều nữ chính", "đa nữ chính", "tam thê tứ thiếp", "đa thê"],
    "trọng sinh": ["trọng sinh", "tái sinh", "sống lại", "trùng sinh"],
    "xuyên không": ["xuyên không", "xuyên việt", "xuyên qua", "xuyên thư", "xuyên sách"],
    "tình cảm": ["tình cảm", "yêu đương", "lãng mạn"],
}


def _expand_synonyms(phrase: str) -> list:
    return _NEGATION_SYNONYMS.get(phrase, [phrase])


# Regex kiểm tra xem ngay TRƯỚC 1 occurrence của cụm bị phủ định có phải là
# 1 từ phủ định hay không (VD "...#Không hậu cung" -> đứng trước "hậu cung" là "Không").
# Nếu đúng, occurrence đó đang XÁC NHẬN sự vắng mặt (tín hiệu TỐT, không phải vi phạm).
_NEGATION_CONFIRM_SUFFIX = re.compile(
    r"(không|chẳng)\s+(?:có\s+)?(?:yếu\s+tố\s+)?(?:thuộc\s+)?(?:thể\s+loại\s+)?$",
    flags=re.IGNORECASE,
)


def _phrase_confirmed_absent_everywhere(text: str, phrase: str, window: int = 30) -> bool:
    """
    True nếu phrase KHÔNG xuất hiện trong text, HOẶC mọi occurrence của nó đều
    được phủ định ngay trước (VD '#Không hậu cung') -> an toàn, không phải vi phạm.
    False nếu có ít nhất 1 occurrence xuất hiện mà KHÔNG đi kèm phủ định phía trước
    -> truyện thực sự có yếu tố đó, cần loại.
    """
    for m in re.finditer(re.escape(phrase), text):
        preceding = text[max(0, m.start() - window):m.start()]
        if not _NEGATION_CONFIRM_SUFFIX.search(preceding):
            return False
    return True


def _apply_negation_filter(df: pd.DataFrame, phrases: list) -> pd.DataFrame:
    """
    Loại các dòng thực sự chứa yếu tố bị phủ định trong query.

    - Mỗi phrase được mở rộng qua _NEGATION_SYNONYMS (VD "harem" cũng khớp
      "hậu cung", "nhiều nữ chính"...) vì mô tả truyện hay diễn đạt cùng 1 ý
      bằng nhiều từ khác nhau, không phải lúc nào cũng dùng đúng từ trong query.
    - title/genre/tags: field ngắn, hiếm khi tự viết dạng "Không X" -> chỉ cần
      substring match đơn giản.
    - description: field dài, THƯỜNG chứa các tag ẩn dạng "#Không hậu cung",
      "#Đơn nữ chính"... -> cần kiểm tra ngữ cảnh phủ định phía trước để tránh
      loại nhầm truyện đang XÁC NHẬN không có yếu tố đó (xem
      _phrase_confirmed_absent_everywhere).
    """
    if not phrases or df.empty:
        return df

    def _col(name):
        return df[name].fillna("") if name in df.columns else pd.Series([""] * len(df), index=df.index)

    title_l = _col("title").str.lower()
    genre_l = _col("genre").str.lower()
    tags_l = _col("tags").str.lower()
    desc_l = _col("description").str.lower()

    keep = []
    for idx in df.index:
        short_text = f"{title_l.loc[idx]} {genre_l.loc[idx]} {tags_l.loc[idx]}"
        desc_text = desc_l.loc[idx]

        row_ok = True
        for phrase in phrases:
            for term in _expand_synonyms(phrase):
                if term in short_text:
                    row_ok = False
                    break
                if term in desc_text and not _phrase_confirmed_absent_everywhere(desc_text, term):
                    row_ok = False
                    break
            if not row_ok:
                break

        keep.append(row_ok)

    return df[pd.Series(keep, index=df.index)]


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

    # Nếu query có phủ định, cần lấy pool LỚN HƠN top_k trước, vì sau khi lọc
    # bỏ các truyện vi phạm, số lượng còn lại có thể ít hơn top_k yêu cầu.
    negated_phrases = extract_negated_phrases(query)
    fetch_k = min(max(top_k * 5, 50), 200) if negated_phrases else top_k

    if method == "bm25":
        results = search_bm25.search(query, top_k=fetch_k)

    elif method == "semantic":
        model, index, stories = _load_semantic_once()
        results = sem.search(query, model, index, stories, top_k=fetch_k)

    else:
        raise ValueError(f"Method không hợp lệ: '{method}'. Chọn 'bm25' hoặc 'semantic'.")

    results = results.reset_index(drop=True)
    results = _enrich_with_db(results)

    if negated_phrases:
        before = len(results)
        results = _apply_negation_filter(results, negated_phrases)
        print(f"[Negation filter] Cụm bị loại: {negated_phrases} "
              f"-> còn {len(results)}/{before} kết quả sau khi lọc.")

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
            print(f"   Thể loại : {row.get('genre')}")
            print(f"   Trạng thái: {row.get('status')}")
            print(f"   URL       : {row.get('url')}")
            print(f"   Score     : {row.get('score'):.4f}")
            print()