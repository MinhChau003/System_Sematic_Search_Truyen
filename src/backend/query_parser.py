"""
Query Parser -- Tầng 2, mục 1+2+3 của roadmap.

Tách 1 câu query tự nhiên thành các constraint có cấu trúc:
    - negated_phrases   : các cụm bị phủ định ("không có X")
    - status            : trạng thái được yêu cầu KHẲNG ĐỊNH (VD "Đang ra")
    - chapter_constraint: tuple (operator, số) VD (">", 500)
    - genre_hints       : các thể loại được nhắc trực tiếp trong câu

Đây là bước TIỀN xử lý, chạy TRƯỚC khi đưa query vào BM25/Semantic.
Không thay thế BM25/Semantic -- chỉ trích thêm thông tin có cấu trúc để
Constraint Filter (bước sau) dùng lọc/rerank kết quả.
"""

import re


# ---------------------------------------------------------------------------
# 1. NEGATION -- dời từ retrieval.py sang đây làm nguồn chuẩn duy nhất.
# ---------------------------------------------------------------------------
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
    """VD: 'không có yếu tố harem' -> ['harem']."""
    phrases = []
    for m in _NEGATION_PATTERN.finditer(query):
        cleaned = _strip_negation_fillers(m.group(1))
        if cleaned and len(cleaned) >= 2:
            phrases.append(cleaned.lower())
    return phrases


# ---------------------------------------------------------------------------
# 2. STATUS constraint (khẳng định) -- phân biệt với status bị phủ định.
# ---------------------------------------------------------------------------
_STATUS_MAP = {
    "đang tiến hành": "Đang ra",
    "đã hoàn thành": "Hoàn thành",
    "đang ra": "Đang ra",
    "hoàn thành": "Hoàn thành",
    "hoàn tất": "Hoàn thành",
    "tạm dừng": "Tạm dừng",
    "ngừng": "Tạm dừng",
}
# Key dài match trước để tránh match nhầm cụm ngắn nằm trong cụm dài.
_STATUS_KEYS_SORTED = sorted(_STATUS_MAP.keys(), key=len, reverse=True)


def extract_status_constraint(query: str):
    """
    Trả về giá trị status chuẩn (VD "Đang ra") nếu query yêu cầu KHẲNG ĐỊNH.
    Bỏ qua nếu cụm đó đang bị phủ định (VD "không phải đang ra").
    """
    query_lower = query.lower()

    for key in _STATUS_KEYS_SORTED:
        idx = query_lower.find(key)
        if idx == -1:
            continue

        context_before = query_lower[max(0, idx - 15):idx]
        if "không" in context_before:
            continue  # đây là phủ định, đã được extract_negated_phrases xử lý riêng

        return _STATUS_MAP[key]

    return None


# ---------------------------------------------------------------------------
# 3. CHAPTER constraint: "hơn 500 chương", "dưới 100 chương"...
# ---------------------------------------------------------------------------
_CHAPTER_PATTERN = re.compile(
    r"(hơn|trên|từ|ít nhất|tối thiểu|dưới|ít hơn|tối đa|không quá)\s+(\d+)\s*chương",
    flags=re.IGNORECASE,
)

_CHAPTER_OP_MAP = {
    "hơn": ">", "trên": ">", "từ": ">=", "ít nhất": ">=", "tối thiểu": ">=",
    "dưới": "<", "ít hơn": "<", "tối đa": "<=", "không quá": "<=",
}


def extract_chapter_constraint(query: str):
    """Trả về (operator, number) hoặc None. VD ('>', 500)."""
    m = _CHAPTER_PATTERN.search(query.lower())
    if not m:
        return None
    keyword, number = m.group(1), int(m.group(2))
    return (_CHAPTER_OP_MAP.get(keyword, ">"), number)


# ---------------------------------------------------------------------------
# 4. GENRE hints -- cần danh sách genre THẬT từ DB, không hardcode.
# ---------------------------------------------------------------------------
def extract_genre_hints(query: str, known_genres: list) -> list:
    """
    known_genres: danh sách thể loại DUY NHẤT lấy từ cột `genre` trong SQLite
    (đã tách theo dấu phẩy qua toàn bộ bảng `stories`).

    Match genre DÀI trước (VD "Cổ Đại Ngôn Tình" trước "Ngôn Tình") để tránh
    match nhầm 1 phần của genre ghép.
    """
    query_lower = query.lower()
    sorted_genres = sorted(set(known_genres), key=len, reverse=True)

    hits = []
    remaining = query_lower
    for genre in sorted_genres:
        if genre.lower() in remaining:
            hits.append(genre)
            remaining = remaining.replace(genre.lower(), " ")

    return hits


# ---------------------------------------------------------------------------
# 5. Gộp lại thành 1 object duy nhất
# ---------------------------------------------------------------------------
class ParsedQuery:
    def __init__(self, raw_query, negated_phrases, status, chapter_constraint, genre_hints):
        self.raw_query = raw_query
        self.negated_phrases = negated_phrases
        self.status = status
        self.chapter_constraint = chapter_constraint
        self.genre_hints = genre_hints

    def has_constraints(self) -> bool:
        return bool(
            self.negated_phrases or self.status or self.chapter_constraint or self.genre_hints
        )

    def __repr__(self):
        return (
            f"ParsedQuery(negated={self.negated_phrases}, status={self.status}, "
            f"chapter={self.chapter_constraint}, genre_hints={self.genre_hints})"
        )


def parse_query(query: str, known_genres: list = None) -> ParsedQuery:
    known_genres = known_genres or []
    return ParsedQuery(
        raw_query=query,
        negated_phrases=extract_negated_phrases(query),
        status=extract_status_constraint(query),
        chapter_constraint=extract_chapter_constraint(query),
        genre_hints=extract_genre_hints(query, known_genres),
    )


if __name__ == "__main__":
    # Test nhanh với các query thật trong test set (điền known_genres tạm để test)
    sample_genres = [
        "Tiên Hiệp", "Huyền Huyễn", "Đô Thị", "Ngôn Tình", "Cổ Đại Ngôn Tình",
        "Hiện Đại Ngôn Tình", "Kiếm Hiệp", "Đồng Nhân", "Dã Sử",
    ]

    test_queries = [
        "Tìm truyện tiên hiệp hơn 500 chương",
        "Tìm truyện hiện đại ngôn tình trạng thái đang ra",
        "Tìm truyện đô thị có yếu tố dị năng có hơn 300 chương",
        "Tìm truyện đã hoàn thành thể loại tiên hiệp",
        "Tìm truyện có nhân vật chính vô địch nhưng không có hệ thống và có trạng thái đang ra",
        "Tìm truyện tiên hiệp nhưng không có yếu tố harem",
        "Tìm truyện có yếu tố huyễn tưởng tu tiên và hệ thống có hơn 500 chương truyện",
    ]

    for q in test_queries:
        print(q)
        print("  ->", parse_query(q, sample_genres))
        print()