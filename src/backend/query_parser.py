"""
Query Parser -- Tầng 2, mục 1+2+3 của roadmap.

Tách 1 câu query tự nhiên thành các constraint có cấu trúc:
    - negated_phrases   : các cụm bị phủ định ("không có X")
    - status            : trạng thái được yêu cầu KHẲNG ĐỊNH (VD "Đang ra")
    - chapter_constraint: tuple (operator, số) VD (">", 500)
    - genre_hints       : các thể loại được nhắc trực tiếp trong câu
    - tag_hints         : các tag được nhắc trực tiếp trong câu (MỚI)

Đây là bước TIỀN xử lý, chạy TRƯỚC khi đưa query vào BM25/Semantic.
Không thay thế BM25/Semantic -- chỉ trích thêm thông tin có cấu trúc để
Constraint Filter (bước sau) dùng lọc/rerank kết quả.
"""

import re


# ---------------------------------------------------------------------------
# 1. NEGATION -- dời từ retrieval.py sang làm nguồn chuẩn duy nhất.
# ---------------------------------------------------------------------------
_NEGATION_FILLERS = [
    "có yếu tố ", "có ", "thuộc thể loại ", "thuộc ",
    "tập trung vào ", "yếu tố ", "thể loại ",
    "phải là ", "phải ", "mang yếu tố ", "chứa yếu tố ", "chứa ",
]

_NEGATION_PATTERN = re.compile(
    r"không\s+([^,\.]+?)(?=\s+(?:và|nhưng|tuy nhiên)\b|,|\.|$)",
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
_STATUS_KEYS_SORTED = sorted(_STATUS_MAP.keys(), key=len, reverse=True)


def extract_status_constraint(query: str):
    query_lower = query.lower()
    for key in _STATUS_KEYS_SORTED:
        idx = query_lower.find(key)
        if idx == -1:
            continue
        context_before = query_lower[max(0, idx - 15):idx]
        if "không" in context_before:
            continue
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
    m = _CHAPTER_PATTERN.search(query.lower())
    if not m:
        return None
    keyword, number = m.group(1), int(m.group(2))
    return (_CHAPTER_OP_MAP.get(keyword, ">"), number)


# ---------------------------------------------------------------------------
# 4. GENRE / TAG hints -- match trực tiếp theo tên THẬT từ DB, không hardcode.
#    Dùng CHUNG 1 cơ chế cho cả genre và tag: match cụm DÀI trước để tránh
#    match nhầm 1 phần của tên ghép (VD "Ngôn Tình" nằm trong "Cổ Đại Ngôn Tình").
# ---------------------------------------------------------------------------
def _extract_hints_from_vocab(query: str, vocab: list) -> list:
    query_lower = query.lower()
    sorted_vocab = sorted(set(vocab), key=len, reverse=True)

    hits = []
    remaining = query_lower
    for term in sorted_vocab:
        if term.lower() in remaining:
            hits.append(term)
            remaining = remaining.replace(term.lower(), " ")

    return hits


def extract_genre_hints(query: str, known_genres: list) -> list:
    """known_genres: danh sách thể loại DUY NHẤT lấy từ cột `genre` trong SQLite."""
    return _extract_hints_from_vocab(query, known_genres)


def extract_tag_hints(query: str, known_tags: list) -> list:
    """
    MỚI. known_tags: danh sách tag DUY NHẤT lấy từ cột `tags` trong SQLite.

    Lý do thêm: nhiều motif người dùng gõ (VD "huyễn tưởng tu tiên") KHÔNG
    khớp tên genre chính thức nào, nhưng lại khớp ĐÚNG 1 tag thật trong DB
    (VD tag "Huyễn Tưởng Tu Tiên"). Dùng chính dữ liệu thật thay vì đoán
    ánh xạ sẽ chính xác và đáng tin hơn.
    """
    return _extract_hints_from_vocab(query, known_tags)


# ---------------------------------------------------------------------------
# 4b. CONCEPT SYNONYMS (MỚI, bổ sung) -- chỉ dùng cho các cách paraphrase
#     KHÔNG khớp trực tiếp bất kỳ tên genre/tag thật nào (VD chỉ gõ "tu tiên"
#     mà không có "huyễn tưởng"). Mọi giá trị trong dict này đều là tên
#     genre/tag THẬT lấy từ danh sách 39 genre + danh sách tag đã xác nhận,
#     KHÔNG bịa ra khái niệm mới.
#
#     GIỚI HẠN CẦN GHI VÀO BÁO CÁO: đây là danh sách nhỏ, không bao phủ hết
#     mọi motif có thể có -- chỉ xử lý các trường hợp phổ biến nhất đã gặp
#     trong test set.
# ---------------------------------------------------------------------------
_CONCEPT_SYNONYMS = {
    "tu tiên": {
        "genres": ["Tiên Hiệp"],
        "tags": ["Huyễn Tưởng Tu Tiên", "Thần Thoại Tu Chân", "Tu Chân Văn Minh",
                  "Cổ Điển Tiên Hiệp", "Tiên Lữ Kỳ Duyên"],
    },
    "tu chân": {
        "genres": ["Tiên Hiệp"],
        "tags": ["Thần Thoại Tu Chân", "Tu Chân Văn Minh"],
    },
    "huyễn tưởng": {
        "genres": ["Huyền Huyễn"],
        "tags": ["Hắc Ám Huyễn Tưởng", "Nguyên Sinh Huyễn Tưởng",
                  "Võ Hiệp Huyễn Tưởng", "Huyễn Tưởng Tu Tiên"],
    },
    "trọng sinh": {"tags": ["Trọng Sinh"]},
    "xuyên không": {"tags": ["Xuyên Không"]},
    "xuyên sách": {"tags": ["Xuyên Sách"]},
    "xuyên nhanh": {"tags": ["Xuyên Nhanh"]},
    "hậu cung": {"tags": ["Hậu Cung", "Harem"]},
    "harem": {"tags": ["Harem", "Hậu Cung"]},
    "vô địch": {"tags": ["Vô Địch"]},
    "làm giàu": {"tags": ["Làm Giàu"]},
    "sảng văn": {"tags": ["Sảng Văn"]},
    "vả mặt": {"tags": ["Vả Mặt"]},
    "ngọt": {"tags": ["Ngọt Sủng"]},
}


def _expand_concept_synonyms(query_lower: str):
    extra_genres, extra_tags = [], []
    for key, val in _CONCEPT_SYNONYMS.items():
        if key in query_lower:
            extra_genres.extend(val.get("genres", []))
            extra_tags.extend(val.get("tags", []))
    return extra_genres, extra_tags

def _remove_contradictory_hints(hints: list, negated_phrases: list) -> list:
    """Loại khỏi genre_hints/tag_hints những hint trùng/lồng với 1 cụm đang
    bị phủ định trong CÙNG câu (VD negated=['hệ thống'] thì không được để
    genre_hints chứa 'Hệ Thống' nữa -- tự mâu thuẫn: vừa loại vừa yêu cầu)."""
    if not negated_phrases:
        return hints
    filtered = []
    for h in hints:
        h_lower = h.lower()
        contradictory = any(h_lower in neg or neg in h_lower for neg in negated_phrases)
        if not contradictory:
            filtered.append(h)
    return filtered


# ---------------------------------------------------------------------------
# 5. Gộp lại thành 1 object duy nhất
# ---------------------------------------------------------------------------
class ParsedQuery:
    def __init__(self, raw_query, negated_phrases, status, chapter_constraint,
                 genre_hints, tag_hints=None):
        self.raw_query = raw_query
        self.negated_phrases = negated_phrases
        self.status = status
        self.chapter_constraint = chapter_constraint
        self.genre_hints = genre_hints
        self.tag_hints = tag_hints or []

    def has_constraints(self) -> bool:
        return bool(
            self.negated_phrases or self.status or self.chapter_constraint
            or self.genre_hints or self.tag_hints
        )

    def __repr__(self):
        return (
            f"ParsedQuery(negated={self.negated_phrases}, status={self.status}, "
            f"chapter={self.chapter_constraint}, genre_hints={self.genre_hints}, "
            f"tag_hints={self.tag_hints})"
        )


def parse_query(query: str, known_genres: list = None, known_tags: list = None) -> ParsedQuery:
    known_genres = known_genres or []
    known_tags = known_tags or []
    query_lower = query.lower()

    negated = extract_negated_phrases(query)

    genre_hits = extract_genre_hints(query, known_genres)
    tag_hits = extract_tag_hints(query, known_tags)

    concept_genres, concept_tags = _expand_concept_synonyms(query_lower)
    for g in concept_genres:
        if g not in genre_hits:
            genre_hits.append(g)
    for t in concept_tags:
        if t not in tag_hits:
            tag_hits.append(t)

    # Mới: loại bỏ mâu thuẫn negation vs genre/tag hint
    genre_hits = _remove_contradictory_hints(genre_hits, negated)
    tag_hits = _remove_contradictory_hints(tag_hits, negated)

    return ParsedQuery(
        raw_query=query,
        negated_phrases=negated,
        status=extract_status_constraint(query),
        chapter_constraint=extract_chapter_constraint(query),
        genre_hints=genre_hits,
        tag_hints=tag_hits,
    )


if __name__ == "__main__":
    sample_genres = [
        "Tiên Hiệp", "Huyền Huyễn", "Đô Thị", "Ngôn Tình", "Cổ Đại Ngôn Tình",
        "Hiện Đại Ngôn Tình", "Kiếm Hiệp", "Đồng Nhân", "Dã Sử", "Hệ Thống",
    ]
    sample_tags = [
        "Huyễn Tưởng Tu Tiên", "Trọng Sinh", "Xuyên Không", "Harem", "Hậu Cung",
        "Vô Địch", "Làm Giàu", "Sảng Văn", "Vả Mặt", "Ngọt Sủng",
    ]

    test_queries = [
        "Tìm truyện có yếu tố huyễn tưởng tu tiên và hệ thống có hơn 500 chương truyện",
        "Tìm truyện có nhân vật chính vô địch nhưng không có hệ thống và có trạng thái đang ra",
        "Tìm truyện tu tiên nhưng không có yếu tố harem",
    ]

    for q in test_queries:
        print(q)
        print("  ->", parse_query(q, sample_genres, sample_tags))
        print()