"""
Constraint Filter -- Tầng 2, bước sau Query Parser.

Nhận 1 DataFrame kết quả (đã enrich đủ cột từ SQLite: title, genre, tags,
description, status, chapters) và 1 ParsedQuery, áp các constraint cứng:
negation -> status -> chapters -> genre.

NGUYÊN TẮC AN TOÀN: mọi filter đều có fallback - nếu lọc xong ra 0 kết quả
(do parse sai, dữ liệu thiếu, hoặc constraint quá chặt), tự động BỎ QUA filter
đó và giữ nguyên kết quả trước khi lọc, kèm log cảnh báo. Tránh trả về màn
hình trống không giải thích được cho người dùng.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. NEGATION filter (đồng nghĩa + tự phủ định ngữ cảnh)
# ---------------------------------------------------------------------------
_NEGATION_SYNONYMS = {
    "harem": ["harem", "hậu cung", "nhiều nữ chính", "đa nữ chính", "tam thê tứ thiếp", "đa thê"],
    "trọng sinh": [
        "trọng sinh", "tái sinh", "sống lại", "trùng sinh",
        "sống lại một đời", "trở lại quá khứ", "quay về quá khứ", "sống lại từ đầu",
    ],
    "xuyên không": [
        "xuyên không", "xuyên việt", "xuyên qua",
        "xuyên thời không", "xuyên sang thế giới khác",
    ],
    "xuyên thư": [
        "xuyên thư", "xuyên sách", "xuyên vào sách", "xuyên vào tiểu thuyết",
        "xuyên vào truyện", "xuyên thành nhân vật",
    ],
    "tình cảm": ["tình cảm", "yêu đương", "lãng mạn", "tình yêu", "ái tình"],
    "bật hack": [
        "bật hack", "hack", "cheat", "gian lận", "trò chơi", "hệ thống",
        "bàn tay vàng", "kim thủ chi", "ngón tay vàng", "buff mạnh",
    ],
    "kim thủ chỉ": ["kim thủ chi", "kim thủ chỉ", "bàn tay vàng", "ngón tay vàng", "hệ thống"],
    "hệ thống": ["hệ thống", "auto game", "bảng", "máy gian lận", "kim thủ chi", "kim thủ chỉ"],
    "tu tiên": ["tu tiên", "tu chân", "tu luyện", "tu đạo", "tu hành", "tiên đạo", "thành tiên"],
    "võ hiệp": ["võ hiệp", "võ lâm", "giang hồ", "hiệp khách", "cao thủ võ lâm"],
    "huyền huyễn": ["huyền huyễn", "huyền ảo", "huyền bí", "thế giới huyền huyễn"],
    "dị giới": ["dị giới", "dị thế", "thế giới khác", "xuyên sang dị giới"],
    "mạt thế": ["mạt thế", "tận thế", "ngày tận thế", "hậu tận thế", "thời kỳ tận thế"],
    "zombie": ["zombie", "xác sống", "thây ma"],
    "vô địch": ["vô địch", "bất bại", "vô song", "mạnh nhất", "đệ nhất"],
    "sảng văn": ["sảng văn", "sảng", "sảng khoái", "sảng khoái văn"],
    "vả mặt": ["vả mặt", "đánh mặt", "phản kích", "phản đòn"],
    "làm ruộng": ["làm ruộng", "trồng trọt", "canh tác", "nông nghiệp"],
    "làm giàu": ["làm giàu", "kiếm tiền", "phát tài", "kinh doanh", "buôn bán", "khởi nghiệp"],
    "trinh thám": ["trinh thám", "phá án", "điều tra", "thám tử", "vụ án"],
    "hài hước": ["hài hước", "hài", "tấu hài", "hài bựa", "vui nhộn"],
    "ngọt": ["ngọt", "ngọt sủng", "sủng", "sủng ái", "ngọt ngào", "cưng chiều"],
    "ngược": ["ngược", "ngược luyến", "ngược tâm", "ngược thân", "bi thương", "đau khổ"],
}


def _expand_synonyms(phrase: str) -> list:
    return _NEGATION_SYNONYMS.get(phrase, [phrase])


def _phrase_confirmed_absent_everywhere(text: str, term: str, window: int = 20) -> bool:
    """True nếu MỌI occurrence của `term` trong text đều có 'không' ngay
    trước đó (trong phạm vi `window` ký tự) -- tức text tự khai KHÔNG có
    yếu tố này (VD '#Không hậu cung'), nên không tính là vi phạm."""
    text_lower = text.lower()
    start = 0
    found_any = False
    while True:
        idx = text_lower.find(term, start)
        if idx == -1:
            break
        found_any = True
        context_before = text_lower[max(0, idx - window):idx]
        if "không" not in context_before:
            return False
        start = idx + len(term)
    return found_any


# MỚI -- các cụm chứa từ đa nghĩa (VD "hệ thống" = nghĩa thường + nghĩa thể
# loại) mà khi xuất hiện trong description theo nghĩa THÔNG THƯỜNG thì
# KHÔNG được tính là vi phạm phủ định.
_NEGATION_FALSE_POSITIVE_PHRASES = {
    "hệ thống": [
        "hệ thống giáo dục", "hệ thống chính trị", "hệ thống pháp luật",
        "hệ thống quan lại", "hệ thống thi cử", "hệ thống chính quyền",
        "hệ thống quân sự", "hệ thống hành chính", "hệ thống pháp lý",
    ],
}


def _desc_contains_term_as_violation(desc_text: str, term: str, window: int = 20) -> bool:
    """True nếu có ÍT NHẤT 1 occurrence của `term` trong desc_text là VI PHẠM
    THẬT -- loại trừ occurrence tự-phủ-định ('không hệ thống') và occurrence
    thuộc cụm nghĩa thông thường không liên quan trope ('hệ thống giáo dục')."""
    exceptions = _NEGATION_FALSE_POSITIVE_PHRASES.get(term, [])
    start = 0
    while True:
        idx = desc_text.find(term, start)
        if idx == -1:
            return False
        context_before = desc_text[max(0, idx - window):idx]
        context_full = desc_text[max(0, idx - window): idx + len(term) + window]
        is_self_negated = "không" in context_before
        is_false_positive = any(exc in context_full for exc in exceptions)
        if not is_self_negated and not is_false_positive:
            return True
        start = idx + len(term)


def _row_violates_phrase(short_text: str, desc_text: str, phrase: str) -> bool:
    for term in _expand_synonyms(phrase):
        if term in short_text:
            return True
        if _desc_contains_term_as_violation(desc_text, term):
            return True
    return False


def filter_negation(df: pd.DataFrame, negated_phrases: list) -> pd.DataFrame:
    if not negated_phrases or df.empty:
        return df

    def _col(name):
        return df[name].fillna("") if name in df.columns else pd.Series([""] * len(df), index=df.index)

    short_text = (_col("title") + " " + _col("genre") + " " + _col("tags")).str.lower()
    desc_text = _col("description").str.lower()

    keep = []
    for s, d in zip(short_text, desc_text):
        violated = any(_row_violates_phrase(s, d, phrase) for phrase in negated_phrases)
        keep.append(not violated)

    return df[pd.Series(keep, index=df.index)]


# ---------------------------------------------------------------------------
# 2. GENRE filter (fix chính cho E1) -- hard filter, có fallback.
# ---------------------------------------------------------------------------
def filter_genre(df: pd.DataFrame, genre_hints: list) -> pd.DataFrame:
    if not genre_hints or df.empty or "genre" not in df.columns:
        return df

    hints_lower = [g.lower() for g in genre_hints]
    genre_lower = df["genre"].fillna("").str.lower()

    mask = genre_lower.apply(lambda g: any(h in g for h in hints_lower))
    filtered = df[mask]

    if filtered.empty:
        print(f"[Genre filter] Không còn kết quả nào khớp genre {genre_hints} "
              f"-> bỏ qua filter này, giữ nguyên kết quả trước lọc.")
        return df

    return filtered


# ---------------------------------------------------------------------------
# 2b. TAG filter (MỚI) -- hard filter trên cột `tags`, có fallback y hệt genre.
# ---------------------------------------------------------------------------
def filter_tags(df: pd.DataFrame, tag_hints: list) -> pd.DataFrame:
    if not tag_hints or df.empty or "tags" not in df.columns:
        return df

    hints_lower = [t.lower() for t in tag_hints]
    tags_lower = df["tags"].fillna("").str.lower()

    mask = tags_lower.apply(lambda t: any(h in t for h in hints_lower))
    filtered = df[mask]

    if filtered.empty:
        print(f"[Tag filter] Không còn kết quả nào khớp tag {tag_hints} "
              f"-> bỏ qua filter này, giữ nguyên kết quả trước lọc.")
        return df

    return filtered


# ---------------------------------------------------------------------------
# 3. STATUS filter -- hard filter, có fallback.
# ---------------------------------------------------------------------------
def filter_status(df: pd.DataFrame, status) -> pd.DataFrame:
    if not status or df.empty or "status" not in df.columns:
        return df

    mask = df["status"].fillna("").str.lower() == status.lower()
    filtered = df[mask]

    if filtered.empty:
        print(f"[Status filter] Không còn kết quả nào có status='{status}' "
              f"-> bỏ qua filter này, giữ nguyên kết quả trước lọc.")
        return df

    return filtered


# ---------------------------------------------------------------------------
# 4. CHAPTER filter -- hard filter theo operator, có fallback.
# ---------------------------------------------------------------------------
_OPS = {
    ">": lambda x, n: x > n,
    ">=": lambda x, n: x >= n,
    "<": lambda x, n: x < n,
    "<=": lambda x, n: x <= n,
}


def filter_chapters(df: pd.DataFrame, chapter_constraint) -> pd.DataFrame:
    if not chapter_constraint or df.empty or "chapters" not in df.columns:
        return df

    operator, number = chapter_constraint
    op_func = _OPS.get(operator)
    if op_func is None:
        return df

    chapters_numeric = pd.to_numeric(df["chapters"], errors="coerce")
    mask = op_func(chapters_numeric, number).fillna(False)
    filtered = df[mask]

    if filtered.empty:
        print(f"[Chapter filter] Không còn kết quả nào thoả '{operator} {number}' "
              f"-> bỏ qua filter này, giữ nguyên kết quả trước lọc.")
        return df

    return filtered


# ---------------------------------------------------------------------------
# 5. SOFT BOOST (MỚI) -- thay cho hard filter genre/tag/status/chapter.
#    Đúng theo rubric relevance: mismatch các tiêu chí này chỉ hạ mức độ
#    liên quan xuống 1, KHÔNG loại bỏ hoàn toàn (chỉ negation mới loại
#    cứng về 0). Nên thay vì exclude, ta ĐẾM số tiêu chí khớp rồi RERANK.
# ---------------------------------------------------------------------------
def _match_genre(row, genre_hints):
    if not genre_hints:
        return None
    genre_lower = str(row.get("genre") or "").lower()
    return any(h.lower() in genre_lower for h in genre_hints)


def _match_tag(row, tag_hints):
    if not tag_hints:
        return None
    tags_lower = str(row.get("tags") or "").lower()
    return any(h.lower() in tags_lower for h in tag_hints)


def _match_status(row, status):
    if not status:
        return None
    return str(row.get("status") or "").lower() == status.lower()


def _match_chapters(row, chapter_constraint):
    if not chapter_constraint:
        return None
    op, num = chapter_constraint
    op_func = _OPS.get(op)
    try:
        val = float(row.get("chapters"))
    except (TypeError, ValueError):
        return False
    return bool(op_func(val, num))


def compute_match_count(df: pd.DataFrame, parsed) -> pd.Series:
    """Đếm số tiêu chí SOFT (genre/tag/status/chapter) mỗi dòng thoả, CHỈ
    tính trên tiêu chí THẬT SỰ được hỏi trong query. Dùng để rerank."""
    if df.empty:
        return pd.Series([], dtype=int)

    counts = pd.Series(0, index=df.index)
    checks = [
        (parsed.genre_hints, lambda r: _match_genre(r, parsed.genre_hints)),
        (parsed.tag_hints, lambda r: _match_tag(r, parsed.tag_hints)),
        (parsed.status, lambda r: _match_status(r, parsed.status)),
        (parsed.chapter_constraint, lambda r: _match_chapters(r, parsed.chapter_constraint)),
    ]
    for constraint, check_fn in checks:
        if not constraint:
            continue
        matched = df.apply(check_fn, axis=1)
        counts = counts + matched.astype(int)

    return counts

# ---------------------------------------------------------------------------
# 6. Như note dưới
# ---------------------------------------------------------------------------
def apply_constraints(df: pd.DataFrame, parsed) -> pd.DataFrame:
    """
    THIẾT KẾ MỚI (đúng theo rubric relevance):
        - NEGATION: HARD FILTER duy nhất (vi phạm phủ định luôn = 0 điểm).
        - GENRE / TAG / STATUS / CHAPTER: SOFT BOOST -- không loại bỏ, chỉ
          thêm cột '_match_count' để retrieval.py rerank ưu tiên.
    """
    df = filter_negation(df, parsed.negated_phrases)
    if df.empty:
        return df

    df = df.copy()
    df["_match_count"] = compute_match_count(df, parsed)
    return df


if __name__ == "__main__":
    # Test nhanh với dữ liệu giả lập
    sample = pd.DataFrame([
        {"title": "Truyện A", "genre": "Tiên Hiệp", "tags": "Hệ Thống", "description": "", "status": "Đang ra", "chapters": 600, "score": 0.9},
        {"title": "Truyện B", "genre": "Đô Thị", "tags": "", "description": "#Không hậu cung, đơn nữ chính", "status": "Hoàn thành", "chapters": 200, "score": 0.8},
        {"title": "Ma Tu Trọng Sinh", "genre": "Tiên Hiệp", "tags": "", "description": "", "status": "Đang ra", "chapters": 100, "score": 0.7},
    ])

    from query_parser import parse_query

    q = "Tìm truyện tiên hiệp hơn 500 chương"
    parsed = parse_query(q, ["Tiên Hiệp", "Đô Thị"])
    print(q, "->", parsed)
    print(apply_constraints(sample.copy(), parsed)[["title", "chapters"]])
    print()

    q2 = "Tìm truyện tu tiên nhưng không có yếu tố trọng sinh"
    parsed2 = parse_query(q2, ["Tiên Hiệp", "Đô Thị"])
    print(q2, "->", parsed2)
    print(apply_constraints(sample.copy(), parsed2)[["title"]])