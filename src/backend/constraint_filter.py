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


def _row_violates_phrase(short_text: str, desc_text: str, phrase: str) -> bool:
    for term in _expand_synonyms(phrase):
        if term in short_text:
            return True
        if term in desc_text and not _phrase_confirmed_absent_everywhere(desc_text, term):
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
# 5. Áp toàn bộ constraint theo thứ tự: negation -> status -> chapters -> genre
#    (negation trước tiên vì đây là điều kiện loại trừ rõ ràng nhất)
# ---------------------------------------------------------------------------
def apply_constraints(df: pd.DataFrame, parsed) -> pd.DataFrame:
    df = filter_negation(df, parsed.negated_phrases)
    df = filter_status(df, parsed.status)
    df = filter_chapters(df, parsed.chapter_constraint)
    df = filter_genre(df, parsed.genre_hints)
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