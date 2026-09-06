
import os
import sys
import time
import html
import streamlit as st

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.backend.retrieval import (
    search,
    parse_query,
    _load_known_genres_once,
    _load_known_tags_once,
)

st.set_page_config(
    page_title="Thiên Cơ Truyện Các",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# STYLE — Stitch-inspired
# =========================
st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 85% 0%, rgba(18,95,130,.15), transparent 28%),
      #080d18;
    color:#e8eef8;
}
[data-testid="stSidebar"] {
    background:#0a101d;
    border-right:1px solid #1a2538;
}
.hero {
    background:linear-gradient(110deg,#1c2234,#0c3040);
    border:1px solid #1d3448;
    border-radius:9px;
    padding:18px 20px;
    margin-bottom:16px;
}
.hero h1 {margin:0;color:#dff8ff;font-size:25px;}
.hero p {margin:5px 0 0;color:#98a8bc;font-size:12px;}
.panel {
    background:#111827;
    border:1px solid #202b3e;
    border-radius:9px;
    padding:16px;
    margin-bottom:15px;
}
.small-label {
    color:#8fa1b8;
    font-size:10px;
    font-weight:800;
    letter-spacing:.4px;
    text-transform:uppercase;
}
.chips {display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;}
.chip {
    display:inline-block;
    color:#a8d9e6;
    background:#172738;
    border:1px solid #244054;
    border-radius:12px;
    padding:4px 9px;
    font-size:10px;
}
.neg {
    color:#ffb6b6 !important;
    background:#321b25 !important;
    border-color:#552734 !important;
}
.ok {
    color:#8ff0d1 !important;
    background:#122b2a !important;
    border-color:#1d4c45 !important;
}
.stat {
    background:#111827;
    border:1px solid #202b3e;
    border-radius:8px;
    padding:13px;
    min-height:92px;
}
.stat-label {
    color:#8e9caf;
    font-size:9px;
    font-weight:800;
    text-transform:uppercase;
}
.stat-value {
    color:#e7f6ff;
    font-size:22px;
    font-weight:900;
    margin-top:7px;
}
.cyan {color:#20d8f2 !important;}
.stat-sub {color:#6f829b;font-size:9px;margin-top:3px;}
.result-card {
    background:#111827;
    border:1px solid #202b3e;
    border-radius:9px;
    padding:15px;
    margin-bottom:11px;
}
.rank {color:#18d9f4;font-size:10px;font-weight:900;}
.title {color:#dce9f7;font-size:16px;font-weight:800;}
.author {color:#8c9aaf;font-size:10px;margin-top:3px;}
.score-box {
    background:#0b1220;
    border:1px solid #1d2a3d;
    border-radius:7px;
    padding:8px 10px;
    text-align:right;
}
.score-label {color:#77869d;font-size:8px;text-transform:uppercase;}
.score {color:#21d8f2;font-size:19px;font-weight:900;}
.meta {color:#93a3b8;font-size:10px;line-height:1.7;margin-top:11px;}
.desc {color:#bac6d6;font-size:11px;line-height:1.55;margin-top:8px;}
.empty {
    background:#101827;
    border:1px dashed #2b3950;
    border-radius:9px;
    padding:30px;
    text-align:center;
    color:#8492a7;
}
.side-title {
    color:#67e8f9;
    font-size:10px;
    font-weight:900;
    letter-spacing:.5px;
    margin:18px 0 7px;
}
.side-note {
    color:#8391a7;
    background:#101827;
    border:1px solid #1e293b;
    border-radius:8px;
    padding:10px;
    font-size:10px;
    line-height:1.55;
    margin-top:12px;
}
div[data-testid="stButton"] > button {
    border:1px solid #12cce9;
    background:linear-gradient(100deg,#0bd7ed,#089bc7);
    color:#03131a;
    font-weight:900;
    border-radius:6px;
}
.stTextInput input {
    background:#090f1b !important;
    color:#e8eef8 !important;
    border:1px solid #202b3e !important;
}
</style>
""", unsafe_allow_html=True)


def text(v, default=""):
    if v is None:
        return default
    try:
        if v != v:
            return default
    except Exception:
        pass
    return str(v).strip()


def num(v):
    try:
        return f"{int(float(v)):,}"
    except Exception:
        return "—"


def truncate(v, n=300):
    s = text(v)
    return s if len(s) <= n else s[:n].rstrip() + "..."


def chip(label, cls=""):
    return f'<span class="chip {cls}">{html.escape(text(label))}</span>'


def render_constraints(parsed):
    items = []
    items += [chip(f"Thể loại: {x}") for x in parsed.genre_hints]
    items += [chip(f"Tag: {x}") for x in parsed.tag_hints]
    items += [chip(f"Không: {x}", "neg") for x in parsed.negated_phrases]
    if parsed.status:
        items.append(chip(f"Trạng thái: {parsed.status}", "ok"))
    if parsed.chapter_constraint:
        op, n = parsed.chapter_constraint
        items.append(chip(f"Chương {op} {n}", "ok"))
    return "".join(items)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:9px;">
      <div style="font-size:25px;">🌌</div>
      <div style="font-weight:900;color:#dbeafe;">
        Thiên Cơ Truyện Các
        <span style="font-size:9px;color:#9fb0c8;background:#1a2435;
        padding:3px 6px;border-radius:5px;">v2.0</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-title">ĐIỀU HƯỚNG TÀNG KINH</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "page",
        ["Tra Cứu Tàng Kinh", "Bảng Xếp Hạng & Thống Kê", "Về Hệ Thống IR"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="side-title">THIẾT LẬP THUẬT TOÁN</div>',
                unsafe_allow_html=True)

    method_label = st.radio(
        "method",
        ["Semantic Search (Vector FAISS)", "BM25 (Từ khóa chuẩn)"],
        index=0,
        label_visibility="collapsed",
    )
    method = "semantic" if method_label.startswith("Semantic") else "bm25"

    top_k = st.slider("Top-K kết quả", 5, 20, 10, 1)

    st.markdown("""
    <div class="side-note">
      <b>HẠ TẦNG TÀNG KINH</b><br>
      Semantic: PhoBERT + FAISS<br>
      BM25: lexical retrieval<br>
      SQLite: metadata + constraint filtering
    </div>
    <div class="side-note">
      <b>PIPELINE</b><br>
      Query → Parser → Retrieval → SQLite → Filter → Rerank → Top-K
    </div>
    """, unsafe_allow_html=True)


# =========================
# SEARCH PAGE
# =========================
if page == "Tra Cứu Tàng Kinh":

    st.markdown("""
    <div class="hero">
      <h1>🌌 Thiên Cơ Truyện Các</h1>
      <p>
        Hệ thống khám phá tiểu thuyết bằng ngôn ngữ tự nhiên
        • PhoBERT & FAISS Semantic Search • BM25 Retrieval
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="small-label">NHẬP Ý NIỆM HOẶC MÔ TẢ CÂU CHUYỆN BẠN MUỐN TÌM</div>',
        unsafe_allow_html=True,
    )

    if "query_input" not in st.session_state:
        st.session_state.query_input = (
            "Tìm truyện tu tiên có hệ thống, không harem, trên 500 chương"
        )

    query = st.text_input(
        "query",
        key="query_input",
        label_visibility="collapsed",
        placeholder="Ví dụ: truyện tiên hiệp không có hệ thống, hơn 500 chương...",
    )

    st.markdown(
        '<div class="small-label" style="margin-top:9px;">Ý NIỆM GỢI Ý</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        "Tu Tiên Trọng Sinh",
        "Hệ Thống Vô Địch",
        "Đô Thị Dị Năng",
        "Phàm Nhân Lưu",
        "Cổ Đại Ngôn Tình",
    ]

    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        with col:
            if st.button(s, key="suggest_" + s):
                st.session_state.query_input = s
                st.rerun()

    clicked = st.button("🔍  Tra Cứu Tàng Kinh", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if clicked:
        st.session_state.last_query = query.strip()

    active_query = st.session_state.get("last_query", "")

    if active_query:
        try:
            genres = _load_known_genres_once()
            tags = _load_known_tags_once()
            parsed = parse_query(active_query, genres, tags)
        except Exception as exc:
            st.error(f"Không thể phân tích truy vấn: {exc}")
            st.stop()

        constraint_html = render_constraints(parsed)

        if constraint_html:
            st.markdown(
                f'<div class="small-label">ĐIỀU KIỆN HỆ THỐNG ĐÃ PHÂN TÍCH</div>'
                f'<div class="chips">{constraint_html}</div>',
                unsafe_allow_html=True,
            )

        start = time.perf_counter()

        try:
            results = search(
                active_query,
                method=method,
                top_k=top_k,
                verbose=False,
            )
        except Exception as exc:
            st.error(f"Lỗi khi tìm kiếm: {exc}")
            st.stop()

        elapsed = (time.perf_counter() - start) * 1000
        count = len(results)

        try:
            best = float(results["score"].iloc[0]) if count else 0.0
        except Exception:
            best = 0.0

        a, b, c, d = st.columns(4)

        with a:
            st.markdown(
                f'<div class="stat"><div class="stat-label">Tổng số kết quả</div>'
                f'<div class="stat-value">{count} bộ</div>'
                f'<div class="stat-sub">Top-K = {top_k}</div></div>',
                unsafe_allow_html=True,
            )

        with b:
            st.markdown(
                f'<div class="stat"><div class="stat-label">Thời gian xử lý</div>'
                f'<div class="stat-value cyan">{elapsed:.0f} ms</div>'
                f'<div class="stat-sub">Query → Retrieval → Filter</div></div>',
                unsafe_allow_html=True,
            )

        with c:
            st.markdown(
                f'<div class="stat"><div class="stat-label">Điểm cao nhất</div>'
                f'<div class="stat-value cyan">{best:.4f}</div>'
                f'<div class="stat-sub">'
                f'{"Cosine / Semantic" if method == "semantic" else "BM25 score"}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with d:
            name = "Semantic Search" if method == "semantic" else "BM25"
            engine = "PhoBERT + FAISS" if method == "semantic" else "BM25 Index"
            st.markdown(
                f'<div class="stat"><div class="stat-label">Thuật toán áp dụng</div>'
                f'<div class="stat-value" style="font-size:17px;">{name}</div>'
                f'<div class="stat-sub">{engine}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="border-bottom:1px solid #1b2637;padding:11px 3px 8px;'
            f'color:#b7c8dc;font-size:12px;font-weight:900;">'
            f'📚 KẾT QUẢ XẾP HẠNG ({count})</div>',
            unsafe_allow_html=True,
        )

        if results.empty:
            st.markdown(
                '<div class="empty">🔭<br><b>Chưa tìm thấy kết quả phù hợp</b><br>'
                'Thử nới lỏng điều kiện hoặc đổi phương thức tìm kiếm.</div>',
                unsafe_allow_html=True,
            )
        else:
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                title = text(row.get("title"), "Không có tên truyện")
                author = text(row.get("author"), "Không rõ tác giả")
                genre = text(row.get("genre"), "Chưa cập nhật")
                tags_text = text(row.get("tags"))
                status = text(row.get("status"), "Chưa rõ")
                chapters = num(row.get("chapters"))
                views = num(row.get("views"))
                desc = truncate(row.get("description"))
                url = text(row.get("url"))
                story_id = text(row.get("id"), "—")

                try:
                    score = float(row.get("score", 0))
                except Exception:
                    score = 0.0

                tag_html = ""
                if tags_text:
                    tag_html = "".join(
                        chip(x.strip())
                        for x in tags_text.split(",")
                        if x.strip()
                    )[:2500]

                st.markdown(
                    f"""
                    <div class="result-card">
                      <div style="display:flex;justify-content:space-between;gap:14px;">
                        <div style="flex:1;">
                          <div>
                            <span class="rank">#{rank:02d}</span>
                            <span class="title">{html.escape(title)}</span>
                          </div>
                          <div class="author">Tác giả: {html.escape(author)}</div>
                        </div>
                        <div class="score-box">
                          <div class="score-label">ĐIỂM HỆ THỐNG</div>
                          <div class="score">{score:.4f}</div>
                        </div>
                      </div>

                      <div class="meta">
                        🏷 <b>Thể loại:</b> {html.escape(genre)}
                        &nbsp; • &nbsp;
                        📖 <b>Chương:</b> {chapters}
                        &nbsp; • &nbsp;
                        👁 <b>Lượt xem:</b> {views}
                        &nbsp; • &nbsp;
                        ● {html.escape(status)}
                      </div>

                      <div class="chips">{tag_html}</div>

                      <div class="desc">
                        {html.escape(desc) if desc else "Chưa có mô tả."}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(
                    f"🔎 Xem chi tiết & phân tích ngữ nghĩa — ID {story_id}"
                ):
                    x, y = st.columns([2, 1])
                    with x:
                        st.write("**Tên truyện:**", title)
                        st.write("**Tác giả:**", author)
                        st.write("**Thể loại:**", genre)
                        st.write("**Tags:**", tags_text or "—")
                        st.write("**Trạng thái:**", status)
                        st.write("**Số chương:**", chapters)
                    with y:
                        st.metric("Score", f"{score:.4f}")
                        st.caption(
                            "Score là điểm retrieval, không phải xác suất."
                        )
                        if url:
                            st.link_button("🌐 Mở trang truyện", url)

# =========================
# STATS
# =========================
elif page == "Bảng Xếp Hạng & Thống Kê":
    st.markdown("""
    <div class="hero">
      <h1>📊 Bảng Xếp Hạng & Thống Kê</h1>
      <p>Khu vực quan sát dữ liệu và cấu hình hệ thống IR.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        genres = _load_known_genres_once()
        tags = _load_known_tags_once()

        a, b, c = st.columns(3)
        a.metric("Số thể loại", len(genres))
        b.metric("Số tag", len(tags))
        c.metric("Retrieval", "Semantic + BM25")

        st.subheader("Danh sách thể loại từ SQLite")
        st.dataframe(
            {"Genre": sorted(genres)},
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.error(f"Không thể tải thống kê: {exc}")

# =========================
# ABOUT
# =========================
else:
    st.markdown("""
    <div class="hero">
      <h1>ℹ️ Về Hệ Thống IR</h1>
      <p>Kiến trúc của hệ thống tìm truyện.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
### Pipeline hiện tại

```text
Query
  ↓
Query Parser
  ↓
BM25 / Semantic Retrieval
  ↓
SQLite Enrich
  ↓
Constraint Filter
  ↓
Rerank
  ↓
Top-K
  ↓
Streamlit
```

### Constraint

- **Negation:** hard filter — loại kết quả vi phạm phủ định.
- **Genre:** constraint dùng cho rerank.
- **Tag:** constraint dùng cho rerank.
- **Status:** constraint dùng cho rerank.
- **Chapter:** hỗ trợ `>`, `>=`, `<`, `<=`.

### Retrieval

**Semantic Search:** PhoBERT + FAISS.

**BM25:** lexical retrieval, dùng làm baseline và phương thức tìm kiếm từ khóa.

> Điểm `score` hiển thị trên giao diện là điểm retrieval của thuật toán,
> không phải xác suất độ liên quan.
""")
