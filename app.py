"""
Giao diện Streamlit đơn giản cho hệ thống Tìm kiếm Truyện.

Chạy:
    streamlit run app.py

Vị trí ở GỐC dự án (ngang hàng với thư mục src/, database/, models/).
"""

import os
import sys

import streamlit as st

# --- Nhớ thêm src/backend vào sys.path để import được retrieval.py ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(CURRENT_DIR, "src", "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

import retrieval  # noqa: E402


st.set_page_config(page_title="Tìm kiếm Truyện", page_icon="🔍", layout="wide")

st.title("🔍 Hệ thống Tìm kiếm Truyện")
st.caption("So sánh trực tiếp giữa BM25 và Semantic Search (FAISS)")

# --- Load model semantic 1 lần, có spinner báo cho người dùng biết ---
@st.cache_resource(show_spinner="Đang tải mô hình Semantic (chỉ lần đầu)...")
def get_semantic_resources():
    return retrieval._load_semantic_once()


with st.sidebar:
    st.header("Tuỳ chọn")
    method = st.radio("Phương pháp tìm kiếm", ["semantic", "bm25"], index=0)
    top_k = st.slider("Số kết quả (Top-K)", min_value=1, max_value=20, value=10)

# Preload model semantic ngay khi chọn, tránh giật khi bấm Tìm kiếm
if method == "semantic":
    get_semantic_resources()

query = st.text_input(
    "Nhập truy vấn của bạn:",
    placeholder="VD: truyện tu tiên có hệ thống, không harem",
)
search_clicked = st.button("🔎 Tìm kiếm", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Vui lòng nhập truy vấn.")
    else:
        with st.spinner("Đang tìm kiếm..."):
            results = retrieval.search(query, method=method, top_k=top_k)

        st.markdown(f"### Kết quả cho: *“{query}”* — phương pháp **{method.upper()}**")

        if results.empty:
            st.info("Không tìm thấy kết quả phù hợp.")

        for rank, row in enumerate(results.itertuples(), start=1):
            with st.container(border=True):
                title = getattr(row, "title", "N/A")
                url = getattr(row, "url", None)
                genre = getattr(row, "genre", "N/A")
                status = getattr(row, "status", "N/A")
                chapters = getattr(row, "chapters", "N/A")
                views = getattr(row, "views", "N/A")
                score = getattr(row, "score", 0.0)

                header = f"**{rank}. {title}**"
                if url:
                    header = f"**{rank}. [{title}]({url})**"
                st.markdown(header)

                c1, c2, c3, c4 = st.columns(4)
                c1.caption(f"📖 Thể loại: {genre}")
                c2.caption(f"📌 Trạng thái: {status}")
                c3.caption(f"📚 Chương: {chapters} | 👁 Lượt xem: {views}")
                c4.caption(f"⭐ Score: {score:.4f}")

st.divider()
st.caption(
    "Backend: BM25 (rank_bm25) + Semantic Search (SentenceTransformer + FAISS) "
    "+ SQLite metadata."
)