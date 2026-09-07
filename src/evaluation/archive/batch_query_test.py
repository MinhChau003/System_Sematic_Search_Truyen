"""
Test hàng loạt: chạy 20 query chính thức + 1 bộ query biên (edge case) qua
pipeline mới, xuất ra Excel để soát nhanh bằng mắt. Tự động gắn CỜ CẢNH BÁO
cho các trường hợp đáng ngờ (0 kết quả, filter không tác dụng gì, v.v.)
để không phải đọc hết từng dòng.
"""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "src" / "backend"
sys.path.append(str(BACKEND_DIR))

from query_parser import parse_query  # noqa: E402
import retrieval                      # noqa: E402

TEST_QUERIES_PATH = PROJECT_ROOT / "data" / "queries" / "test_queries.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "results" / "evaluation" / "batch_query_test_report.xlsx"

TOP_K = 10

# --- Bộ query biên (edge case), tự soạn để lộ các lỗi tiềm ẩn --------------
EDGE_CASE_QUERIES = [
    # Phủ định kép trong 1 câu
    "Tìm truyện không có harem và không có hệ thống",
    # Phủ định + đa điều kiện khác
    "Tìm truyện tiên hiệp không trọng sinh, đã hoàn thành, trên 800 chương",
    # Không nhắc genre/tag nào cả -- chỉ có chapter + status
    "Tìm truyện trên 2000 chương đang ra",
    # Chỉ có 1 điều kiện chapter ở biên (bằng đúng ngưỡng thường gặp)
    "Tìm truyện có đúng 500 chương",
    # Câu rất ngắn, không structure rõ
    "tu tiên",
    "harem",
    # Câu rất dài, nhiều mệnh đề
    "Tìm truyện có yếu tố tiên hiệp, không có harem, không có hệ thống, "
    "đã hoàn thành, có hơn 1000 chương và nhân vật chính vô địch",
    # Genre viết thường, không dấu (test robustness)
    "tim truyen tien hiep dang ra",
    # Genre gõ sai chính tả nhẹ
    "Tìm truyện tiên hiêp có hệ thông",  # thiếu dấu ở "hiêp", "thông"
    # Phủ định 1 cụm không nằm trong _NEGATION_SYNONYMS
    "Tìm truyện không có yếu tố ma pháp",
    # Trạng thái phủ định (câu khó cho status parser)
    "Tìm truyện không phải đang ra",
    # Chapter với từ khoá ít phổ biến
    "Tìm truyện tối thiểu 300 chương thuộc thể loại đô thị",
    # Tag hiếm, kết hợp genre
    "Tìm truyện thể loại tiên hiệp có tag báo thù",
    # Rỗng / vô nghĩa (edge case cực đoan)
    "truyện hay",
]


def _run_one(query: str, method: str, source: str):
    known_genres = retrieval._load_known_genres_once()
    known_tags = retrieval._load_known_tags_once()
    parsed = parse_query(query, known_genres, known_tags)

    start = time.time()
    try:
        results = retrieval.search(query, method=method, top_k=TOP_K, verbose=False)
        error = ""
    except Exception as e:
        results = pd.DataFrame()
        error = str(e)
    elapsed = time.time() - start

    flags = []
    if error:
        flags.append(f"LỖI THỰC THI: {error}")
    if results.empty and not error:
        flags.append("0 KẾT QUẢ")
    if parsed.has_constraints() and not results.empty and "score" in results.columns:
        # Không thể truy lại match_count sau khi retrieval.search() đã drop cột này,
        # nên chỉ cảnh báo dựa trên các dấu hiệu gián tiếp còn lại.
        pass
    if parsed.negated_phrases and not results.empty:
        # Kiểm tra nhanh xem top-1 có vô tình chứa cụm bị phủ định không
        # (double-check độc lập với logic filter, để bắt sót nếu có).
        top = results.iloc[0]
        short_text = f"{top.get('title','')} {top.get('genre','')} {top.get('tags','')}".lower()
        for phrase in parsed.negated_phrases:
            if phrase in short_text:
                flags.append(f"NGHI VẤN: top-1 có thể vẫn chứa cụm bị phủ định '{phrase}'")

    top_title = results.iloc[0].get("title") if not results.empty else ""
    top_genre = results.iloc[0].get("genre") if not results.empty else ""
    top_score = results.iloc[0].get("score") if not results.empty else None

    return {
        "source": source,
        "query": query,
        "method": method,
        "parsed": repr(parsed),
        "num_results": len(results),
        "top_title": top_title,
        "top_genre": top_genre,
        "top_score": top_score,
        "elapsed_sec": round(elapsed, 2),
        "flags": " | ".join(flags),
    }


def main():
    rows = []

    # 1. 20 query chính thức
    if TEST_QUERIES_PATH.exists():
        test_df = pd.read_excel(TEST_QUERIES_PATH)
        for _, row in test_df.iterrows():
            query = row["query"]
            for method in ("bm25", "semantic"):
                rows.append(_run_one(query, method, source=f"official_{row['query_id']}"))
    else:
        print(f"⚠️ Không tìm thấy {TEST_QUERIES_PATH}, bỏ qua 20 query chính thức.")

    # 2. Bộ query biên
    for query in EDGE_CASE_QUERIES:
        for method in ("bm25", "semantic"):
            rows.append(_run_one(query, method, source="edge_case"))

    report_df = pd.DataFrame(rows)

    flagged = report_df[report_df["flags"] != ""]
    print(f"Tổng số lượt chạy: {len(report_df)}")
    print(f"Số lượt CÓ CỜ CẢNH BÁO: {len(flagged)}")
    if not flagged.empty:
        print("\n--- CHI TIẾT CÁC LƯỢT ĐÁNG NGỜ ---")
        for _, r in flagged.iterrows():
            print(f"[{r['source']}][{r['method']}] {r['query']}")
            print(f"    -> {r['flags']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\nĐã lưu báo cáo đầy đủ -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()