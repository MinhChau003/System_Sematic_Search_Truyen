"""
Bước Error Analysis - Trích các candidate SAI (relevance = 0) từ gold_relevance.xlsx
cho từng method (BM25 / Semantic riêng biệt), để con người phân loại theo taxonomy E1-E8.

Input:
    results/evaluation/gold_relevance.xlsx
    (đã đủ dữ liệu: query_id, query, query_type, bm25_rank, semantic_rank,
     retrieved_by, relevance... không cần merge thêm từ test_queries.xlsx)

Output:
    results/evaluation/error_candidates.xlsx
        - sheet "error_candidates": các dòng lỗi, cột "error_type" để trống cho bạn điền
        - sheet "taxonomy": bảng chú thích E1-E8

Chạy:
    python src/evaluation/extract_errors.py
"""

import os

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

GOLD_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation", "gold_relevance.xlsx")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation", "error_candidates.xlsx")

ERROR_TAXONOMY = {
    "E1": "Genre mismatch (thể loại không khớp, VD: hỏi đô thị ra huyền huyễn)",
    "E2": "Negative constraint violation (hỏi 'không X' nhưng vẫn có X)",
    "E3": "Tag mismatch (tag/yếu tố không khớp, VD: hỏi 'có hệ thống' nhưng truyện không có)",
    "E4": "Description semantic mismatch (nội dung mô tả không liên quan dù embedding gần)",
    "E5": "Status constraint mismatch (hỏi trạng thái cụ thể nhưng sai, VD: 'đang ra' ra 'hoàn thành')",
    "E6": "Chapter constraint mismatch (hỏi số chương cụ thể nhưng sai, VD: '>500 chương' ra truyện ít chương)",
    "E7": "Mixed constraint (sai do kết hợp nhiều điều kiện cùng lúc, khó quy về 1 loại trên)",
    "E8": "Other (lỗi khác, không thuộc các loại trên)",
}


def build_error_rows(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Lọc các dòng mà `method` có trả về (rank không NaN) nhưng relevance = 0."""
    rank_col = f"{method}_rank"
    score_col = f"{method}_score"

    subset = df[df[rank_col].notna() & (df["relevance"] == 0)].copy()

    subset["method"] = method
    subset["rank"] = subset[rank_col]
    subset["score"] = subset[score_col]

    subset["error_type"] = ""      # <-- bạn điền E1..E8 vào đây
    subset["analyst_note"] = ""    # <-- ghi chú thêm nếu cần (VD: giải thích cụ thể)

    keep_cols = [
        "query_id", "query", "query_type", "method", "rank", "score",
        "story_id", "title", "genre", "tags", "description", "status",
        "chapters", "url", "relevance", "retrieved_by",
        "evaluator", "notes", "error_type", "analyst_note",
    ]
    keep_cols = [c for c in keep_cols if c in subset.columns]

    return subset[keep_cols]


def run():
    df = pd.read_excel(GOLD_PATH)

    required_cols = {"bm25_rank", "semantic_rank", "relevance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"gold_relevance.xlsx thiếu cột bắt buộc: {missing}")

    bm25_errors = build_error_rows(df, "bm25")
    semantic_errors = build_error_rows(df, "semantic")

    result = pd.concat([bm25_errors, semantic_errors], ignore_index=True)
    result = result.sort_values(["query_id", "method", "rank"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        result.to_excel(writer, sheet_name="error_candidates", index=False)

        taxonomy_df = pd.DataFrame(
            [{"code": k, "meaning": v} for k, v in ERROR_TAXONOMY.items()]
        )
        taxonomy_df.to_excel(writer, sheet_name="taxonomy", index=False)

    print(f"Tổng số dòng lỗi (relevance=0): {len(result)}")
    print(f"  - BM25     : {len(bm25_errors)}")
    print(f"  - Semantic : {len(semantic_errors)}")
    print(f"Đã xuất: {OUTPUT_PATH}")
    print("=> Mở file, xem sheet 'taxonomy' rồi điền cột 'error_type' (E1-E8) "
          "cho từng dòng trong sheet 'error_candidates'.")


if __name__ == "__main__":
    run()