"""
Chạy SAU KHI đã điền xong cột "error_type" trong error_candidates.xlsx.
Tổng hợp thành bảng thống kê: số lỗi theo từng loại (E1-E8) x method,
và theo query_type (genre/tag/multi_condition/semantic/negative) x error_type
-- cái này quan trọng để trả lời "sai vì nguyên nhân gì" trong báo cáo.

Chạy:
    python src/evaluation/summarize_errors.py
"""

import os

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation", "error_candidates.xlsx")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation", "error_analysis_summary.xlsx")


def run():
    df = pd.read_excel(INPUT_PATH, sheet_name="error_candidates")

    unfilled = df["error_type"].isna() | (df["error_type"].astype(str).str.strip() == "")
    if unfilled.any():
        print(f"[Cảnh báo] Còn {unfilled.sum()}/{len(df)} dòng chưa điền 'error_type'. "
              f"Các dòng này sẽ bị loại khỏi bảng thống kê bên dưới.")

    df = df[~unfilled].copy()
    df["error_type"] = df["error_type"].astype(str).str.strip().str.upper()

    # --- Số lỗi theo error_type x method ---
    by_error_method = (
        df.groupby(["error_type", "method"]).size()
        .reset_index(name="count")
        .sort_values(["error_type", "method"])
    )

    # --- Số lỗi theo query_type x error_type -> loại query nào gây lỗi gì ---
    by_querytype_error = (
        df.groupby(["query_type", "error_type"]).size()
        .reset_index(name="count")
        .sort_values(["query_type", "count"], ascending=[True, False])
    )

    # --- Tỉ lệ lỗi theo method (trên tổng số dòng lỗi, không phải trên toàn bộ Top-10) ---
    method_share = (
        df.groupby("method").size().reset_index(name="count")
    )
    method_share["share"] = method_share["count"] / method_share["count"].sum()

    # --- Riêng 2 lỗi trọng tâm thầy hay hỏi: E1 và E2 ---
    e1_e2_examples = df[df["error_type"].isin(["E1", "E2"])][
        ["query_id", "query", "query_type", "method", "title", "genre", "tags", "notes", "analyst_note"]
    ]

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        by_error_method.to_excel(writer, sheet_name="error_by_type_method", index=False)
        by_querytype_error.to_excel(writer, sheet_name="error_by_querytype", index=False)
        method_share.to_excel(writer, sheet_name="error_share_by_method", index=False)
        e1_e2_examples.to_excel(writer, sheet_name="E1_E2_examples", index=False)

    print(f"Đã xuất tổng hợp ra: {OUTPUT_PATH}\n")

    print("=== Số lỗi theo loại (E1-E8) x method ===")
    print(by_error_method.to_string(index=False))

    print("\n=== Tỉ lệ lỗi theo method (trong tổng số lỗi) ===")
    print(method_share.to_string(index=False))

    print(f"\n=== Số ví dụ E1 (genre mismatch) + E2 (negative violation): {len(e1_e2_examples)} ===")
    print("(Chi tiết xem sheet 'E1_E2_examples')")


if __name__ == "__main__":
    run()