"""
Bước 2/3 -- gộp relevance đã chấm thủ công (to_label.xlsx đã điền) trở lại
vào gold_relevance_after.xlsx.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation" / "gold_after"

GOLD_AFTER_PATH = OUTPUT_DIR / "gold_relevance_after.xlsx"
TO_LABEL_PATH = OUTPUT_DIR / "to_label.xlsx"
FINAL_PATH = OUTPUT_DIR / "gold_relevance_after_final.xlsx"


def main():
    gold_after = pd.read_excel(GOLD_AFTER_PATH)
    labeled = pd.read_excel(TO_LABEL_PATH)

    gold_after["query_id"] = gold_after["query_id"].astype(str).str.strip()
    labeled["query_id"] = labeled["query_id"].astype(str).str.strip()
    gold_after["story_id"] = gold_after["story_id"].astype(int)
    labeled["story_id"] = labeled["story_id"].astype(int)

    if labeled["relevance"].isna().any():
        n = labeled["relevance"].isna().sum()
        print(f"⚠️ Còn {n} dòng CHƯA điền relevance trong to_label.xlsx -- "
              f"điền đủ (0/1/2) rồi chạy lại script này.")
        return

    label_map = labeled.set_index(["query_id", "story_id"])["relevance"]

    def _fill(row):
        if pd.notna(row["relevance"]):
            return row["relevance"]
        return label_map.get((row["query_id"], row["story_id"]), row["relevance"])

    gold_after["relevance"] = gold_after.apply(_fill, axis=1)
    gold_after["relevance"] = pd.to_numeric(gold_after["relevance"], errors="coerce")

    still_na = gold_after["relevance"].isna().sum()
    if still_na:
        print(f"⚠️ Vẫn còn {still_na} dòng thiếu relevance sau khi gộp -- kiểm tra lại.")

    gold_after.to_excel(FINAL_PATH, index=False)
    print(f"Đã lưu bảng cuối cùng -> {FINAL_PATH}")


if __name__ == "__main__":
    main()