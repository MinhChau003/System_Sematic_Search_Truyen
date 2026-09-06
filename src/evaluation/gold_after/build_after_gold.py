"""
Bước 1/3 (Hướng A) -- tạo Top-10 MỚI (pipeline có Query Parser + Constraint
Filter) cho 20 test query chính thức, so khớp với gold_relevance.xlsx cũ:

    - Truyện đã có relevance (từng ở Top-10 cũ) -> giữ nguyên relevance.
    - Truyện MỚI xuất hiện (chưa từng chấm) -> relevance để TRỐNG, xuất
      riêng ra to_label.xlsx để chấm thủ công.

Output:
    results/evaluation/gold_relevance_after.xlsx  (bảng đầy đủ)
    results/evaluation/to_label.xlsx               (chỉ các dòng cần chấm)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "src" / "backend"
sys.path.append(str(BACKEND_DIR))
import retrieval  # noqa: E402

TEST_QUERIES_PATH = PROJECT_ROOT / "data" / "queries" / "test_queries.xlsx"
GOLD_PATH = PROJECT_ROOT / "results" / "evaluation" / "gold_relevance.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation" / "gold_after"
OUTPUT_GOLD_AFTER = OUTPUT_DIR / "gold_relevance_after.xlsx"
OUTPUT_TO_LABEL = OUTPUT_DIR / "to_label.xlsx"

TOP_K = 10
META_COLS = ["title", "genre", "tags", "description", "status", "chapters", "url"]


def _get_new_rankings(query_id, query, method):
    df = retrieval.search(query, method=method, top_k=TOP_K, verbose=False)
    df = df.reset_index(drop=True)
    df["rank_after"] = df.index + 1
    df["score_after"] = df["score"]
    df["query_id"] = query_id
    return df[["query_id", "id", "rank_after", "score_after"] + META_COLS]


def main():
    test_df = pd.read_excel(TEST_QUERIES_PATH)
    gold_df = pd.read_excel(GOLD_PATH)

    test_df["query_id"] = test_df["query_id"].astype(str).str.strip()
    gold_df["query_id"] = gold_df["query_id"].astype(str).str.strip()

    all_after_rows = []

    for _, row in test_df.iterrows():
        qid, query = row["query_id"], row["query"]

        bm25_new = _get_new_rankings(qid, query, "bm25").rename(
            columns={"rank_after": "bm25_rank_after", "score_after": "bm25_score_after"}
        )
        sem_new = _get_new_rankings(qid, query, "semantic").rename(
            columns={"rank_after": "semantic_rank_after", "score_after": "semantic_score_after"}
        )

        merged_new = pd.merge(
            bm25_new[["query_id", "id", "bm25_rank_after", "bm25_score_after"] + META_COLS],
            sem_new[["query_id", "id", "semantic_rank_after", "semantic_score_after"]],
            on=["query_id", "id"], how="outer",
        )

        missing_meta = merged_new[META_COLS].isna().any(axis=1)
        if missing_meta.any():
            meta_lookup = pd.concat([bm25_new[["id"] + META_COLS], sem_new[["id"] + META_COLS]])
            meta_lookup = meta_lookup.drop_duplicates("id").set_index("id")
            for col in META_COLS:
                merged_new.loc[missing_meta, col] = merged_new.loc[missing_meta, "id"].map(meta_lookup[col])

        all_after_rows.append(merged_new)

    after_df = pd.concat(all_after_rows, ignore_index=True).rename(columns={"id": "story_id"})

    gold_df["story_id"] = gold_df["story_id"].astype(int)
    after_df["story_id"] = after_df["story_id"].astype(int)

    combined = pd.merge(gold_df, after_df, on=["query_id", "story_id"],
                         how="outer", suffixes=("", "_newmeta"))

    for col in META_COLS:
        newcol = f"{col}_newmeta"
        if newcol in combined.columns:
            combined[col] = combined[col].fillna(combined[newcol])
            combined = combined.drop(columns=[newcol])

    query_map = test_df.set_index("query_id")["query"].to_dict()
    combined["query"] = combined["query"].fillna(combined["query_id"].map(query_map))
    if "query_type" in test_df.columns:
        type_map = test_df.set_index("query_id")["query_type"].to_dict()
        combined["query_type"] = combined["query_type"].fillna(combined["query_id"].map(type_map))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_excel(OUTPUT_GOLD_AFTER, index=False)
    print(f"Đã lưu bảng đầy đủ -> {OUTPUT_GOLD_AFTER}")

    need_label = combined[combined["relevance"].isna()].copy()
    need_label = need_label[["query_id", "query", "story_id"] + META_COLS +
                             ["bm25_rank_after", "semantic_rank_after", "relevance"]]
    need_label.to_excel(OUTPUT_TO_LABEL, index=False)
    print(f"Số dòng CẦN CHẤM relevance thủ công: {len(need_label)}")
    print(f"Đã lưu -> {OUTPUT_TO_LABEL}")


if __name__ == "__main__":
    main()