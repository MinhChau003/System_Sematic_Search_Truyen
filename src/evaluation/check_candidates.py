import pandas as pd

BM25_PATH = "../../results/evaluation/bm25_candidates.xlsx"
SEMANTIC_PATH = "../../results/evaluation/semantic_candidates.xlsx"


def main():

    bm25 = pd.read_excel(BM25_PATH)
    semantic = pd.read_excel(SEMANTIC_PATH)

    # Mỗi query: tập story_id mà BM25 / Semantic trả về
    bm25_groups = (
        bm25.groupby("query_id")["story_id"]
        .apply(set)
    )

    semantic_groups = (
        semantic.groupby("query_id")["story_id"]
        .apply(set)
    )

    total_overlap = 0

    print("===== KIỂM TRA OVERLAP =====\n")

    for query_id in bm25_groups.index:

        bm25_ids = bm25_groups[query_id]
        semantic_ids = semantic_groups.get(query_id, set())

        overlap = bm25_ids & semantic_ids

        total_overlap += len(overlap)

        print(
            f"{query_id}: "
            f"BM25={len(bm25_ids)}, "
            f"Semantic={len(semantic_ids)}, "
            f"Trùng={len(overlap)}"
        )

    print("\n===== TỔNG KẾT =====")

    print(f"BM25 tổng: {len(bm25)}")
    print(f"Semantic tổng: {len(semantic)}")
    print(f"Số cặp trùng: {total_overlap}")

    possible = len(bm25)

    print(
        f"Tỷ lệ overlap trên BM25: "
        f"{total_overlap / possible * 100:.2f}%"
    )


if __name__ == "__main__":
    main()