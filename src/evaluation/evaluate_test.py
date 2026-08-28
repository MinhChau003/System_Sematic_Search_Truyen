"""
Evaluate BM25 vs Semantic Retrieval on TEST set.

Input:
    data/queries/test_queries.xlsx
    results/evaluation/gold_relevance.xlsx

Relevance:
    0 = không liên quan
    1 = liên quan một phần
    2 = liên quan cao

Quy ước:
    Precision / Recall / MRR:
        relevance >= 1 được xem là relevant

    nDCG:
        sử dụng graded relevance 0/1/2
"""

from pathlib import Path
import math
import pandas as pd


# 1. CONFIG

# src/evaluation/evaluate_test.py
# parents[0] = evaluation
# parents[1] = src
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_QUERIES_PATH = (
    PROJECT_ROOT / "data" / "queries" / "test_queries.xlsx"
)

GOLD_PATH = (
    PROJECT_ROOT / "results" / "evaluation" / "gold_relevance.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"

OUTPUT_PER_QUERY = (
    OUTPUT_DIR / "test_evaluation_per_query.xlsx"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "test_evaluation_summary.xlsx"
)

K_VALUES = [5, 10]

# 2. METRICS

def precision_at_k(relevances, k):
    """
    Precision@K

    relevance >= 1 được xem là relevant.
    """

    top_k = relevances[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        relevance >= 1
        for relevance in top_k
    )

    return relevant_count / len(top_k)


def recall_at_k(relevances, total_relevant, k):
    """
    Recall@K

    relevance >= 1 được xem là relevant.
    """

    if total_relevant == 0:
        return 0.0

    top_k = relevances[:k]

    retrieved_relevant = sum(
        relevance >= 1
        for relevance in top_k
    )

    return retrieved_relevant / total_relevant


def reciprocal_rank(relevances):
    """
    Reciprocal Rank của một query.

    Tìm vị trí relevant đầu tiên.
    """

    for rank, relevance in enumerate(
        relevances,
        start=1
    ):
        if relevance >= 1:
            return 1.0 / rank

    return 0.0


def dcg_at_k(relevances, k):
    """
    Graded DCG.

    relevance sử dụng trực tiếp giá trị 0/1/2.
    """

    score = 0.0

    for rank, relevance in enumerate(
        relevances[:k],
        start=1
    ):
        score += (
            (2 ** relevance - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(relevances, k):
    """
    Graded nDCG@K.
    """

    actual_dcg = dcg_at_k(
        relevances,
        k
    )

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    ideal_dcg = dcg_at_k(
        ideal_relevances,
        k
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

# 3. VALIDATE COLUMNS

def check_columns(
    df,
    required_columns,
    filename
):
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} thiếu các cột: {missing}\n"
            f"Các cột hiện có:\n{list(df.columns)}"
        )

# 4. LOAD DATA

def load_data():

    if not TEST_QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy:\n{TEST_QUERIES_PATH}"
        )

    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy:\n{GOLD_PATH}"
        )

    test_df = pd.read_excel(
        TEST_QUERIES_PATH
    )

    gold_df = pd.read_excel(
        GOLD_PATH
    )

    check_columns(
        test_df,
        [
            "query_id",
            "query"
        ],
        "test_queries.xlsx"
    )

    check_columns(
        gold_df,
        [
            "query_id",
            "relevance",
            "bm25_rank",
            "semantic_rank"
        ],
        "gold_relevance.xlsx"
    )

    return test_df, gold_df

# 5. PREPARE DATA

def prepare_data(
    test_df,
    gold_df
):

    test_df = test_df.copy()
    gold_df = gold_df.copy()

    # Chuẩn hóa query_id
    test_df["query_id"] = (
        test_df["query_id"]
        .astype(str)
        .str.strip()
    )

    gold_df["query_id"] = (
        gold_df["query_id"]
        .astype(str)
        .str.strip()
    )

    # relevance -> numeric
    gold_df["relevance"] = pd.to_numeric(
        gold_df["relevance"],
        errors="coerce"
    )

    gold_df["relevance"] = (
        gold_df["relevance"]
        .fillna(0)
        .astype(int)
    )

    # rank -> numeric
    gold_df["bm25_rank"] = pd.to_numeric(
        gold_df["bm25_rank"],
        errors="coerce"
    )

    gold_df["semantic_rank"] = pd.to_numeric(
        gold_df["semantic_rank"],
        errors="coerce"
    )

    # Chỉ lấy query thuộc TEST
    test_query_ids = set(
        test_df["query_id"]
    )

    gold_test = gold_df[
        gold_df["query_id"].isin(
            test_query_ids
        )
    ].copy()

    if gold_test.empty:
        raise ValueError(
            "Không tìm thấy query_id trong "
            "gold_relevance.xlsx khớp với TEST."
        )

    # Kiểm tra relevance
    invalid = gold_test[
        ~gold_test["relevance"].isin(
            [0, 1, 2]
        )
    ]

    if not invalid.empty:

        print(
            "\n⚠️ CẢNH BÁO:"
        )

        print(
            "Có relevance ngoài khoảng 0-2."
        )

    return test_df, gold_test

# 6. GET RANKED RELEVANCE

def get_ranked_relevances(
    query_gold,
    rank_column
):

    # Chỉ lấy những truyện mà model đã retrieve
    ranked = query_gold.dropna(
        subset=[rank_column]
    ).copy()

    # Sắp xếp theo rank
    ranked = ranked.sort_values(
        by=rank_column,
        ascending=True
    )

    return (
        ranked["relevance"]
        .astype(int)
        .tolist()
    )

# 7. TOTAL RELEVANT DOCUMENTS

def get_total_relevant(
    query_gold
):

    return int(
        (
            query_gold["relevance"] >= 1
        ).sum()
    )

# 8. EVALUATE ONE MODEL

def evaluate_model(
    query_gold,
    rank_column
):

    relevances = get_ranked_relevances(
        query_gold,
        rank_column
    )

    total_relevant = (
        get_total_relevant(
            query_gold
        )
    )

    result = {
        "num_retrieved": len(
            relevances
        ),

        "num_relevant_gold": (
            total_relevant
        ),

        "MRR": reciprocal_rank(
            relevances
        )
    }

    for k in K_VALUES:

        result[
            f"Precision@{k}"
        ] = precision_at_k(
            relevances,
            k
        )

        result[
            f"Recall@{k}"
        ] = recall_at_k(
            relevances,
            total_relevant,
            k
        )

        result[
            f"nDCG@{k}"
        ] = ndcg_at_k(
            relevances,
            k
        )

    return result

# 9. EVALUATE TEST SET

def evaluate_test(
    test_df,
    gold_test
):

    results = []

    test_query_ids = (
        test_df["query_id"]
        .drop_duplicates()
        .tolist()
    )

    # Map query -> text
    query_map = (
        test_df
        .drop_duplicates(
            "query_id"
        )
        .set_index(
            "query_id"
        )["query"]
        .to_dict()
    )

    # Map query -> type
    query_type_map = {}

    if "query_type" in test_df.columns:

        query_type_map = (
            test_df
            .drop_duplicates(
                "query_id"
            )
            .set_index(
                "query_id"
            )["query_type"]
            .to_dict()
        )

    for query_id in test_query_ids:

        query_gold = gold_test[
            gold_test["query_id"]
            == query_id
        ].copy()

        if query_gold.empty:

            print(
                f"⚠️ {query_id} không có Gold."
            )

            continue

        # BM25

        bm25_metrics = evaluate_model(
            query_gold,
            "bm25_rank"
        )

        results.append(
            {
                "query_id": query_id,
                "query": query_map.get(
                    query_id,
                    ""
                ),
                "query_type": query_type_map.get(
                    query_id,
                    ""
                ),
                "model": "BM25",
                **bm25_metrics
            }
        )

        # Semantic

        semantic_metrics = evaluate_model(
            query_gold,
            "semantic_rank"
        )

        results.append(
            {
                "query_id": query_id,
                "query": query_map.get(
                    query_id,
                    ""
                ),
                "query_type": query_type_map.get(
                    query_id,
                    ""
                ),
                "model": "Semantic",
                **semantic_metrics
            }
        )

    return pd.DataFrame(results)

# 10. SUMMARY

def create_summary(
    per_query_df
):

    metric_columns = [
        "Precision@5",
        "Precision@10",
        "Recall@5",
        "Recall@10",
        "MRR",
        "nDCG@5",
        "nDCG@10"
    ]

    summary = (
        per_query_df
        .groupby("model")[
            metric_columns
        ]
        .mean()
        .reset_index()
    )

    summary[
        metric_columns
    ] = summary[
        metric_columns
    ].round(4)

    return summary

# 11. SAVE RESULTS

def save_results(
    per_query_df,
    summary_df
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Chi tiết từng query
    per_query_df.to_excel(
        OUTPUT_PER_QUERY,
        index=False
    )

    # Tổng hợp BM25 vs Semantic
    summary_df.to_excel(
        OUTPUT_SUMMARY,
        index=False
    )

# 12. MAIN

def main():

    print("=" * 60)
    print(
        "ĐÁNH GIÁ BM25 vs SEMANTIC - TEST SET"
    )
    print("=" * 60)

    # Load

    print(
        "\n[1/5] Đọc dữ liệu..."
    )

    test_df, gold_df = load_data()

    print(
        f"  Test queries: "
        f"{test_df['query_id'].nunique()}"
    )

    print(
        f"  Gold rows: "
        f"{len(gold_df)}"
    )

    # Prepare

    print(
        "\n[2/5] Lọc Gold theo TEST..."
    )

    test_df, gold_test = prepare_data(
        test_df,
        gold_df
    )

    print(
        f"  Gold rows thuộc TEST: "
        f"{len(gold_test)}"
    )

    print(
        f"  Query TEST có Gold: "
        f"{gold_test['query_id'].nunique()}"
    )

    # Evaluate

    print(
        "\n[3/5] Tính metric..."
    )

    per_query_df = evaluate_test(
        test_df,
        gold_test
    )

    if per_query_df.empty:

        raise ValueError(
            "Không tạo được kết quả đánh giá."
        )

    print(
        f"  Đã đánh giá "
        f"{per_query_df['query_id'].nunique()} "
        f"query."
    )

    # Summary

    print(
        "\n[4/5] Tạo bảng tổng hợp..."
    )

    summary_df = create_summary(
        per_query_df
    )

    print(
        "\n"
        + summary_df.to_string(
            index=False
        )
    )

    # Save

    print(
        "\n[5/5] Xuất kết quả..."
    )

    save_results(
        per_query_df,
        summary_df
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "✅ HOÀN TẤT!"
    )

    print(
        f"\nFile chi tiết:"
        f"\n{OUTPUT_PER_QUERY}"
    )

    print(
        f"\nFile tổng hợp:"
        f"\n{OUTPUT_SUMMARY}"
    )

# RUN

if __name__ == "__main__":
    main()