"""
Evaluate AFTER: giống hệt evaluate_test.py, chỉ khác nguồn dữ liệu và tên
cột rank (dùng bản MỚI sau Query Parser + Constraint Filter).
"""

from pathlib import Path
import math
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_QUERIES_PATH = PROJECT_ROOT / "data" / "queries" / "test_queries.xlsx"
GOLD_PATH = PROJECT_ROOT / "results" / "evaluation" / "gold_after" / "gold_relevance_after_final.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation" / "gold_after"
OUTPUT_PER_QUERY = OUTPUT_DIR / "test_evaluation_per_query_after.xlsx"
OUTPUT_SUMMARY = OUTPUT_DIR / "test_evaluation_summary_after.xlsx"
K_VALUES = [5, 10]


def precision_at_k(relevances, k):
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(r >= 1 for r in top_k) / len(top_k)


def recall_at_k(relevances, total_relevant, k):
    if total_relevant == 0:
        return 0.0
    return sum(r >= 1 for r in relevances[:k]) / total_relevant


def reciprocal_rank(relevances):
    for rank, r in enumerate(relevances, start=1):
        if r >= 1:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances, k):
    return sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(relevances[:k], start=1))


def ndcg_at_k(relevances, k):
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def load_data():
    test_df = pd.read_excel(TEST_QUERIES_PATH)
    gold_df = pd.read_excel(GOLD_PATH)
    for col in ["bm25_rank_after", "semantic_rank_after"]:
        if col not in gold_df.columns:
            raise ValueError(f"gold_relevance_after_final.xlsx thiếu cột {col}")
    return test_df, gold_df


def prepare_data(test_df, gold_df):
    test_df = test_df.copy()
    gold_df = gold_df.copy()
    test_df["query_id"] = test_df["query_id"].astype(str).str.strip()
    gold_df["query_id"] = gold_df["query_id"].astype(str).str.strip()
    gold_df["relevance"] = pd.to_numeric(gold_df["relevance"], errors="coerce").fillna(0).astype(int)
    gold_df["bm25_rank_after"] = pd.to_numeric(gold_df["bm25_rank_after"], errors="coerce")
    gold_df["semantic_rank_after"] = pd.to_numeric(gold_df["semantic_rank_after"], errors="coerce")
    test_ids = set(test_df["query_id"])
    gold_test = gold_df[gold_df["query_id"].isin(test_ids)].copy()
    return test_df, gold_test


def get_ranked_relevances(query_gold, rank_col):
    ranked = query_gold.dropna(subset=[rank_col]).sort_values(by=rank_col)
    return ranked["relevance"].astype(int).tolist()


def get_total_relevant(query_gold):
    return int((query_gold["relevance"] >= 1).sum())


def evaluate_model(query_gold, rank_col):
    relevances = get_ranked_relevances(query_gold, rank_col)
    total_relevant = get_total_relevant(query_gold)
    result = {
        "num_retrieved": len(relevances),
        "num_relevant_gold": total_relevant,
        "MRR": reciprocal_rank(relevances),
    }
    for k in K_VALUES:
        result[f"Precision@{k}"] = precision_at_k(relevances, k)
        result[f"Recall@{k}"] = recall_at_k(relevances, total_relevant, k)
        result[f"nDCG@{k}"] = ndcg_at_k(relevances, k)
    return result


def evaluate_test(test_df, gold_test):
    results = []
    query_map = test_df.drop_duplicates("query_id").set_index("query_id")["query"].to_dict()
    type_map = {}
    if "query_type" in test_df.columns:
        type_map = test_df.drop_duplicates("query_id").set_index("query_id")["query_type"].to_dict()

    for qid in test_df["query_id"].drop_duplicates():
        query_gold = gold_test[gold_test["query_id"] == qid].copy()
        if query_gold.empty:
            print(f"⚠️ {qid} không có Gold.")
            continue
        for model_name, rank_col in [("BM25", "bm25_rank_after"), ("Semantic", "semantic_rank_after")]:
            metrics = evaluate_model(query_gold, rank_col)
            results.append({"query_id": qid, "query": query_map.get(qid, ""),
                             "query_type": type_map.get(qid, ""), "model": model_name, **metrics})
    return pd.DataFrame(results)


def create_summary(per_query_df):
    cols = ["Precision@5", "Precision@10", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]
    summary = per_query_df.groupby("model")[cols].mean().reset_index()
    summary[cols] = summary[cols].round(4)
    return summary


def main():
    test_df, gold_df = load_data()
    test_df, gold_test = prepare_data(test_df, gold_df)
    per_query_df = evaluate_test(test_df, gold_test)
    summary_df = create_summary(per_query_df)
    print(summary_df.to_string(index=False))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_query_df.to_excel(OUTPUT_PER_QUERY, index=False)
    summary_df.to_excel(OUTPUT_SUMMARY, index=False)
    print(f"\nĐã lưu -> {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()