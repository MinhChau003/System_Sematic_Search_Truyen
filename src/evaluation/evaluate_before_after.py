"""
Đo Before/After (Hướng B) -- đếm vi phạm rule-based trên Top-10, KHÔNG chấm
lại relevance đầy đủ. Dùng lại đúng logic filter trong constraint_filter.py
để đảm bảo số liệu nhất quán với pipeline thật.

Before = search_bm25.search()/sem.search() thẳng, không qua Parser/Filter.
After  = retrieval.search() (pipeline đầy đủ có Constraint Filter).
"""

import os
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)                 # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)

BACKEND_DIR = os.path.join(SRC_DIR, "backend")
BM25_DIR = os.path.join(SRC_DIR, "bm25")
SEMANTIC_DIR = os.path.join(SRC_DIR, "semantic")

for p in (BACKEND_DIR, BM25_DIR, SEMANTIC_DIR):
    if p not in sys.path:
        sys.path.append(p)

import search_bm25                       # noqa: E402
import search_semantic as sem            # noqa: E402
from query_parser import parse_query     # noqa: E402
import constraint_filter as cf           # noqa: E402
import retrieval                         # noqa: E402


TEST_QUERIES_PATH = os.path.join(PROJECT_ROOT, "data", "queries", "test_queries.xlsx")
COL_QUERY_ID = "query_id"
COL_QUERY = "query"
COL_CATEGORY = "query_type"   

TOP_K = 10
METHODS = ["bm25", "semantic"]


def _count_violations(df: pd.DataFrame, parsed) -> dict:
    """Đếm mismatch THẬT bằng cách match trực tiếp, KHÔNG dùng cf.filter_*
    (vì các hàm đó có fallback, sẽ làm sai lệch số đếm khi fallback kích hoạt
    ngay trong lúc đếm). Dùng để đo cả trước và sau pipeline một cách nhất quán."""
    n = len(df)
    counts = {}

    def _col(name):
        return df[name].fillna("") if name in df.columns else pd.Series([""] * n, index=df.index)

    if parsed.negated_phrases:
        short_text = (_col("title") + " " + _col("genre") + " " + _col("tags")).str.lower()
        desc_text = _col("description").str.lower()
        violated = [
            any(cf._row_violates_phrase(s, d, p) for p in parsed.negated_phrases)
            for s, d in zip(short_text, desc_text)
        ]
        counts["E2_negation"] = sum(violated)
    else:
        counts["E2_negation"] = None

    if parsed.genre_hints:
        hints_lower = [g.lower() for g in parsed.genre_hints]
        genre_lower = _col("genre").str.lower()
        matched = genre_lower.apply(lambda g: any(h in g for h in hints_lower))
        counts["E1_genre"] = int((~matched).sum())
    else:
        counts["E1_genre"] = None

    if parsed.tag_hints:
        hints_lower = [t.lower() for t in parsed.tag_hints]
        tags_lower = _col("tags").str.lower()
        matched = tags_lower.apply(lambda t: any(h in t for h in hints_lower))
        counts["E3_tag"] = int((~matched).sum())
    else:
        counts["E3_tag"] = None

    if parsed.status:
        status_lower = _col("status").str.lower()
        matched = status_lower == parsed.status.lower()
        counts["status_mismatch"] = int((~matched).sum())
    else:
        counts["status_mismatch"] = None

    if parsed.chapter_constraint:
        operator, number = parsed.chapter_constraint
        op_func = cf._OPS.get(operator)
        chapters_numeric = pd.to_numeric(_col("chapters"), errors="coerce")
        matched = op_func(chapters_numeric, number).fillna(False)
        counts["chapter_mismatch"] = int((~matched).sum())
    else:
        counts["chapter_mismatch"] = None

    return counts


def main():
    test_df = pd.read_excel(TEST_QUERIES_PATH)
    known_genres = retrieval._load_known_genres_once()
    known_tags = retrieval._load_known_tags_once()

    rows = []

    for _, row in test_df.iterrows():
        qid = row[COL_QUERY_ID]
        query = row[COL_QUERY]
        category = row[COL_CATEGORY]

        parsed = parse_query(query, known_genres, known_tags)

        for method in METHODS:
            # --- BEFORE: raw, không qua Parser/Filter ---
            if method == "bm25":
                before_raw = search_bm25.search(query, top_k=TOP_K).reset_index(drop=True)
            else:
                model, index, stories = retrieval._load_semantic_once()
                before_raw = sem.search(query, model, index, stories, top_k=TOP_K).reset_index(drop=True)
            before_enriched = retrieval._enrich_with_db(before_raw)

            # --- AFTER: pipeline đầy đủ ---
            after_enriched = retrieval.search(query, method=method, top_k=TOP_K, verbose=False)

            before_counts = _count_violations(before_enriched, parsed)
            after_counts = _count_violations(after_enriched, parsed)

            for err_type in before_counts:
                rows.append({
                    "query_id": qid,
                    "category": category,
                    "method": method,
                    "error_type": err_type,
                    "before": before_counts[err_type],
                    "after": after_counts[err_type],
                })

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(CURRENT_DIR, "before_after_violations.xlsx")
    result_df.to_excel(out_path, index=False)
    print(f"Đã lưu chi tiết -> {out_path}")

    # Bảng tổng hợp: tổng vi phạm mỗi loại lỗi, gộp cả 2 method
    summary = (
        result_df.dropna(subset=["before"])
        .groupby("error_type")[["before", "after"]]
        .sum()
    )
    print("\n===== TỔNG HỢP BEFORE vs AFTER =====")
    print(summary)


if __name__ == "__main__":
    main()