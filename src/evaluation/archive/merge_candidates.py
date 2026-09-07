import os
import pandas as pd

BM25_PATH = "../../results/evaluation/bm25_candidates.xlsx"
SEMANTIC_PATH = "../../results/evaluation/semantic_candidates.xlsx"
STORIES_PATH = "../../data/cleaned/stories_cleaned.xlsx"
OUTPUT_PATH = "../../results/evaluation/candidate_set.xlsx"


def load_results():
    bm25 = pd.read_excel(BM25_PATH)
    semantic = pd.read_excel(SEMANTIC_PATH)

    return bm25, semantic


def load_story_metadata():
    stories = pd.read_excel(STORIES_PATH)

    # Chỉ lấy những trường cần bổ sung
    stories = stories[
        [
            "ID",
            "Tags",
            "Description",
            "Chapters"
        ]
    ].copy()

    # Đổi tên cột metadata để join
    stories = stories.rename(
        columns={
            "ID": "story_id",
            "Tags": "tags",
            "Description": "description",
            "Chapters": "chapters"
        }
    )

    return stories


def merge_candidates(bm25, semantic, stories):

    # Đổi tên rank và score
    bm25 = bm25.rename(
        columns={
            "rank": "bm25_rank",
            "score": "bm25_score"
        }
    )

    semantic = semantic.rename(
        columns={
            "rank": "semantic_rank",
            "score": "semantic_score"
        }
    )

    # Chỉ giữ các cột cần thiết từ BM25
    bm25 = bm25[
        [
            "query_id",
            "query",
            "query_type",
            "story_id",
            "title",
            "genre",
            "status",
            "url",
            "bm25_rank",
            "bm25_score"
        ]
    ]

    # Chỉ giữ các cột cần thiết từ Semantic
    semantic = semantic[
        [
            "query_id",
            "query",
            "query_type",
            "story_id",
            "title",
            "genre",
            "status",
            "url",
            "semantic_rank",
            "semantic_score"
        ]
    ]

    # Gộp BM25 và Semantic
    merged = pd.merge(
        bm25,
        semantic,
        on=[
            "query_id",
            "story_id"
        ],
        how="outer",
        suffixes=("_bm25", "_semantic")
    )

    # Lấy thông tin chung từ BM25 hoặc Semantic
    merged["query"] = (
        merged["query_bm25"]
        .combine_first(merged["query_semantic"])
    )

    merged["query_type"] = (
        merged["query_type_bm25"]
        .combine_first(merged["query_type_semantic"])
    )

    merged["title"] = (
        merged["title_bm25"]
        .combine_first(merged["title_semantic"])
    )

    merged["genre"] = (
        merged["genre_bm25"]
        .combine_first(merged["genre_semantic"])
    )

    merged["status"] = (
        merged["status_bm25"]
        .combine_first(merged["status_semantic"])
    )

    merged["url"] = (
        merged["url_bm25"]
        .combine_first(merged["url_semantic"])
    )

    # Bổ sung Tags và Description
    merged = pd.merge(
        merged,
        stories,
        on="story_id",
        how="left"
    )

    # Xác định hệ thống retrieve
    def get_source(row):
        has_bm25 = pd.notna(row["bm25_rank"])
        has_semantic = pd.notna(row["semantic_rank"])

        if has_bm25 and has_semantic:
            return "Both"

        if has_bm25:
            return "BM25"

        return "Semantic"

    merged["retrieved_by"] = merged.apply(
        get_source,
        axis=1
    )

    # Để trống relevance
    merged["relevance"] = pd.NA

    # Người đánh giá
    merged["evaluator"] = ""

    # Ghi chú trường hợp khó
    merged["notes"] = ""

    # Tạo rank để sắp xếp
    merged["sort_rank"] = (
        merged["bm25_rank"]
        .fillna(999)
        .combine(
            merged["semantic_rank"].fillna(999),
            min
        )
    )

    merged = merged.sort_values(
        by=[
            "query_id",
            "sort_rank"
        ]
    )

    # Chọn các cột cuối cùng
    result = merged[
        [
            "query_id",
            "query",
            "query_type",
            "story_id",
            "title",
            "genre",
            "tags",
            "description",
            "status",
            "chapters",
            "url",
            "bm25_rank",
            "bm25_score",
            "semantic_rank",
            "semantic_score",
            "retrieved_by",
            "relevance",
            "evaluator",
            "notes"
        ]
    ]

    return result


def save_results(df):
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_excel(
        OUTPUT_PATH,
        index=False
    )

    print("\nĐã tạo Candidate Set!")
    print(OUTPUT_PATH)
    print(f"Tổng số candidate: {len(df)}")

    print("\n===== Kiểm tra dữ liệu =====")

    print(
        f"Candidate có Tags       : "
        f"{df['tags'].notna().sum()}/{len(df)}"
    )

    print(
        f"Candidate có Description: "
        f"{df['description'].notna().sum()}/{len(df)}"
    )
    
    print(
        f"Candidate có Chapters   : "
        f"{df['chapters'].notna().sum()}/{len(df)}"
    )

    print("\n===== Nguồn Candidate =====")

    print(
        df["retrieved_by"]
        .value_counts()
    )


def main():
    bm25, semantic = load_results()

    print(
        f"BM25 candidates     : {len(bm25)}"
    )

    print(
        f"Semantic candidates : {len(semantic)}"
    )

    print(
        "\nĐang đọc metadata từ stories_cleaned.xlsx..."
    )

    stories = load_story_metadata()

    print(
        f"Số truyện trong cleaned dataset: "
        f"{len(stories)}"
    )

    candidates = merge_candidates(
        bm25,
        semantic,
        stories
    )

    save_results(candidates)


if __name__ == "__main__":
    main()