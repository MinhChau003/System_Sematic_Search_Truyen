import os
import pandas as pd

from search_semantic import (
    load_model,
    load_index,
    load_stories,
    search
)


# Đường dẫn file Query
QUERY_PATH = "../../data/queries/all_queries.xlsx"

# Đường dẫn lưu kết quả
OUTPUT_PATH = "../../results/evaluation/semantic_candidates.xlsx"


def load_queries():
    # Đọc danh sách query
    df_queries = pd.read_excel(QUERY_PATH)

    return df_queries


def run_queries(df_queries):

    results = []

    # Load model và dữ liệu semantic một lần
    print("Đang tải Semantic Model...")

    model = load_model()
    index = load_index()
    stories = load_stories()

    print(f"Tổng số truyện trong Semantic Index: {len(stories)}")
    print()

    # Chạy từng query
    for _, row in df_queries.iterrows():

        query_id = row["query_id"]
        query = row["query"]
        query_type = row["query_type"]

        print(f"Đang chạy Query: {query_id} - {query}")

        top10 = search(
            query,
            model,
            index,
            stories,
            top_k=10
        )

        rank = 1

        for _, story in top10.iterrows():

            results.append({

                "query_id": query_id,

                "query": query,

                "query_type": query_type,

                "rank": rank,

                "story_id": story["id"],

                "title": story["title"],

                "genre": story["genre"],

                "status": story["status"],

                "score": round(float(story["score"]), 4),

                "url": story["url"]

            })

            rank += 1

    return pd.DataFrame(results)


def save_results(df_results):

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df_results.to_excel(
        OUTPUT_PATH,
        index=False
    )

    print("\nĐã lưu kết quả thành công!")
    print(OUTPUT_PATH)

    print(f"Tổng số dòng: {len(df_results)}")


def main():

    df_queries = load_queries()

    print(f"Tổng số query: {len(df_queries)}")
    print()

    df_results = run_queries(df_queries)

    save_results(df_results)


if __name__ == "__main__":

    main()