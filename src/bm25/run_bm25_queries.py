import os
import pandas as pd

from search_bm25 import search

# Đường dẫn file Query
QUERY_PATH = "../../data/queries/test_queries.xlsx"

# Đường dẫn lưu kết quả
OUTPUT_PATH = "../../results/bm25_baseline/bm25_top10.xlsx"

def load_queries():
    #Doc danh sach query 
    df_queries = pd.read_excel(QUERY_PATH)

    return df_queries


def run_queries(df_queries):
    #Chay bang bm25

    results = []

    for _, row in df_queries.iterrows():

        query_id = row["id"]
        query = row["query"]

        print(f"Đang chạy Query: {query}")

        top10 = search(query)

        rank = 1

        for _, story in top10.iterrows():

            results.append({

                "query_id": query_id,

                "query": query,

                "rank": rank,

                "story_id": story["id"],

                "title": story["title"],

                "genre": story["genre"],

                "status": story["status"],

                "score": round(story["score"], 4)

            })

            rank += 1

    return pd.DataFrame(results)


def save_results(df_results):
   #Lưu kết quả ra 

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df_results.to_excel(
        OUTPUT_PATH,
        index=False
    )

    print("\nĐã lưu kết quả thành công!")
    print(OUTPUT_PATH)


def main():

    df_queries = load_queries()

    df_results = run_queries(df_queries)

    save_results(df_results)


if __name__ == "__main__":

    main()