import pickle
import pandas as pd

from preprocess import normalize_text, tokenize

with open("../../models/bm25/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("../../models/bm25/documents.pkl", "rb") as f:
    documents = pickle.load(f)

df = pd.read_pickle("../../models/bm25/stories.pkl")

def search(query, top_k=10):

    query = normalize_text(query)

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    df_result = df.copy()

    df_result["score"] = scores

    df_result = df_result.sort_values(
        by="score",
        ascending=False
    )

    return df_result.head(top_k)

if __name__ == "__main__":

    while True:

        query = input("\nNhập truy vấn: ")

        if query.lower() == "exit":
            break

        results = search(query)

        print("\n===== KẾT QUẢ =====\n")

        for i, row in enumerate(results.itertuples(), start=1):

            print(f"{i}. {row.title}")

            print(f"   Thể loại : {row.genre}")

            print(f"   Trạng thái: {row.status}")

            print(f"   Score     : {row.score:.3f}")

            print()