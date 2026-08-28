import os
import pickle
import pandas as pd

from preprocess import normalize_text, tokenize

# --- dùng đường dẫn tuyệt đối dựa trên vị trí file này ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "..", "models", "bm25")

BM25_PATH = os.path.join(MODEL_DIR, "bm25.pkl")
DOCUMENTS_PATH = os.path.join(MODEL_DIR, "documents.pkl")
STORIES_PATH = os.path.join(MODEL_DIR, "stories.pkl")

with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

with open(DOCUMENTS_PATH, "rb") as f:
    documents = pickle.load(f)

df = pd.read_pickle(STORIES_PATH)


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