import os
import pickle

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from preprocess_semantic import normalize_text

# --- MODEL_DIR tuyệt đối dựa trên vị trí file, không phụ thuộc cwd ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "..", "models", "semantic")
MODEL_NAME = "AITeamVN/Vietnamese_Embedding"

FAISS_PATH = os.path.join(
    MODEL_DIR,
    "faiss.index"
)

STORIES_PATH = os.path.join(
    MODEL_DIR,
    "stories.pkl"
)


def load_model():
    # Load mô hình Embedding

    print("Đang tải mô hình...")

    model = SentenceTransformer(MODEL_NAME)

    print("Đã tải mô hình!")

    return model


def load_index():
    # Đọc FAISS Index

    index = faiss.read_index(FAISS_PATH)

    return index


def load_stories():
    # Đọc metadata truyện

    with open(STORIES_PATH, "rb") as f:

        stories = pickle.load(f)

    return stories


def embed_query(model, query):
    # Sinh Embedding cho Query

    query = normalize_text(query)

    embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embedding)

    return embedding


def search(query,
           model,
           index,
           stories,
           top_k=10):
    # Tìm kiếm Semantic

    query_embedding = embed_query(
        model,
        query
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = stories.iloc[
        indices[0]
    ].copy()

    results["score"] = scores[0]

    return results


def display_results(results):

    print("\n===== KẾT QUẢ =====\n")

    for rank, (_, row) in enumerate(results.iterrows(), start=1):

        print(f"{rank}. {row['title']}")

        print(f"   Thể loại : {row['genre']}")

        print(f"   Trạng thái: {row['status']}")

        print(f"   Score     : {row['score']:.4f}")

        print()


def main():

    model = load_model()

    index = load_index()

    stories = load_stories()

    while True:

        query = input("Nhập truy vấn: ")

        if query.lower() == "exit":

            break

        results = search(
            query,
            model,
            index,
            stories
        )

        display_results(results)


if __name__ == "__main__":

    main()