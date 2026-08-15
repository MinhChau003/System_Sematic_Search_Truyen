import os

import faiss
import numpy as np

# Thư mục lưu Model
MODEL_DIR = "../../models/semantic"
# File Embedding
EMBEDDING_PATH = os.path.join(
    MODEL_DIR,
    "embeddings.npy"
)
# File FAISS Index
FAISS_PATH = os.path.join(
    MODEL_DIR,
    "faiss.index"
)

def load_embeddings():
    # Đọc Embedding đã lưu

    embeddings = np.load(EMBEDDING_PATH)

    return embeddings

def normalize_embeddings(embeddings):
    # Chuẩn hóa Vector để dùng Cosine Similarity

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    return embeddings

def build_index(embeddings):
    # Xây dựng FAISS Index

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index

def save_index(index):
    # Lưu FAISS Index

    faiss.write_index(
        index,
        FAISS_PATH
    )

    print("Đã lưu FAISS Index!")

def main():

    embeddings = load_embeddings()

    print(f"Số lượng vector: {embeddings.shape[0]}")
    print(f"Số chiều vector : {embeddings.shape[1]}")

    embeddings = normalize_embeddings(
        embeddings
    )

    index = build_index(
        embeddings
    )

    save_index(index)
    print(f"Tổng số vector trong Index: {index.ntotal}")


if __name__ == "__main__":

    main()