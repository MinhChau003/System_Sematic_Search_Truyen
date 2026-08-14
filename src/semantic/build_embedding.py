import os
import sqlite3
import pickle

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from preprocess_semantic import create_semantic_text

# Đường dẫn Database
DB_PATH = "../../database/stories.db"
# Thư mục lưu Model
MODEL_DIR = "../../models/semantic"
# Model Embedding
MODEL_NAME = "AITeamVN/Vietnamese_Embedding"

def load_data():
    # Đọc dữ liệu từ SQLite

    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        id,
        title,
        author,
        description,
        genre,
        tags,
        status,
        chapters,
        views,
        url
    FROM stories
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df

def prepare_semantic_text(df):
    # Tạo cột semantic_text

    df["semantic_text"] = df.apply(
        create_semantic_text,
        axis=1
    )

    return df

def load_embedding_model():
    # Load model Embedding

    print("Đang tải mô hình Embedding...")

    model = SentenceTransformer(MODEL_NAME)

    print("Đã tải mô hình thành công!")

    return model

def create_embeddings(model, df):
    # Sinh Embedding cho toàn bộ truyện

    print("Đang tạo Embedding...")

    embeddings = model.encode(
        df["semantic_text"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Hoàn thành!")

    return embeddings

def save_embeddings(df, embeddings):
    # Lưu Embedding và Metadata

    os.makedirs(MODEL_DIR, exist_ok=True)

    np.save(
        os.path.join(MODEL_DIR, "embeddings.npy"),
        embeddings
    )

    with open(
        os.path.join(MODEL_DIR, "stories.pkl"),
        "wb"
    ) as f:

        pickle.dump(df, f)

    print("Đã lưu Embedding thành công!")
    
def main():

    df = load_data()

    df = prepare_semantic_text(df)

    model = load_embedding_model()

    embeddings = create_embeddings(
        model,
        df
    )

    save_embeddings(
        df,
        embeddings
    )


if __name__ == "__main__":

    main()