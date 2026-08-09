import sqlite3
import pandas as pd
import pickle
import os
from rank_bm25 import BM25Okapi

from preprocess import create_search_text, tokenize

DB_PATH = "../database/stories.db"    #cd src

def load_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        id,
        title,
        description,
        genre,
        tags,
        status,
        url
    FROM stories
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df

def build_bm25(df):
    # Tạo cột SearchText
    df["search_text"] = df.apply(create_search_text, axis=1)

    # Tokenize
    documents = [
        tokenize(text)
        for text in df["search_text"]
    ]

    # Xây dựng BM25
    bm25 = BM25Okapi(documents)

    return bm25, documents, df

def save_bm25(bm25, documents, df):

    # Nếu thư mục chưa tồn tại thì tạo
    os.makedirs("../models/bm25", exist_ok=True)

    # Lưu BM25
    with open("../models/bm25/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # Lưu documents đã tokenize
    with open("../models/bm25/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    # Lưu DataFrame
    df.to_pickle("../models/bm25/stories.pkl")

    print("Đã lưu BM25 thành công!")

if __name__ == "__main__":

    # Đọc dữ liệu
    df = load_data()

    print("===== Dữ liệu gốc =====")
    print(df.head())

    # Xây dựng BM25
    bm25, documents, df = build_bm25(df)

    print("\n===== Search Text =====")
    print(df[["title", "search_text"]].head())
    
    # Lưu model
    save_bm25(bm25, documents, df)

    print(f"\nTổng số truyện: {len(documents)}")
    print("Xây dựng BM25 thành công!")