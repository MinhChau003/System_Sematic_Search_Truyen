# Hệ thống Tìm kiếm Ngữ nghĩa Truyện chữ

So sánh BM25 (lexical) và Semantic Search (SentenceTransformer + FAISS) trên kho
truyện chữ tiếng Việt, có Query Parser + Constraint Filter hậu xử lý ràng buộc
tường minh (thể loại, tag, trạng thái, số chương, phủ định), Backend nối SQLite,
và giao diện Streamlit.

> Đồ án môn học — Xây dựng hệ thống tìm kiếm ngữ nghĩa truyện chữ từ mô tả nhu cầu
> bằng ngôn ngữ tự nhiên.

## Yêu cầu hệ thống

- Python 3.11+
- Windows (đã test trên Windows)

## Cài đặt

```powershell
# 1. Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Cài thư viện
pip install -r requirements.txt
```

## Cấu trúc thư mục

```text
System_Sematic_Search_Truyen/
├── data/
│   ├── raw/                    # Dữ liệu crawl thô (stories_raw.xlsx, story_urls.txt)
│   ├── cleaned/                # Dữ liệu đã làm sạch (stories_cleaned.xlsx)
│   └── queries/                # Bộ query: all/dev/test_queries.xlsx
├── database/
│   └── stories.db              # SQLite, bảng `stories`
├── models/
│   ├── bm25/                   # bm25.pkl, documents.pkl, stories.pkl
│   └── semantic/               # embeddings.npy, faiss.index, stories.pkl
├── results/
│   └── evaluation/             # Gold Relevance, Error Analysis, kết quả Before/After
│       ├── archive/            # Script/output trung gian đã hoàn thành nhiệm vụ
│       └── gold_after/         # Kết quả đánh giá After (Hướng A, pipeline Tầng 2)
├── src/
│   ├── GetData/                # Crawl + làm sạch dữ liệu
│   ├── bm25/                   # Build & search BM25
│   ├── semantic/               # Build embedding, FAISS, search Semantic
│   ├── backend/                # Query Parser + Constraint Filter + Retrieval orchestrator
│   └── evaluation/             # Đánh giá, Error Analysis, Before/After
│       └── gold_after/         # Script đo After đầy đủ
├── app.py                      # Giao diện Streamlit
├── requirements.txt
└── README.md
```

## Kiến trúc pipeline (Tầng 2)

```text
Query
→ Query Parser (query_parser.py)
trích: negated_phrases, status, chapter_constraint, genre_hints, tag_hints
→ Retrieval pool rộng (BM25 hoặc Semantic)
→ SQLite enrich (join đủ metadata)
→ Constraint Filter (constraint_filter.py)
- NEGATION: hard filter duy nhất
- GENRE / TAG / STATUS / CHAPTER: soft boost (rerank theo số tiêu chí khớp)
→ Sort → Top-K
```

Chi tiết đầy đủ về thiết kế, đánh giá Before/After, hạn chế và hướng phát triển: xem 

## Cách chạy

### 1. Chuẩn bị dữ liệu (bỏ qua nếu đã có sẵn `database/stories.db`)

```powershell
python src/GetData/get_story_urls.py
python src/GetData/crawl_stories.py
python src/GetData/clean_dataset.py
python src/GetData/import_to_sqlite.py
```

### 2. Build model (bỏ qua nếu đã có sẵn file trong `models/`)

```powershell
python src/bm25/build_bm25.py
python src/semantic/build_embedding.py
python src/semantic/build_faiss.py
```

### 3. Chạy thử backend qua CLI

```powershell
python src/backend/retrieval.py
```

### 4. Chạy giao diện Streamlit

```powershell
streamlit run app.py
```

### 5. Đánh giá hệ thống

```powershell
# Đánh giá chính thức (BM25 vs Semantic) trên tập Test
python src/evaluation/evaluate_test.py

# Đánh giá After (pipeline Tầng 2) — chạy theo đúng thứ tự
python src/evaluation/gold_after/build_after_gold.py
# -> mở results/evaluation/gold_after/to_label.xlsx, chấm relevance thủ công cho truyện mới
python src/evaluation/gold_after/merge_labels.py
python src/evaluation/gold_after/evaluate_test_after.py

# Error Analysis rule-based
python src/evaluation/extract_errors.py
python src/evaluation/summarize_errors.py
```

## Kết quả chính (tóm tắt)

| Metric | BM25 Before | BM25 After | Semantic Before | Semantic After |
|---|---|---|---|---|
| P@10 | 0.640 | 0.91 | 0.715 | 0.93 |
| R@10 | 0.4943 | 0.4761 | 0.5974 | 0.4937 |
| MRR | 0.7905 | 0.9417 | 0.9375 | 0.9750 |
| nDCG@10 | 0.7019 | 0.9258 | 0.8138 | 0.9429 |

Chi tiết đầy đủ: xem 

## Hạn chế & Hướng phát triển

Xem mục 