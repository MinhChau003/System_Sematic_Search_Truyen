import pandas as pd
from pathlib import Path

# CONFIG

INPUT_FILE = Path("../../results/evaluation/gold_relevance.xlsx")
OUTPUT_FILE = Path("../../results/evaluation/gold_relevance_check.xlsx")

REQUIRED_COLUMNS = [
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
    "notes",
]

VALID_RELEVANCE = {0, 1, 2}

# LOAD DATA

if not INPUT_FILE.exists():
    print(f"❌ Không tìm thấy file: {INPUT_FILE}")
    print("Hãy kiểm tra lại đường dẫn INPUT_FILE.")
    raise SystemExit

df = pd.read_excel(INPUT_FILE)

print("=" * 70)
print("CHECK GOLD RELEVANCE")
print("=" * 70)

print(f"\n📂 File: {INPUT_FILE}")
print(f"📊 Tổng số dòng: {len(df)}")


# 1. CHECK COLUMNS

print("\n" + "=" * 70)
print("1. KIỂM TRA CỘT")
print("=" * 70)

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    print("❌ Thiếu các cột:")
    for col in missing_columns:
        print(f"   - {col}")
else:
    print("✅ Đủ toàn bộ các cột cần thiết.")


# 2. CHECK RELEVANCE

print("\n" + "=" * 70)
print("2. KIỂM TRA RELEVANCE")
print("=" * 70)

# Chuẩn hóa relevance
df["_relevance_numeric"] = pd.to_numeric(
    df["relevance"],
    errors="coerce"
)

# Missing
missing_relevance = df[
    df["_relevance_numeric"].isna()
]

print(f"❌ Số dòng chưa có relevance: {len(missing_relevance)}")

if len(missing_relevance) > 0:
    print("\nCác dòng chưa chấm:")
    print(
        missing_relevance[
            ["query_id", "query", "story_id", "title", "evaluator"]
        ].to_string(index=False)
    )

# Invalid values
invalid_relevance = df[
    ~df["_relevance_numeric"].isin(VALID_RELEVANCE)
    & df["_relevance_numeric"].notna()
]

print(f"\n❌ Số dòng relevance không hợp lệ: {len(invalid_relevance)}")

if len(invalid_relevance) > 0:
    print("\nCác relevance không hợp lệ:")
    print(
        invalid_relevance[
            ["query_id", "story_id", "title", "relevance"]
        ].to_string(index=False)
    )

# Distribution
print("\n📌 Phân bố relevance:")

for value in [0, 1, 2]:
    count = (
        df["_relevance_numeric"] == value
    ).sum()

    percentage = count / len(df) * 100

    print(
        f"   relevance = {value}: "
        f"{count} dòng ({percentage:.2f}%)"
    )


# 3. CHECK DUPLICATE

print("\n" + "=" * 70)
print("3. KIỂM TRA TRÙNG QUERY + STORY")
print("=" * 70)

duplicates = df[
    df.duplicated(
        subset=["query_id", "story_id"],
        keep=False
    )
]

print(
    f"❌ Số dòng thuộc nhóm duplicate: "
    f"{len(duplicates)}"
)

if len(duplicates) > 0:
    print("\nCác dòng bị trùng:")

    print(
        duplicates[
            [
                "query_id",
                "story_id",
                "title",
                "relevance",
                "evaluator"
            ]
        ].sort_values(
            ["query_id", "story_id"]
        ).to_string(index=False)
    )


# 4. CHECK QUERY COVERAGE

print("\n" + "=" * 70)
print("4. KIỂM TRA MỖI QUERY ĐÃ ĐƯỢC CHẤM ĐỦ CHƯA")
print("=" * 70)

query_stats = (
    df.groupby("query_id")
    .agg(
        query=("query", "first"),
        total_candidates=("story_id", "count"),
        judged=("relevance", lambda x: x.notna().sum()),
        missing=("relevance", lambda x: x.isna().sum()),
    )
    .reset_index()
)

query_stats["complete"] = (
    query_stats["missing"] == 0
)

incomplete_queries = query_stats[
    ~query_stats["complete"]
]

print(
    f"📌 Tổng số query: {len(query_stats)}"
)

print(
    f"✅ Query đã chấm đủ: "
    f"{query_stats['complete'].sum()}"
)

print(
    f"❌ Query chưa chấm đủ: "
    f"{len(incomplete_queries)}"
)

if len(incomplete_queries) > 0:

    print("\nCác query chưa chấm đủ:")

    print(
        incomplete_queries[
            [
                "query_id",
                "query",
                "total_candidates",
                "judged",
                "missing"
            ]
        ].to_string(index=False)
    )


# 5. CHECK EVALUATOR

print("\n" + "=" * 70)
print("5. KIỂM TRA EVALUATOR")
print("=" * 70)

missing_evaluator = df[
    df["evaluator"].isna()
    | (
        df["evaluator"]
        .astype(str)
        .str.strip()
        .eq("")
    )
]

print(
    f"❌ Số dòng chưa ghi evaluator: "
    f"{len(missing_evaluator)}"
)

if len(missing_evaluator) > 0:
    print(
        missing_evaluator[
            [
                "query_id",
                "story_id",
                "title",
                "relevance"
            ]
        ].to_string(index=False)
    )


# 6. CHECK NOTES

print("\n" + "=" * 70)
print("6. KIỂM TRA NOTES")
print("=" * 70)

missing_notes = df[
    df["notes"].isna()
    | (
        df["notes"]
        .astype(str)
        .str.strip()
        .eq("")
    )
]

print(
    f"⚠️ Số dòng không có notes: "
    f"{len(missing_notes)}"
)

print(
    "ℹ️ Notes có thể để trống nếu rubric của bạn "
    "không bắt buộc giải thích cho mọi dòng."
)


# 7. TÌM CÁC DÒNG CẦN REVIEW

print("\n" + "=" * 70)
print("7. TÌM CÁC DÒNG NÊN REVIEW THỦ CÔNG")
print("=" * 70)

review_rows = []

for idx, row in df.iterrows():

    relevance = row["_relevance_numeric"]

    query = str(row["query"]).lower()
    genre = str(row["genre"]).lower()
    tags = str(row["tags"]).lower()
    description = str(row["description"]).lower()

    text = f"{genre} {tags} {description}"

    # Case A:
    # relevance = 2 nhưng query có nhiều từ không xuất hiện
    # trong metadata -> cần xem lại

    query_words = [
        word.strip(".,!?;:()[]{}\"'")
        for word in query.split()
        if len(word.strip(".,!?;:()[]{}\"'")) >= 4
    ]

    matched_words = [
        word for word in query_words
        if word in text
    ]

    if relevance == 2 and len(query_words) >= 3:

        match_ratio = len(matched_words) / len(query_words)

        if match_ratio < 0.2:
            review_rows.append(
                (
                    idx,
                    "HIGH",
                    "relevance=2 nhưng metadata "
                    "khớp rất ít từ query"
                )
            )

    # Case B:
    # relevance = 0 nhưng query lại có nhiều từ khớp

    if relevance == 0 and len(query_words) >= 3:

        match_ratio = len(matched_words) / len(query_words)

        if match_ratio >= 0.5:
            review_rows.append(
                (
                    idx,
                    "HIGH",
                    "relevance=0 nhưng metadata "
                    "khớp khá nhiều từ query"
                )
            )

    # Case C:
    # relevance bị thiếu

    if pd.isna(relevance):
        review_rows.append(
            (
                idx,
                "HIGH",
                "Chưa chấm relevance"
            )
        )


if review_rows:

    review_df = pd.DataFrame(
        review_rows,
        columns=[
            "row_index",
            "priority",
            "reason"
        ]
    )

    print(
        f"⚠️ Có {len(review_df)} dòng "
        f"cần review thủ công."
    )

    print(
        "\n"
        + review_df.to_string(index=False)
    )

else:

    review_df = pd.DataFrame(
        columns=[
            "row_index",
            "priority",
            "reason"
        ]
    )

    print(
        "Không phát hiện dòng bất thường "
        "theo heuristic."
    )

# 8. SUMMARY

print("\n" + "=" * 70)
print("8. TỔNG KẾT")
print("=" * 70)

total = len(df)

valid = df["_relevance_numeric"].isin(
    VALID_RELEVANCE
).sum()

missing = df["_relevance_numeric"].isna().sum()

invalid = (
    (~df["_relevance_numeric"].isin(VALID_RELEVANCE))
    & df["_relevance_numeric"].notna()
).sum()

print(f"Tổng dòng              : {total}")
print(f"Relevance hợp lệ 0/1/2 : {valid}")
print(f"Relevance bị thiếu     : {missing}")
print(f"Relevance không hợp lệ : {invalid}")
print(
    f"Query hoàn chỉnh       : "
    f"{query_stats['complete'].sum()}/{len(query_stats)}"
)
print(
    f"Dòng cần review        : "
    f"{len(review_df)}"
)

# 9. EXPORT FILE REVIEW

# Thêm cột trạng thái kiểm tra
df["check_status"] = "OK"

df.loc[
    df["_relevance_numeric"].isna(),
    "check_status"
] = "MISSING_RELEVANCE"

df.loc[
    (
        ~df["_relevance_numeric"].isin(VALID_RELEVANCE)
        & df["_relevance_numeric"].notna()
    ),
    "check_status"
] = "INVALID_RELEVANCE"

# Đánh dấu heuristic review
for idx, _, _ in review_rows:
    if df.loc[idx, "check_status"] == "OK":
        df.loc[idx, "check_status"] = "REVIEW"

# Xóa cột phụ
df = df.drop(columns=["_relevance_numeric"])

df.to_excel(OUTPUT_FILE, index=False)

print(
    f"\n💾 Đã xuất file kiểm tra: "
    f"{OUTPUT_FILE}"
)

print("\nCHECK GOLD RELEVANCE HOÀN TẤT")