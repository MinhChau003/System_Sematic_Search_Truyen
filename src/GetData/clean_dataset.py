import pandas as pd
import re

# Đường dẫn file

INPUT_FILE = "../data/raw/stories_raw.xlsx"
OUTPUT_FILE = "../data/cleaned/stories_cleaned.xlsx"

# Hàm làm sạch văn bản

def clean_text(text):

    if pd.isna(text):
        return ""
    text = str(text)
    
    # bỏ xuống dòng, tab, nhiều khoảng trắng
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Chuẩn hóa Status

def normalize_status(status):

    if pd.isna(status):
        return ""
    mapping = {
        "Completed": "Hoàn thành",
        "Ongoing": "Đang ra",
        "Dropped": "Tạm ngưng"
    }

    return mapping.get(status, status)

# Đọc dữ liệu

df = pd.read_excel(INPUT_FILE)

print(f"Số truyện ban đầu: {len(df)}")

# Làm sạch các cột văn bản

text_columns = [
    "Title",
    "Author",
    "Description",
    "Genre",
    "Tags"
]
for col in text_columns:
    df[col] = df[col].apply(clean_text)

# Chuẩn hóa Status

df["Status"] = df["Status"].apply(normalize_status)

# Xóa bản ghi trùng

before = len(df)

df = df.drop_duplicates(subset=["ID"])
after = len(df)
print(f"Đã xóa {before-after} truyện trùng ID")

# Xóa dữ liệu thiếu

required_columns = [
    "Title",
    "Author",
    "Description",
    "Genre"
]

before = len(df)

for col in required_columns:
    df = df[df[col] != ""]
after = len(df)
print(f"Đã xóa {before-after} truyện thiếu dữ liệu")

# Sắp xếp theo ID

df = df.sort_values("ID")

# Reset index

df = df.reset_index(drop=True)

# Lưu dữ liệu

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("--------------------------------")
print("Hoàn thành làm sạch dữ liệu")
print(f"Tổng số truyện: {len(df)}")
print(f"Đã lưu tại: {OUTPUT_FILE}")