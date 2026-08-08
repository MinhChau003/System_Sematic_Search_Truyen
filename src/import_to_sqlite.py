import sqlite3
import pandas as pd

# Đọc file Excel
df = pd.read_excel("../data/cleaned/stories_cleaned.xlsx")

# Kết nối SQLite
conn = sqlite3.connect("../database/stories.db")

# Ghi vào bảng stories
df.to_sql(
    "stories",
    conn,
    if_exists="append",
    index=False
)

conn.commit()
conn.close()

print("Import thành công!")
print("Số truyện:", len(df))