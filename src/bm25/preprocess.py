import re
import unicodedata

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()                            #Chuyển về chữ thường
    text = unicodedata.normalize("NFC", text)           #Chuẩn hóa ký tự
    text = re.sub(r"\s+", " ", text)                    #Loại bỏ khoảng trắng thừa

    return text.strip()

def create_search_text(row):                            #Ghép thành 1 đoạn văn bản duy nhất để tìm kiếm
    fields = [
        row["title"],
        row["genre"],
        row["tags"],
        row["description"],
        row["status"]
    ]

    search_text = " ".join(
        normalize_text(field)
        for field in fields
    )

    return search_text


def tokenize(text):
    
    return text.split()   #Tách từ đơn giản bằng khoảng trắng.