import re
import unicodedata


def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()                            # Chuyển về chữ thường
    text = unicodedata.normalize("NFC", text)           # Chuẩn hóa Unicode
    text = re.sub(r"\s+", " ", text)                    # Loại bỏ khoảng trắng thừa

    return text.strip()


def create_semantic_text(row):
     #Ghép các trường thành một đoạn văn bản để sinh Embedding.

    fields = [

    row.get("title", ""),

    row.get("genre", ""),

    row.get("tags", ""),

    row.get("description", "")

]

    semantic_text = " ".join(

        normalize_text(field)

        for field in fields

    )

    return semantic_text