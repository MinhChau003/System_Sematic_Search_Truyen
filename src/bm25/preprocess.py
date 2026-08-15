import re
import unicodedata

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()                            # Chuyển về chữ thường
    text = unicodedata.normalize("NFC", text)           # Chuẩn hóa ký tự Unicode
    text = re.sub(r"\s+", " ", text)                    # Loại bỏ khoảng trắng thừa

    return text.strip()

def create_search_text(row):                            # Ghép các trường quan trọng thành SearchText

    search_text = ""

    # Tăng trọng số Title vì tên truyện là thông tin quan trọng nhất
    search_text += (normalize_text(row["title"]) + " ") * 5


    # Tăng trọng số Genre vì thể loại ảnh hưởng nhiều đến nhu cầu tìm kiếm
    search_text += (normalize_text(row["genre"]) + " ") * 4


    # Tags chứa các đặc điểm như hệ thống, trọng sinh, xuyên không...
    search_text += (normalize_text(row["tags"]) + " ") * 3


    # Description dùng để bổ sung nội dung chi tiết
    search_text += normalize_text(row["description"])


    return search_text.strip()

# Các từ xuất hiện nhiều nhưng không mang nhiều ý nghĩa khi tìm kiếm
STOPWORDS = {

    "truyện",
    "bộ",
    "của",
    "và",
    "là",
    "có",
    "một",
    "những"

}

def tokenize(text):

    tokens = text.split()                               # Tách từ đơn giản bằng khoảng trắng


    tokens = [
        token 
        for token in tokens
        if token not in STOPWORDS                       # Loại bỏ các từ gây nhiễu
    ]


    return tokens