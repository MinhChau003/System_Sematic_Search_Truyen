import requests
from bs4 import BeautifulSoup

# Sitemap chứa danh sách truyện
sitemap_url = "https://sitruyencv.com/sitemap-stories-1.xml"

# Gửi request
response = requests.get(sitemap_url)
response.encoding = "utf-8"

# Đọc XML
soup = BeautifulSoup(
    response.text,
    "xml"
)

# Lấy tất cả URL
story_urls = []

for url in soup.find_all("url"):
    loc = url.find("loc")
    
    if loc:
        story_urls.append(loc.text)
        
print("Tổng số truyện tìm thấy:", len(story_urls))

# Lấy 300 truyện đầu tiên
story_urls = story_urls[:300]
print("Số truyện sẽ crawl:", len(story_urls))

# Lưu file
with open(
    "../data/raw/story_urls.txt",
    "w",
    encoding="utf-8"
) as f:

    for url in story_urls:
        f.write(url + "\n")

print("Đã lưu story_urls.txt")