import requests
import pandas as pd
import time
import re


API_URL = "https://api.sitruyencv.com/api/stories/{}"

headers = {
    "User-Agent": "Mozilla/5.0"
}

def normalize_status(status):

    mapping = {
        "Completed": "Hoàn thành",
        "Ongoing": "Đang ra",
        "Dropped": "Tạm ngưng"
    }

    return mapping.get(status, status)

def extract_id(url):
    """
    Lấy ID từ URL story
    """
    match = re.search(r"/story/(\d+)", url)

    if match:
        return match.group(1)

    return None

def get_story(story_id):
    url = API_URL.format(story_id)
    
    # Retry 3 lần nếu lỗi mạng
    for attempt in range(3):
        try:
            
            response = requests.get(
                url,
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                return None

            data = response.json()
            if "data" not in data:
                return None

            story = data["data"]
            return {
                "ID": story["id"],
                "Title": story["title"],
                "Author": story["author"]["name"],
                "Description": story["description"],
                "Genre": ", ".join(
                    [
                        c["name"]
                        for c in story["categories"]
                    ]
                ),
                "Tags": ", ".join(
                    [
                        t["name"]
                        for t in story["tags"]
                    ]
                ),
                "Status": normalize_status(
                    story["status"]
                ),
                "Chapters": story["total_chapters"],
                "Views": story["total_views"],
                "URL": f"https://sitruyencv.com/story/{story['id']}-{story['slug']}"
            }
            
        except requests.exceptions.RequestException:

            print(
                f"Lỗi kết nối ID {story_id} "
                f"- thử lại {attempt+1}/3"
            )
            time.sleep(5)

    print(
        f"Bỏ qua ID {story_id} sau 3 lần thử"
    )
    return None

# đọc URL
with open(
    "../data/raw/story_urls.txt",
    encoding="utf-8"
) as f:
    urls = f.readlines()

stories = []

for i, url in enumerate(urls):
    url = url.strip()
    story_id = extract_id(url)

    if story_id:

        print(
            f"Crawl {i+1}/{len(urls)} - ID {story_id}"
        )
        story = get_story(story_id)
        if story:

            stories.append(story)

            # Backup mỗi 20 truyện
            if len(stories) % 20 == 0:

                df_backup = pd.DataFrame(stories)

                df_backup.to_excel(
                    "../data/raw/stories_backup.xlsx",
                    index=False
                )

                print(
                    f"Đã backup {len(stories)} truyện"
                )
    # nghỉ tránh spam request
    time.sleep(3)

# tạo dataframe cuối

df = pd.DataFrame(stories)
df.to_excel(
    "../data/raw/stories_raw.xlsx",
    index=False
)

print("Hoàn thành!")
print("Số truyện:", len(df))