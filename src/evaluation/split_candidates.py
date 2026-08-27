import os
import pandas as pd

# Đường dẫn

CANDIDATE_PATH = "../../results/evaluation/candidate_set.xlsx"

DEV_QUERY_PATH = "../../data/queries/dev_queries.xlsx"
TEST_QUERY_PATH = "../../data/queries/test_queries.xlsx"

OUTPUT_DIR = "../../results/evaluation"

DEV_CANDIDATE_PATH = os.path.join(
    OUTPUT_DIR,
    "dev_candidates.xlsx"
)

TEST_CANDIDATE_PATH = os.path.join(
    OUTPUT_DIR,
    "test_candidates.xlsx"
)


# Đọc dữ liệu

def load_data():

    candidates = pd.read_excel(
        CANDIDATE_PATH
    )

    dev_queries = pd.read_excel(
        DEV_QUERY_PATH
    )

    test_queries = pd.read_excel(
        TEST_QUERY_PATH
    )

    return candidates, dev_queries, test_queries

# Kiểm tra dữ liệu

def validate_data(
    candidates,
    dev_queries,
    test_queries
):

    required_candidate_columns = [
        "query_id",
        "story_id",
        "title",
        "relevance"
    ]

    required_query_columns = [
        "query_id"
    ]

    # Kiểm tra Candidate Set
    missing_candidates = [
        column
        for column in required_candidate_columns
        if column not in candidates.columns
    ]

    if missing_candidates:

        raise ValueError(
            f"Candidate Set thiếu cột: "
            f"{missing_candidates}"
        )

    # Kiểm tra Dev
    missing_dev = [
        column
        for column in required_query_columns
        if column not in dev_queries.columns
    ]

    if missing_dev:

        raise ValueError(
            f"Dev Query thiếu cột: "
            f"{missing_dev}"
        )

    # Kiểm tra Test
    missing_test = [
        column
        for column in required_query_columns
        if column not in test_queries.columns
    ]

    if missing_test:

        raise ValueError(
            f"Test Query thiếu cột: "
            f"{missing_test}"
        )

# Chia Candidate

def split_candidates(
    candidates,
    dev_queries,
    test_queries
):

    dev_ids = set(
        dev_queries["query_id"]
    )

    test_ids = set(
        test_queries["query_id"]
    )

    # Kiểm tra query bị xuất hiện
    # đồng thời trong Dev và Test

    overlap = dev_ids.intersection(
        test_ids
    )

    if overlap:

        raise ValueError(
            "Có query_id xuất hiện "
            "ở cả Dev và Test: "
            f"{sorted(overlap)}"
        )

    # Candidate thuộc Dev
    dev_candidates = candidates[
        candidates["query_id"].isin(dev_ids)
    ].copy()

    # Candidate thuộc Test
    test_candidates = candidates[
        candidates["query_id"].isin(test_ids)
    ].copy()

    return (
        dev_candidates,
        test_candidates
    )

# Lưu kết quả

def save_results(
    dev_candidates,
    test_candidates
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    dev_candidates.to_excel(
        DEV_CANDIDATE_PATH,
        index=False
    )

    test_candidates.to_excel(
        TEST_CANDIDATE_PATH,
        index=False
    )

    print("\nĐã chia Candidate Set thành công!")

    print(
        f"Dev candidates  : "
        f"{len(dev_candidates)}"
    )

    print(
        f"Test candidates : "
        f"{len(test_candidates)}"
    )

    print(
        f"Tổng candidates : "
        f"{len(dev_candidates) + len(test_candidates)}"
    )

    print("\n===== Query trong Dev =====")

    print(
        dev_candidates["query_id"]
        .nunique()
    )

    print("\n===== Query trong Test =====")

    print(
        test_candidates["query_id"]
        .nunique()
    )

    print("\n===== Output =====")

    print(
        f"Dev  : {DEV_CANDIDATE_PATH}"
    )

    print(
        f"Test : {TEST_CANDIDATE_PATH}"
    )


# Main

def main():

    print("Đang đọc dữ liệu...")

    candidates, dev_queries, test_queries = load_data()

    print(
        f"Candidate Set : "
        f"{len(candidates)}"
    )

    print(
        f"Dev queries   : "
        f"{len(dev_queries)}"
    )

    print(
        f"Test queries  : "
        f"{len(test_queries)}"
    )

    # Kiểm tra
    validate_data(
        candidates,
        dev_queries,
        test_queries
    )

    # Chia
    dev_candidates, test_candidates = split_candidates(
        candidates,
        dev_queries,
        test_queries
    )

    # Kiểm tra tổng
    total = (
        len(dev_candidates)
        + len(test_candidates)
    )

    if total != len(candidates):

        raise ValueError(
            "Số lượng candidate sau khi "
            "chia không khớp Candidate Set ban đầu!"
        )

    # Lưu
    save_results(
        dev_candidates,
        test_candidates
    )


if __name__ == "__main__":

    main()