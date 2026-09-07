import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "../../data/queries/all_queries.xlsx"

OUTPUT_DIR = "../../data/queries"

DEV_PATH = os.path.join(
    OUTPUT_DIR,
    "dev_queries.xlsx"
)

TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "test_queries.xlsx"
)


def load_queries():
    queries = pd.read_excel(INPUT_PATH)

    required_columns = [
        "query_id",
        "query",
        "query_type"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in queries.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Thiếu cột: {missing_columns}"
        )

    return queries


def split_queries(queries):
    dev_parts = []
    test_parts = []

    for query_type, group in queries.groupby(
        "query_type"
    ):
        group = group.sample(
            frac=1,
            random_state=42
        )

        n_test = max(
            1,
            round(len(group) * 0.2)
        )

        test_group = group.iloc[
            :n_test
        ]

        dev_group = group.iloc[
            n_test:
        ]

        test_parts.append(test_group)
        dev_parts.append(dev_group)

    dev = pd.concat(
        dev_parts,
        ignore_index=True
    )

    test = pd.concat(
        test_parts,
        ignore_index=True
    )

    dev = dev.sort_values(
        "query_id"
    ).reset_index(drop=True)

    test = test.sort_values(
        "query_id"
    ).reset_index(drop=True)

    return dev, test


def save_results(dev, test):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    dev.to_excel(
        DEV_PATH,
        index=False
    )

    test.to_excel(
        TEST_PATH,
        index=False
    )

    print("\nĐã tạo Dev/Test Split!")

    print(
        f"Dev queries  : {len(dev)}"
    )

    print(
        f"Test queries : {len(test)}"
    )

    print("\n===== Dev theo query_type =====")

    print(
        dev["query_type"]
        .value_counts()
        .sort_index()
    )

    print("\n===== Test theo query_type =====")

    print(
        test["query_type"]
        .value_counts()
        .sort_index()
    )

    print("\n===== Tổng kiểm tra =====")

    print(
        f"Dev + Test   : "
        f"{len(dev) + len(test)}"
    )

    print(
        f"Output Dev   : {DEV_PATH}"
    )

    print(
        f"Output Test  : {TEST_PATH}"
    )


def main():
    queries = load_queries()

    print(
        f"Tổng số queries: {len(queries)}"
    )

    print("\n===== Query type ban đầu =====")

    print(
        queries["query_type"]
        .value_counts()
        .sort_index()
    )

    dev, test = split_queries(
        queries
    )

    save_results(
        dev,
        test
    )


if __name__ == "__main__":
    main()