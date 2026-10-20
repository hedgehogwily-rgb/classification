import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

REPORTS = Path("reports")


def load_data(file_path):
    return pd.read_csv(file_path)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = load_data("data/financial_phrasebank.csv")
    data = data.dropna(subset=["text"])

    print(data.shape)
    print(data.head())
    print(data.info())
    print(data.isnull().sum())
    print(data["sentiment"].value_counts())

    sns.countplot(data=data, x="sentiment")
    plt.title("Распределение классов")
    plt.tight_layout()
    plt.savefig(REPORTS / "sentiment_distribution.png")
    plt.close()

    data["text_length"] = data["text"].str.len()
    data["word_count"] = data["text"].str.split().str.len()
    print(data[["text_length", "word_count"]].describe())

    data["text_length"].hist(bins=50)
    plt.xlabel("Длина текста")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(REPORTS / "text_length_distribution.png")
    plt.close()

    for sentiment in data["sentiment"].unique():
        print(f"\n=== Класс {sentiment} ===")
        print(data[data["sentiment"] == sentiment]["text"].iloc[0])


if __name__ == "__main__":
    main()
