from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(TP: int, FN: int, FP: int, TN: int) -> None:
    ticklabels = ("Minority", "Majority")
    sns.heatmap(((TP, FN), (FP, TN)), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)

    plt.title("Confusion matrix")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()


def split_csv(fp: str = "./data/creditcard.csv", fp_dest: str = "./data",
              name: str = "credit", test_size: int = 0.2, strat_col: str = "Class") -> None:
    """
    Splits a csv file in two, stratified.
    """
    df = pd.read_csv(fp)
    train, test = train_test_split(df, test_size=0.2, stratify=df[strat_col])

    for i, chunk in enumerate((train, test)):
        chunk.to_csv(f"{fp_dest}/{name}{i}.csv", index=False)


if __name__ == "__main__":
    split_csv()
