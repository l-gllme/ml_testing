import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def add_label_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column to the dataframe with the category of the label"""

    label_names = {0: "sadness(0)", 1: "joy(1)", 2: "love(2)",
                   3: "anger(3)", 4: "fear(4)", 5: "surprise(5)"}

    df['category'] = df['label'].map(label_names)
    print("\n", df['category'].value_counts(), "\n")

    return df


def plot_category(df: pd.DataFrame):
    """Plot the category of the labels"""

    category_count = df['category'].value_counts()
    colors = plt.cm.viridis(np.linspace(0, 1, len(category_count)))
    category_count.plot(kind='bar', color=colors)

    plt.title('Distribution of Categories')
    plt.xlabel('Category')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('category_distribution.png')
    plt.show()


def open_df(path) -> pd.DataFrame:
    """Open the dataframe from the path"""

    df = pd.read_csv(path)
    if df is None:
        raise Exception("Dataframe is empty")
    else:
        print(f"Dataframe \"{path}\" loaded successfully !")
        return df
