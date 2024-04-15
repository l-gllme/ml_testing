import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def open_df(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def show_df(df: pd.DataFrame):

    pass


def main():
    try:
        df = open_df("text.csv")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
    import os
