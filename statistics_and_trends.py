"""
Statistics and Trends Assignment
Shows graphs only (no saving)
Updated matrix colour and improved visuals
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_statistical_plot(df):
    """
    1. Distribution of Mathematics Scores
    """
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("No numeric columns available.")
        return

    maths_col = numeric_cols[0]

    plt.figure()
    sns.histplot(df[maths_col], kde=True, color="royalblue")

    plt.title("Distribution of Mathematics Scores", fontsize=14)
    plt.xlabel(maths_col)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    return


def plot_relational_plot(df):
    """
    2. Relationship Between Mathematics and Reading Scores
    """
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) < 2:
        print("Not enough numeric columns.")
        return

    maths_col = numeric_cols[0]
    reading_col = numeric_cols[1]

    plt.figure()
    sns.scatterplot(
        data=df,
        x=maths_col,
        y=reading_col,
        color="darkgreen"
    )

    plt.title("Relationship Between Mathematics and Reading Scores", fontsize=14)
    plt.xlabel(maths_col)
    plt.ylabel(reading_col)

    plt.tight_layout()
    plt.show()
    return


def plot_categorical_plot(df):
    """
    3. Correlation Between Academic Subjects
    """
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) < 2:
        print("Not enough numeric columns.")
        return

    corr = df[numeric_cols].corr()

    plt.figure()
    sns.heatmap(
        corr,
        annot=True,
        cmap="viridis",  
        linewidths=0.5,
        fmt=".2f"
    )

    plt.title("Correlation Between Academic Subjects", fontsize=14)

    plt.tight_layout()
    plt.show()
    return


def statistical_analysis(df, col: str):
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy="omit")
    excess_kurtosis = ss.kurtosis(df[col], nan_policy="omit")

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    print("Columns found:", list(df.columns))
    print(df.head())
    print(df.describe())
    print(df.corr(numeric_only=True))

    df = df.dropna()
    return df


def writing(moments, col):
    print(f"\nFor the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, "
        f"Excess Kurtosis = {moments[3]:.2f}"
    )
    return


def main():
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return

    maths_col = numeric_cols[0]

    plot_statistical_plot(df)
    plot_relational_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, maths_col)
    writing(moments, maths_col)
    return


if __name__ == "__main__":
    main()
