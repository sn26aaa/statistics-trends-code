"""
Statistics and Trends Assignment
--------------------------------
This script performs preprocessing, visualization, and statistical
analysis on the StudentsPerformance dataset.
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns



def plot_relational_plot(df):
    """
    Creates and saves a relational plot.
    """
    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x='math score',
        y='reading score',
        ax=ax
    )

    ax.set_title('Math Score vs Reading Score')
    ax.set_xlabel('Math Score')
    ax.set_ylabel('Reading Score')

    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.show()
    plt.close()


def plot_categorical_plot(df):
    """
    Creates and saves a categorical plot.
    """
    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x='gender',
        y='math score',
        ax=ax
    )

    ax.set_title('Math Score by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Math Score')

    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.show()
    plt.close()


def plot_statistical_plot(df):
    """
    Creates and saves a statistical distribution plot.
    """
    fig, ax = plt.subplots()

    sns.histplot(
        df['math score'],
        kde=True,
        ax=ax
    )

    ax.set_title('Distribution of Math Scores')
    ax.set_xlabel('Math Score')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.show()
    plt.close()


def statistical_analysis(df, col: str):
    """
    Calculates statistical moments for a selected column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy='omit')
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Performs basic preprocessing and exploratory analysis.
    """
    print('First five rows of the dataset:')
    print(df.head())
    print()

    print('Last five rows of the dataset:')
    print(df.tail())
    print()

    print('Statistical summary:')
    print(df.describe())
    print()

    print('Correlation matrix:')
    print(df.corr(numeric_only=True))
    print()

    df = df.dropna()

    return df


def writing(moments, col):
    """
    Prints a written interpretation of the statistical moments.
    """
    print(f'For the attribute {col}:')
    print(
        f'Mean = {moments[0]:.2f}, '
        f'Standard Deviation = {moments[1]:.2f}, '
        f'Skewness = {moments[2]:.2f}, and '
        f'Excess Kurtosis = {moments[3]:.2f}.'
    )

    if moments[2] > 2:
        skewness = 'right skewed'
    elif moments[2] < -2:
        skewness = 'left skewed'
    else:
        skewness = 'not skewed'

    if moments[3] > 2:
        kurtosis = 'leptokurtic'
    elif moments[3] < -2:
        kurtosis = 'platykurtic'
    else:
        kurtosis = 'mesokurtic'

    print(f'The data was {skewness} and {kurtosis}.')


def main():
    """
    Main execution function.
    """
    df = pd.read_csv('StudentsPerformance.csv')
    df = preprocessing(df)

    col = 'math score'

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
