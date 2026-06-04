import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.plotting_utils import COLORS, new_figure, style_axis


def plot_price_history(df: pd.DataFrame, title: str = "Price History"):
    fig, ax = new_figure()
    ax.plot(df.index, df["Close"], color=COLORS["primary"], linewidth=2, label="Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    style_axis(ax, title)
    fig.tight_layout()
    return fig


def plot_sentiment_index(sentiment: pd.Series, title: str = "Investor Sentiment PC1"):
    fig, ax = new_figure()
    colors = [COLORS["secondary"] if value >= 0 else COLORS["danger"] for value in sentiment]
    ax.bar(sentiment.index, sentiment.values, color=colors, alpha=0.8)
    ax.axhline(0, color=COLORS["muted"], linewidth=1)
    ax.set_ylabel("PC1")
    style_axis(ax, title)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(actual, predicted, title: str = "Actual vs Predicted"):
    fig, ax = new_figure()
    ax.plot(actual.index, actual, label="Actual", color=COLORS["primary"], linewidth=2)
    ax.plot(predicted.index, predicted, label="Predicted", color=COLORS["danger"], linewidth=2, linestyle="--")
    ax.axhline(0, color=COLORS["muted"], linewidth=1)
    ax.set_xlabel("Date")
    ax.legend()
    style_axis(ax, title)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(data: pd.DataFrame, title: str = "Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_heatmap(data: pd.DataFrame):
    return plot_correlation_heatmap(data)
