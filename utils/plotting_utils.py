import platform

import matplotlib.pyplot as plt


if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


COLORS = {
    "primary": "#2563eb",
    "secondary": "#16a34a",
    "danger": "#dc2626",
    "muted": "#64748b",
    "grid": "#e2e8f0",
}


def style_axis(ax, title: str | None = None) -> None:
    if title:
        ax.set_title(title)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def new_figure(figsize: tuple[int, int] = (10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
