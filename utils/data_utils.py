from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, **kwargs)


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")


def drop_missing_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    validate_columns(df, columns)
    return df.dropna(subset=columns).copy()
