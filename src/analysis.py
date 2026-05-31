import pandas as pd


class DataAnalyzer:
    """
    Simple EDA utilities for reasoning-puzzle datasets.
    """

    @staticmethod
    def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "prompt" in df.columns:
            df["prompt_char_len"] = df["prompt"].astype(str).str.len()
            df["prompt_word_count"] = df["prompt"].astype(str).str.split().str.len()

        if "answer" in df.columns:
            df["answer_char_len"] = df["answer"].astype(str).str.len()
            df["answer_word_count"] = df["answer"].astype(str).str.split().str.len()

        return df

    @staticmethod
    def show_text_length_summary(df: pd.DataFrame) -> None:
        length_cols = [
            col for col in df.columns
            if "len" in col or "word_count" in col
        ]

        if not length_cols:
            print("No length columns found.")
            return

        print(df[length_cols].describe())