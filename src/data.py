import pandas as pd

from src.config import ProjectConfig


class DataLoader:
    """
    Handles loading and basic validation of competition data.
    """

    def __init__(self, config: ProjectConfig = ProjectConfig):
        self.config = config

    def load_train(self) -> pd.DataFrame:
        if not self.config.TRAIN_FILE.exists():
            raise FileNotFoundError(f"Missing train file: {self.config.TRAIN_FILE}")

        return pd.read_csv(self.config.TRAIN_FILE)

    def load_test(self) -> pd.DataFrame:
        if not self.config.TEST_FILE.exists():
            raise FileNotFoundError(f"Missing test file: {self.config.TEST_FILE}")

        return pd.read_csv(self.config.TEST_FILE)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = self.load_train()
        test = self.load_test()
        return train, test

    @staticmethod
    def summarize_dataframe(df: pd.DataFrame, name: str) -> None:
        print(f"{name} shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nMissing values:")
        print(df.isna().sum())

        print("\nPreview:")
        print(df.head())