from sklearn.model_selection import train_test_split

from src.config import ProjectConfig


class ValidationSplitter:
    """
    Creates reproducible train/validation splits.
    """

    def __init__(self, config: ProjectConfig = ProjectConfig):
        self.config = config

    def split(self, train_df):
        train_split, val_split = train_test_split(
            train_df,
            test_size=self.config.VALIDATION_SIZE,
            random_state=self.config.RANDOM_STATE,
            shuffle=True,
        )

        return train_split.reset_index(drop=True), val_split.reset_index(drop=True)