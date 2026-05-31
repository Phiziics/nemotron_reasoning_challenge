from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.data import DataLoader
from src.prompts import PromptBuilder
from src.validation import ValidationSplitter


class SFTDatasetBuilder:
    """
    Creates supervised fine-tuning datasets for the Nemotron reasoning challenge.
    """

    def __init__(self):
        self.config = ProjectConfig
        self.loader = DataLoader()
        self.splitter = ValidationSplitter()
        self.prompt_builder = PromptBuilder()

    def build(self) -> None:
        train, test = self.loader.load_data()

        train_split, val_split = self.splitter.split(train)

        train_sft = self._format_sft_data(train_split)
        val_sft = self._format_sft_data(val_split)
        test_prompts = self._format_test_prompts(test)

        self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_output = self.config.PROCESSED_DATA_DIR / "sft_train.jsonl"
        val_output = self.config.PROCESSED_DATA_DIR / "sft_val.jsonl"
        test_output = self.config.PROCESSED_DATA_DIR / "test_prompts.jsonl"

        train_sft.to_json(train_output, orient="records", lines=True)
        val_sft.to_json(val_output, orient="records", lines=True)
        test_prompts.to_json(test_output, orient="records", lines=True)

        print("SFT datasets created successfully.")
        print(f"Train SFT: {train_sft.shape} -> {train_output}")
        print(f"Validation SFT: {val_sft.shape} -> {val_output}")
        print(f"Test prompts: {test_prompts.shape} -> {test_output}")

    def _format_sft_data(self, df: pd.DataFrame) -> pd.DataFrame:
        formatted = df.copy()

        formatted["text"] = formatted.apply(
            lambda row: self.prompt_builder.build_sft_text(
                prompt=row["prompt"],
                answer=row["answer"],
            ),
            axis=1,
        )

        return formatted[["id", "prompt", "answer", "text"]]

    def _format_test_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        formatted = df.copy()

        formatted["inference_prompt"] = formatted["prompt"].apply(
            self.prompt_builder.build_inference_prompt
        )

        return formatted[["id", "prompt", "inference_prompt"]]


if __name__ == "__main__":
    builder = SFTDatasetBuilder()
    builder.build()