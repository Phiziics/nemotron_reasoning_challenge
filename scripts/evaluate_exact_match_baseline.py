import pandas as pd

from src.data import DataLoader
from src.validation import ValidationSplitter
from src.config import ProjectConfig
from src.puzzle_types import PuzzleTypeDetector


class ExactMatchBaselineEvaluator:
    """
    Evaluates an exact prompt match baseline on a holdout validation split.
    """

    def __init__(self):
        self.loader = DataLoader()
        self.splitter = ValidationSplitter()
        self.config = ProjectConfig
        self.detector = PuzzleTypeDetector()

    def run(self) -> None:
        train = self.loader.load_train()

        train_split, val_split = self.splitter.split(train)

        prompt_to_answer = dict(zip(train_split["prompt"], train_split["answer"]))

        val_results = val_split.copy()
        val_results["prediction"] = val_results["prompt"].map(prompt_to_answer)
        val_results["prediction"] = val_results["prediction"].fillna("")
        val_results["is_correct"] = val_results["prediction"] == val_results["answer"]

        val_results = self.detector.add_puzzle_type(val_results)

        accuracy = val_results["is_correct"].mean()
        missing_rate = (val_results["prediction"] == "").mean()

        print("=" * 80)
        print("EXACT MATCH BASELINE")
        print("=" * 80)
        print(f"Validation rows: {len(val_results)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Missing prediction rate: {missing_rate:.4f}")

        print("\nAccuracy by puzzle type:")
        print(
            val_results.groupby("puzzle_type")["is_correct"]
            .mean()
            .sort_values(ascending=False)
        )

        print("\nCounts by puzzle type:")
        print(val_results["puzzle_type"].value_counts())

        print("\nSample wrong predictions:")
        wrong = val_results[val_results["is_correct"] == False].head(10)

        for _, row in wrong.iterrows():
            print("-" * 80)
            print("Puzzle type:", row["puzzle_type"])
            print("Prompt:")
            print(row["prompt"][:700])
            print("True answer:", row["answer"])
            print("Prediction:", row["prediction"])

        self.config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        output_path = self.config.REPORTS_DIR / "exact_match_validation_results.csv"
        val_results.to_csv(output_path, index=False)

        print("\nSaved validation results:")
        print(output_path)


if __name__ == "__main__":
    evaluator = ExactMatchBaselineEvaluator()
    evaluator.run()