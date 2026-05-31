import pandas as pd

from src.config import ProjectConfig
from src.data import DataLoader


class BaselineSubmissionBuilder:
    """
    Creates a simple exact-match baseline submission.

    Strategy:
    1. Load train and test.
    2. Match test prompts against train prompts.
    3. If an exact prompt exists in train, use the known answer.
    4. Save submission.csv.
    """

    def __init__(self):
        self.config = ProjectConfig
        self.loader = DataLoader()

    def build(self) -> None:
        train, test = self.loader.load_data()

        self.config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

        prompt_to_answer = dict(zip(train["prompt"], train["answer"]))

        submission = test.copy()
        submission["answer"] = submission["prompt"].map(prompt_to_answer)

        missing_count = submission["answer"].isna().sum()

        if missing_count > 0:
            print(f"Warning: {missing_count} test prompts were not found in train.")
            submission["answer"] = submission["answer"].fillna("")

        output_path = self.config.SUBMISSIONS_DIR / "baseline_exact_match_submission.csv"

        submission[["id", "answer"]].to_csv(output_path, index=False)

        print("Baseline submission created.")
        print(f"Output path: {output_path}")
        print(f"Submission shape: {submission[['id', 'answer']].shape}")
        print(f"Missing answers: {missing_count}")
        print("\nPreview:")
        print(submission[["id", "answer"]])


if __name__ == "__main__":
    builder = BaselineSubmissionBuilder()
    builder.build()