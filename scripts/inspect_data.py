from src.data import DataLoader
from src.analysis import DataAnalyzer


class DataInspectionRunner:
    """
    Runs first-pass data inspection for the Nemotron reasoning challenge.
    """

    def __init__(self):
        self.loader = DataLoader()
        self.analyzer = DataAnalyzer()

    def run(self) -> None:
        train, test = self.loader.load_data()

        print("=" * 80)
        print("TRAIN SUMMARY")
        print("=" * 80)
        self.loader.summarize_dataframe(train, "Train")

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        self.loader.summarize_dataframe(test, "Test")

        train_with_lengths = self.analyzer.add_text_length_features(train)
        test_with_lengths = self.analyzer.add_text_length_features(test)

        print("\n" + "=" * 80)
        print("TRAIN TEXT LENGTH SUMMARY")
        print("=" * 80)
        self.analyzer.show_text_length_summary(train_with_lengths)

        print("\n" + "=" * 80)
        print("TEST TEXT LENGTH SUMMARY")
        print("=" * 80)
        self.analyzer.show_text_length_summary(test_with_lengths)

        print("\n" + "=" * 80)
        print("SAMPLE TRAINING EXAMPLES")
        print("=" * 80)

        for index in range(min(5, len(train))):
            row = train.iloc[index]

            print("\n" + "-" * 80)
            print(f"Example {index + 1}")
            print("-" * 80)
            print("Prompt:")
            print(row["prompt"])
            print("\nAnswer:")
            print(row["answer"])

        print("\n" + "=" * 80)
        print("SAMPLE TEST EXAMPLES")
        print("=" * 80)

        for index in range(min(3, len(test))):
            row = test.iloc[index]

            print("\n" + "-" * 80)
            print(f"Test Example {index + 1}")
            print("-" * 80)
            print("Prompt:")
            print(row["prompt"])


if __name__ == "__main__":
    runner = DataInspectionRunner()
    runner.run()