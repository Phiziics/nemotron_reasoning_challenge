import re
import pandas as pd


class PuzzleTypeDetector:
    """
    Detects high-level puzzle categories from prompt text.

    This is not meant to solve the puzzle.
    It gives us visibility into what kinds of reasoning tasks exist.
    """

    @staticmethod
    def detect(prompt: str) -> str:
        prompt_lower = str(prompt).lower()

        if "bit manipulation" in prompt_lower or "8-bit binary" in prompt_lower:
            return "bit_manipulation"

        if "decrypt" in prompt_lower or "secret encryption" in prompt_lower:
            return "text_decryption"

        if "numeral system" in prompt_lower or "roman" in prompt_lower:
            return "numeral_conversion"

        if "unit" in prompt_lower or "converted" in prompt_lower:
            return "unit_conversion"

        if "equation" in prompt_lower:
            return "equation"

        if "gravity" in prompt_lower:
            return "gravity"

        if "cryptarithm" in prompt_lower:
            return "cryptarithm"

        return "unknown"

    @classmethod
    def add_puzzle_type(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["puzzle_type"] = df["prompt"].apply(cls.detect)
        return df