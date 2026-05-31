class PromptBuilder:
    """
    Builds prompt formats for supervised fine-tuning and inference.
    """

    @staticmethod
    def build_sft_text(prompt: str, answer: str) -> str:
        return f"""You are solving a logical reasoning puzzle.

Read the puzzle carefully.
Identify the transformation rule.
Apply the rule step by step.
Return only the final answer.

Puzzle:
{prompt}

Final answer:
{answer}"""

    @staticmethod
    def build_inference_prompt(prompt: str) -> str:
        return f"""You are solving a logical reasoning puzzle.

Read the puzzle carefully.
Identify the transformation rule.
Apply the rule step by step.
Return only the final answer.

Puzzle:
{prompt}

Final answer:"""