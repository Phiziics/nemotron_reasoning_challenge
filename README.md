# NVIDIA Nemotron Model Reasoning Challenge

## Introduction

This project is built for the Kaggle NVIDIA Nemotron Model Reasoning Challenge. The goal is to improve reasoning performance on puzzle-style prompts using a disciplined AI engineering workflow.

The project follows a competition-grade process:

1. Understand the task and evaluation rules
2. Inspect the data carefully
3. Build a valid baseline submission
4. Train and validate a LoRA adapter
5. Improve through prompt formatting, feature/error analysis, and focused training
6. Document results clearly for portfolio and recruiter review

## Project Structure

```text
nemotron_reasoning_challenge/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_sft.ipynb
│   ├── 03_improved_sft.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── prompts.py
│   ├── validation.py
│   └── analysis.py
├── outputs/
│   ├── models/
│   ├── adapters/
│   ├── submissions/
│   └── reports/
├── requirements.txt
├── requirements-lock.txt
└── README.md