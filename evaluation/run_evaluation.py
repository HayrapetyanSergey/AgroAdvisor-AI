import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app import generate_smart_advice


QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "evaluation_questions.csv"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "evaluation_results.csv"


def check_answer(answer, expected):
    return expected.lower() in answer.lower()


def main():
    questions = pd.read_csv(QUESTIONS_PATH)

    results = []

    for _, row in questions.iterrows():
        question = row["question"]
        expected = row["expected_entity"]
        question_type = row["question_type"]

        print(f"Evaluating: {question}")

        answer, data_summary, sources, detected_crop, detected_region, intent = generate_smart_advice(question)

        passed = check_answer(answer, expected)

        results.append({
            "question": question,
            "question_type": question_type,
            "expected_entity": expected,
            "detected_intent": intent,
            "passed": passed,
            "answer": answer[:500]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    accuracy = results_df["passed"].mean() * 100

    print("\nEvaluation completed.")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()