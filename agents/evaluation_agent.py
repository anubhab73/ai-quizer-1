# agents/evaluation_agent.py
from utils.compare_utils import hybrid_evaluate
import re

def evaluate_answers(questions_data, student_answers_data: dict):
    student = student_answers_data.get("individual_answers", [])
    questions = questions_data.get("questions", {})
    answers = questions_data.get("answers", {})

    results = []
    total_score = 0
    qcount = 0
    idx = 0

    # === MCQs: Exact match (10 or 0) ===
    for q, a in zip(questions.get("mcqs", []), answers.get("mcqs", [])):
        model_raw = str(a.get("model_answer", "")).strip().upper()
        correct_letter = re.search(r"[A-D]", model_raw)
        correct_letter = correct_letter.group(0) if correct_letter else model_raw[0]

        student_raw = student[idx] if idx < len(student) else ""
        student_letter = student_raw[:1].upper() if student_raw else ""

        score = 10.0 if student_letter == correct_letter else 0.0

        results.append({
            "question_number": idx + 1,
            "question_type": "mcq",
            "question": q.get("question", ""),
            "student_answer": student_raw,
            "model_answer": correct_letter,
            "evaluation": {
                "score": score,
                "feedback": "Correct!" if score > 0 else "Incorrect answer.",
                "weak_areas": [] if score > 0 else ["Wrong option selected"]
            }
        })
        total_score += score
        qcount += 1
        idx += 1

    # === Short & Long: Hybrid (embedding score + LLM feedback) ===
    for typ in ("shorts", "longs"):
        for q, a in zip(questions.get(typ, []), answers.get(typ, [])):
            model_ans = a.get("model_answer", "")
            student_ans = student[idx] if idx < len(student) else ""

            eval_result = hybrid_evaluate(model_ans, student_ans)

            results.append({
                "question_number": idx + 1,
                "question_type": typ[:-1],
                "question": q.get("question", ""),
                "student_answer": student_ans,
                "model_answer": model_ans,
                "evaluation": eval_result
            })
            total_score += eval_result["score"]
            qcount += 1
            idx += 1

    avg = round(total_score / qcount, 1) if qcount > 0 else 0

    return {
        "avg_score": avg,
        "total_questions": qcount,
        "results": results
    }