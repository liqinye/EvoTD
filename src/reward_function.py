import re
import numpy as np

from typing import Dict, List, Sequence

from utils.python_executor import PythonExecutor
from utils.parsers import strip_think_token

def format_reward(task_type: str, predict: str):
    predict = predict.strip()
    pattern = re.compile(r"(?s).*?</think>\s*<answer>(?s).*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    if format_match:
        return 1.0
    else:
        return 0.0


def extract_answer(task_type: str, predict: str, BASE: bool = False):
    if task_type == "code_out":
        keyword = ["output"]
    elif task_type == "code_in":
        keyword = ["input"]
    elif task_type == "code_func":
        keyword = ["python"]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    predict = predict.strip()
    flags = re.DOTALL | re.IGNORECASE

    if BASE:
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict, flags=flags)
        search_space = answer_match.group(1) if answer_match else strip_think_token(predict)
    else:
        search_space = strip_think_token(predict)

    def _fallback_extract(extracted_content: str, return_input: bool = True) -> str:
        flags_local = re.DOTALL | re.IGNORECASE
        if return_input:
            matches = list(re.finditer(r"# Input:\s*(.*?)(?=\n```|$)", extracted_content, flags_local))
            if not matches:
                matches = list(re.finditer(r'input\s*\((.*?)\)', extracted_content, flags_local))
            if not matches:
                matches = list(re.finditer(r"<input>\s*(.*?)(?:</input>|\s*$)", extracted_content, flags_local))
            if not matches:
                matches = list(re.finditer(r"the input is\s*(.*?)\.?$", extracted_content, flags_local))
            return matches[-1].group(1) if matches else extracted_content

        # return output
        matches = list(re.finditer(r"# Output:\s*(.*?)(?=\n```|$)", extracted_content, flags_local))
        if not matches:
            matches = list(re.finditer(r'output\s*\((.*?)\)', extracted_content, flags_local))
        if not matches:
            matches = list(re.finditer(r"<output>\s*(.*?)(?:</output>|\s*$)", extracted_content, flags_local))
        if not matches:
            matches = list(re.finditer(r"the output is\s*(.*?)\.?$", extracted_content, flags_local))
        return matches[-1].group(1) if matches else extracted_content
        
    answers = [None]*len(keyword)

    for idx, kw in enumerate(keyword):
        # allow optional whitespace/newline after the opening ```<kw>
        pattern = fr"```{kw}\s*\n?(.*?)\n?```"
        match = re.search(pattern, search_space, flags)

        if match:
            answers[idx] = match.group(1).strip()

    if len(answers) == 1 and answers[0] is None:
        if task_type == "code_in":
            answers[0] = _fallback_extract(search_space, return_input=True)
        elif task_type == "code_out":
            answers[0] = _fallback_extract(search_space, return_input=False)
        else:
            answers[0] = search_space.strip()

    return answers[0] if len(answers)==1 else answers


def accuracy_reward(
    task_type: str,
    code: str,
    answer: str | Sequence[str] | Dict[str, str],
    ground_truth: Dict[str, List[str] | str] | str,
    executor: PythonExecutor,
) -> float:
    if task_type == "code_out":
        reward = executor.eval_output_prediction(code=code, gold_output=ground_truth, agent_output=answer)
    elif task_type == "code_in":
        reward = executor.eval_input_prediction(code=code, gold_output=ground_truth, agent_input=answer)
    elif task_type == "code_func":
        rewards = []
        for input, output in ground_truth:
            reward = executor.eval_input_prediction(code=answer, gold_output=output, agent_input=input)
            rewards.append(reward)
        reward = np.mean(rewards)
    return reward


def compute_score(task_types: List[str], codes: List[str], predicts: List[str], ground_truths: List[str], BASE: bool = False) -> List[Dict[str, float]]:
    """
    For code_in & code_out tasks, codes, predicts, ground_truths are all necessary.
    For code_func tasks, codes is redundant. Predicts represents the model generated code. Ground truths are input-output pairs.
    """

    executor = PythonExecutor(ast_check=True, timeout_length=10)
    scores = []
    for task_type, code, predict, ground_truth in zip(task_types, codes, predicts, ground_truths):
        format_score = format_reward(task_type, predict)
        answer = extract_answer(task_type, predict, BASE)
        accuracy_score = accuracy_reward(task_type, code, answer, ground_truth, executor)


        if BASE:
            formatted_ok = format_score == 1.0
            if not formatted_ok:
                overall = -1.0  # formatting errors
            elif accuracy_score > 0:
                overall = accuracy_score
            else:
                overall = -0.5  # wrong but well-formatted
        else:
            overall = accuracy_score
            
        scores.append({
            "overall": overall,
            "format": format_score,
            "accuracy": accuracy_score,
        })
    return scores