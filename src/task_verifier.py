import json
import logging
import litellm
import re
import random
import os
import numpy as np

from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.python_executor import PythonExecutor
from utils.utils import format_messages, save_jsonl
from utils.parsers import strip_json, strip_think_token
from prompts import skill_reflection_prompt, get_code_problem_predictor_prompt, instruction_following, instruction_following_system_prompt

logger = logging.getLogger("TaskVerifier")

SKILL_REASON = "Proposed task does not mainly focused on target skill(s)."
DIFFICULTY_REASON = "Proposed task is either too easy or too hard."

MODEL_MAP = {
    "o4-mini": "azure/o4-mini",
}

class TaskVerifier:
    def __init__(
        self,
        proposer: str,
        solver: LLM,
        sampling_params: SamplingParams,
        tokenizer: AutoTokenizer,
        skill_candidates: List[str],
        BASE: bool = False,
    ):
        self.proposer = MODEL_MAP[proposer]
        self.solver = solver
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.executor = PythonExecutor(ast_check=True, timeout_length=10)
        self.skill_candidates = skill_candidates
        self.BASE = BASE
    
    def verify_task(
        self,
        tasks: List[Dict],
        task_type: str,
        banned_keywords: List[str],
        check_determinism: bool,
        solver_batch_size: int,
        proposer_batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        curriculum: bool = False,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ):
        valid_tasks, invalid_tasks = [], []
        # Verify task proposal
        for task in tasks:
            if task_type in ["code_in", "code_out"]:
                if task is not None and task.get("code"):
                    valid_tasks.append(task)
                else:
                    if task is None:
                        logger.info(f"Task is None")
                    else:
                        logger.info(f"{task['skill']} {task['task_type']} failed at PROPOSAL")
                        invalid_tasks.append(task)
            elif task_type == "code_func":
                if task is not None and task.get("problem"):
                    valid_tasks.append(task)
                else:
                    if task is None:
                        logger.info(f"Task is None")
                    else:
                        logger.info(f"{task['skill']} {task['task_type']} failed at PROPOSAL")
                        invalid_tasks.append(task)

        invalid_skill_list = []
        if task_type in ["code_in", "code_out"]:
            if not curriculum:  # No need to verify skill for curriculum tasks
                valid_skill_tasks, invalid_skill_tasks = self._verify_skill(
                    tasks=valid_tasks,
                    batch_size=proposer_batch_size,
                    max_tokens=max_tokens,
                )

                valid_tasks = valid_skill_tasks
                # Preserve more data points for training
                invalid_skill_list = [it["skill"] for it in invalid_skill_tasks]
                # Although the task has invalid skill, we still proceed to difficulty verification.
                # If it passes difficulty verification, it will be kept for training.
                valid_tasks += invalid_skill_tasks
                invalid_tasks += invalid_skill_tasks

        valid_execution_tasks, invalid_execution_tasks = [], []
        # Verify task code execution
        for task in valid_tasks:
            success, task = self._verify_execution(
                                task,
                                banned_keywords,
                                check_determinism
                            )
            if not success:
                logger.info(f"{task['skill']} {task['task_type']} failed at CODE EXECUTION")
                # Avoid duplicate invalid skill
                if task["skill"] not in invalid_skill_list:
                    invalid_execution_tasks.append(task)
            else:
                valid_execution_tasks.append(task)
        
        valid_tasks = valid_execution_tasks
        invalid_tasks += invalid_execution_tasks

        # Verify task difficulty
        valid_difficulty_tasks, invalid_difficulty_tasks = self._verify_difficulty(
            valid_tasks, solver_batch_size, invalid_skill_list, lower_bound=lower_bound, upper_bound=upper_bound)

        valid_tasks = valid_difficulty_tasks
        invalid_tasks += invalid_difficulty_tasks

        return valid_tasks, invalid_tasks

    def _verify_difficulty(
        self, 
        tasks: List[Dict], 
        solver_batch_size: int, 
        invalid_skill_list: List[str],
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ):
        valid_tasks = []
        invalid_tasks = []

        for task_batch in self._iter_batches(tasks, solver_batch_size):
            batch_task, batch_hidden_inputs_outputs = self._prepare_solver_batch(task_batch)
            if self.BASE:
                inputs = [
                    p[0]["content"]
                    for p in batch_task
                ]
            else:
                inputs = [
                    self.tokenizer.apply_chat_template(
                        p,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                    for p in batch_task
                ]
            outputs = self.solver.generate(inputs, self.sampling_params)
            batch_answers = []
            for output in outputs:
                k_answers = []
                for res in output.outputs:
                    k_answers.append(res.text)
                batch_answers.append(k_answers)

            for task, answer, hidden_inputs_outputs in zip(task_batch, batch_answers, batch_hidden_inputs_outputs):
                k_extract_answer = []
                task_type = task["task_type"]
                for ans in answer:
                    k_extract_answer.append(self._extract_answer(task_type, ans))
                score = self._eval_answers(task, k_extract_answer, hidden_inputs_outputs)
                task["difficulty"] = score
                too_easy_or_hard = False
                if task_type in ["code_in", "code_out", "code_func"]:
                    if score <= lower_bound or score >= upper_bound:
                        too_easy_or_hard = True

                if too_easy_or_hard:
                    # Avoid duplicate invalid skill
                    if task["skill"] not in invalid_skill_list:
                        logger.info(f"{task['skill']} {task['task_type']} failed at DIFFICULTY: {score}")
                        task["reason"] = DIFFICULTY_REASON
                        task["feedback"] = f"The success rate for solver to solve the task is {score} (out of 10 attempts). {lower_bound} means too hard; {upper_bound} means too easy."
                        invalid_tasks.append(task)
                else:
                    logger.info(f"{task['skill']} {task['task_type']} passed at DIFFICULTY: {score}")
                    valid_tasks.append(task)
        return valid_tasks, invalid_tasks
            

    def _verify_execution(
        self,
        task: Dict,
        banned_keywords: List[str],
        check_determinism: bool
    ):
        if task["task_type"] in ["code_in", "code_out"]:
            success, output = self.executor.check_all(
                task["code"],
                task["input"],
                banned_keywords,
                check_determinism=check_determinism,
                imports=list(set(task["imports"]))    
            )
            if success:
                task["output"] = output
        elif task["task_type"] == "code_func":
            outputs = []
            for input in task["inputs"]:
                success, output = self.executor.check_all(
                    task["code"],
                    input,
                    banned_keywords,
                    check_determinism=check_determinism,
                    imports=list(set(task["imports"]))    
                )
                if not success:
                    return False, task
                outputs.append(output)
            task["outputs"] = outputs
        return success, task


    def _verify_skill(
        self, 
        tasks: List[Dict],
        batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        valid_tasks: List[Dict[str, Any]] = []
        invalid_tasks: List[Dict[str, Any]] = []

        for batch in tqdm(
            self._iter_batches(tasks, batch_size),
            total=self._batch_count(tasks, batch_size),
            desc="Verifying skills",
        ):
            batch_tasks = list(batch)
            messages = [
                format_messages(
                    prompt=skill_reflection_prompt.format(
                        skill_list="\n".join(self.skill_candidates),
                        code_snippet=task["code"],
                    )
                )
                for task in batch_tasks
            ]

            raw = self._thread_call(messages=messages, max_tokens=max_tokens)
            v, inv = self._process_skill_responses(batch_tasks, raw)
            valid_tasks.extend(v)
            invalid_tasks.extend(inv)

        return valid_tasks, invalid_tasks

    def _thread_call(
        self,
        messages: List[List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
    ) -> List[Optional[str]]:
        try:
            responses = litellm.batch_completion(
                model=self.proposer,
                messages=messages,
                max_tokens=max_tokens,
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASE"),
                api_version=os.getenv("AZURE_API_VERSION"),
                request_timeout=60,
                num_retries=10,
            )
            return [res["choices"][0]["message"]["content"] for res in responses]
        except Exception:
            logger.exception("Error in thread call")
            return [None] * len(messages)

    def _process_skill_responses(
        self,
        tasks: List[Dict[str, Any]],
        raw_responses: Optional[List[Optional[str]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        valid_tasks: List[Dict[str, Any]] = []
        invalid_tasks: List[Dict[str, Any]] = []
        raw_responses = raw_responses or []

        if len(raw_responses) != len(tasks):
            logger.warning(
                "Mismatch between requests and responses: %s vs %s",
                len(tasks),
                len(raw_responses),
            )
            raw_responses = list(raw_responses) + [None] * (len(tasks) - len(raw_responses))

        for task, res in zip(tasks, raw_responses):
            if not res:
                logger.exception(
                    "Empty response for skill %s task type %s",
                    task.get("skill"),
                    task.get("task_type"),
                    exc_info=(ValueError, ValueError("Empty response"), None),
                )
                task["detected_skill"] = None
                invalid_tasks.append(task)
                continue

            try:
                parsed = json.loads(strip_json(res))
                detect_skill = [parsed.get("main_skill")] + parsed.get("other_skills", [])
                task["detected_skill"] = detect_skill

                if task["skill"] in detect_skill:
                    valid_tasks.append(task)
                else:
                    logger.info(f"{task['skill']} {task['task_type']} failed at SKILL: detect {detect_skill}")
                    task["reason"] = SKILL_REASON
                    task["feedback"] = (
                        f"The skills detected in the code snippet are {detect_skill}.\n"
                        f"The target skills are {task['skill']}."
                    )
                    invalid_tasks.append(task)
            except Exception:
                logger.exception("Error in JSON loading response")
                invalid_tasks.append(task)

        return valid_tasks, invalid_tasks

    def _iter_batches(self, items: List[Any], batch_size: Optional[int]):
        if not items:
            return
        if not batch_size:
            yield items
            return
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _batch_count(self, items: List[Any], batch_size: Optional[int]) -> int:
        if not items:
            return 0
        if not batch_size:
            return 1
        return (len(items) + batch_size - 1) // batch_size


    def _prepare_solver_batch(self, tasks):
        batch_task = []
        batch_hidden_inputs_outputs = []
        for task in tasks:
            if task["task_type"] in ["code_in", "code_out"]:
                prompt_content = get_code_problem_predictor_prompt(
                    problem_type=task["task_type"], 
                    snippet=task["code"], 
                    input=task["input"], 
                    output=task["output"]
                )
                batch_hidden_inputs_outputs.append(None)
            elif task["task_type"] == "code_func":
                input_output_pairs = [(input, output) for input, output in zip(task["inputs"], task["outputs"])]
                observed_input_output_pairs = input_output_pairs[:len(input_output_pairs) // 2]
                hidden_inputs_outputs = input_output_pairs[len(input_output_pairs) // 2:]
                prompt_content = get_code_problem_predictor_prompt(
                    problem_type=task["task_type"], 
                    snippet=task["code"], 
                    input_output_pairs=observed_input_output_pairs, 
                    problem=task["problem"]
                )
                # For code_in and code_out, hidden_inputs_outputs is empty
                batch_hidden_inputs_outputs.append(hidden_inputs_outputs)
            if self.BASE:
                prompt_content = instruction_following.format(prompt_content)
            if self.BASE:
                batch_task.append(
                    [
                        {"role": "user", "content": prompt_content}
                    ]
                )
            else:
                batch_task.append(
                    [{"role": "user", "content": prompt_content}]
                )
        return batch_task, batch_hidden_inputs_outputs


    def _extract_answer(self, task_type: str, response: str) -> str:
        """
        Extract the output from the model response for code_output tasks.
        The response should contain ```output ... ``` blocks.
        """
        if task_type == "code_out":
            keyword = ["output"]
        elif task_type == "code_in":
            keyword = ["input"]
        elif task_type == "code_func":
            keyword = ["python"]

        cleaned_response = strip_think_token(response)

        # Prefer content inside <answer>...</answer> when the instruction_following
        # template is used; otherwise fall back to the whole cleaned response.
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            cleaned_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        search_space = answer_match.group(1).strip() if answer_match else cleaned_response

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

        answers = [None] * len(keyword)
        for idx, kw in enumerate(keyword):
            pattern = fr"```{kw}\s*\n?(.*?)\n?```"
            
            # Use flags for case-insensitive matching and dotall
            flags = re.DOTALL | re.IGNORECASE
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

        return answers[0] if len(answers) == 1 else answers

    
    def _eval_answers(self, task: Dict, answers, hidden_inputs_outputs: List[Tuple[str, str]]):
        task_type = task["task_type"]
        if task_type == "code_out":
            scores = self.executor.eval_k_output_prediction(
                code=task["code"],
                gold_output=task["output"],
                k_agent_outputs=answers,
            )
        elif task_type == "code_in":
            scores = self.executor.eval_k_input_prediction(
                code=task["code"],
                gold_output=task["output"],
                k_agent_inputs=answers,
            )
        elif task_type == "code_func":
            scores = []
            for answer in answers:
                answer_scores = []
                for input, output in hidden_inputs_outputs:
                    score = self.executor.eval_input_prediction(
                        code=answer,
                        gold_output=output,
                        agent_input=input
                    )
                    answer_scores.append(score)
                scores.append(np.mean(answer_scores))
        return np.mean(scores)
