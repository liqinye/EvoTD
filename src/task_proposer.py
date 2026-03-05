import json
import logging
import litellm
import os
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from itertools import repeat
from collections import defaultdict

from utils.parsers import (
    parse_code_input_output,
    parse_inputs_message,
    parse_hint_message,
    parse_mutations,
    parse_crossover,
    parse_curriculum,
)
from utils.utils import format_messages
from prompts import generate_task_prompt


logger = logging.getLogger("TaskProposer")

MODEL_MAP = {
    "o4-mini": "azure/o4-mini",
}

class TaskProposer:
    def __init__(
        self,
        model: str,
        skill_list: List[Dict[str, Any]],
        attribute_list: List[Dict[str, Any]],
    ):
        self.model = MODEL_MAP[model]
        self.skill_list = skill_list
        self.attribute_list = attribute_list

    def propose_task(
        self,
        propose_type: str,
        task_type: str,
        task_in: List[Any],
        batch_size: int = 1,
        reject_info: Optional[List[Any]] = None,
        complexity_attributes: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        banned_keywords: Optional[List[str]] = None,
        remove_after_return: Optional[bool] = None,
        remove_comments: Optional[bool] = None,
        remove_print: Optional[bool] = None,
        num_inputs: Optional[int] = None,
        remove_input_from_snippet: Optional[bool] = None,
        **kwargs,
    ):
        if propose_type not in {"init", "mutate", "crossover", "curriculum"}:
            raise ValueError(f"Unsupported propose_type: {propose_type}")

        existing_combinations = None
        if propose_type == "crossover":
            existing_combinations = self._build_existing_combinations(task_in)
            # Only single unique single skill is needed for crossover
            task_in = self.skill_list

        normalized_rejects = self._normalize_reject_info(reject_info, len(task_in))
        paired = list(zip(task_in, normalized_rejects))

        responses: List[Optional[Dict[str, Any]]] = []
        for batch in tqdm(
            self._iter_batches(paired, batch_size),
            total=self._batch_count(paired, batch_size),
            desc="Proposing tasks",
        ):
            contexts = self._build_batch_contexts(
                propose_type=propose_type,
                batch=batch,
                task_type=task_type,
                banned_keywords=banned_keywords,
                remove_after_return=remove_after_return,
                num_inputs=num_inputs,
                remove_input_from_snippet=remove_input_from_snippet,
                complexity_attributes=complexity_attributes,
                existing_combinations=existing_combinations,
                **kwargs,
            )

            raw = self._thread_call(messages=contexts["messages"], max_tokens=max_tokens)
            responses.extend(
                contexts["parser"](
                    raw_responses=raw,
                    tasks=batch,
                    task_type=task_type,
                    num_inputs=num_inputs,
                    remove_after_return=remove_after_return,
                    remove_comments=remove_comments,
                    remove_print=remove_print,
                )
            )

        return responses

    def _build_batch_contexts(
        self,
        propose_type: str,
        batch: List[Any],
        task_type: str,
        banned_keywords: Optional[List[str]],
        remove_after_return: Optional[bool],
        num_inputs: Optional[int],
        remove_input_from_snippet: Optional[bool],
        complexity_attributes: Optional[List[str]],
        existing_combinations: Optional[Dict[Tuple[str, ...], Dict[frozenset, str]]],
        **kwargs,
    ) -> Dict[str, Any]:
        messages: List[List[Dict[str, Any]]] = []
        skill_meta: List[Any] = []
        skill_description_meta: List[Any] = []
        category_meta: List[Any] = []
        prev_code, prev_input, prev_performance = None, None, None

        if propose_type == "init":
            parser = self._parse_task_responses
            for task, rejection in batch:
                skill_meta.append(task["skill"])
                skill_description_meta.append(task["skill_description"])
                category_meta.append(task["category"])
                # iterative way
                if "code" in task and task_type in ["code_out", "code_in"]:
                    logger.info("Iterative proposing based on previous task")
                    prev_code = task["code"]
                    prev_input = task["input"]
                    prev_performance = task["difficulty"]
                    
                messages.append(
                    format_messages(
                        prompt=generate_task_prompt(
                            task_type=task_type,
                            tasks=task,
                            banned_keywords=banned_keywords,
                            remove_after_return=remove_after_return,
                            num_inputs=num_inputs,
                            remove_input_from_snippet=remove_input_from_snippet,
                            reject_info=rejection,
                            prev_code=prev_code,
                            prev_input=prev_input,
                            prev_performance=prev_performance,
                        )
                    )
                )

        elif propose_type == "mutate":
            parser = self._parse_mutation_responses
            for task, rejection in batch:
                skill_meta.append(task["skill"])
                skill_description_meta.append(task["skill_description"])
                category_meta.append(task["category"])
                messages.append(
                    format_messages(
                        prompt=generate_task_prompt(
                            task_type=task_type,
                            tasks=task,
                            banned_keywords=banned_keywords,
                            remove_after_return=remove_after_return,
                            num_inputs=num_inputs,
                            remove_input_from_snippet=remove_input_from_snippet,
                            reject_info=rejection,
                            complexity_attributes=complexity_attributes,
                            mutate=True,
                        )
                    )
                )

        elif propose_type == "crossover":
            skill_pool = [f"{skill['skill']}: {skill['skill_description']}" for skill in self.skill_list]
            parser = self._parse_crossover_responses
            for skill, rejection in batch:
                skill_meta.append(skill["skill"])
                skill_description_meta.append(skill["skill_description"])
                category_meta.append(skill["category"])
                messages.append(
                    format_messages(
                        prompt=generate_task_prompt(
                            task_type=task_type,
                            tasks=skill,
                            banned_keywords=banned_keywords,
                            remove_after_return=remove_after_return,
                            num_inputs=num_inputs,
                            remove_input_from_snippet=remove_input_from_snippet,
                            reject_info=rejection,
                            existing_combinations=existing_combinations[skill["skill"]],
                            skill_pool=skill_pool,
                            crossover=True,
                        )
                    )
                )

        elif propose_type == "curriculum":
            assert kwargs["skills"] is not None
            parser = self._parse_curriculum_responses
            for task, rejection in batch:
                messages.append(
                    format_messages(
                        prompt=generate_task_prompt(
                            task_type=task_type,
                            tasks=task,
                            banned_keywords=banned_keywords,
                            remove_after_return=remove_after_return,
                            num_inputs=num_inputs,
                            remove_input_from_snippet=remove_input_from_snippet,
                            cluster_skills=kwargs["skills"],
                            curriculum=True,
                        )
                    )
                )

        return {
            "messages": messages,
            "skill_meta": skill_meta,
            "skill_description_meta": skill_description_meta,
            "category_meta": category_meta,
            "parser": parser,
        }

    def _build_existing_combinations(
        self,
        task_list: List[Dict[str, Any]],
    ):
        # Get the mapping between skill combo and code
        combos = dict()
        for task in task_list:
            if frozenset(task["detected_skill"]) not in combos:
                combos[frozenset(task["detected_skill"])] = task["code"]
        
        # Link skill with its relevant combo
        combinations = defaultdict(dict)
        for skill in [skill["skill"] for skill in self.skill_list]:
            for combo in combos:
                if skill in combo:
                    combinations[skill][combo] = combos[combo]
        return combinations

    def _thread_call(
        self,
        messages: List[List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
    ) -> List[Optional[str]]:
        try:
            responses = litellm.batch_completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASE"),
                api_version=os.getenv("AZURE_API_VERSION"),
                request_timeout=60,
                num_retries=3,
            )
            return [res["choices"][0]["message"]["content"] for res in responses]
        except Exception:
            logger.exception("Error in thread call")
            return [None] * len(messages)

    def _parse_task_responses(
        self,
        raw_responses: Optional[List[Optional[str]]],
        tasks: List[Dict[str, Any]],
        task_type: str,
        num_inputs: Optional[int],
        remove_after_return: Optional[bool],
        remove_comments: Optional[bool],
        remove_print: Optional[bool],
    ) -> List[Optional[Dict[str, Any]]]:
        parsed: List[Optional[Dict[str, Any]]] = []
        raw_responses = raw_responses or []

        if len(raw_responses) != len(tasks):
            logger.warning(
                "Mismatch between requests and responses: %s vs %s",
                len(tasks),
                len(raw_responses),
            )
            raw_responses = list(raw_responses) + [None] * (len(tasks) - len(raw_responses))

        for (task, rejection), res in zip(tasks, raw_responses):
            if not res:
                logger.exception(
                    "Empty response for skill %s category %s",
                    task["skill"],
                    task["category"],
                    exc_info=(ValueError, ValueError("Empty response"), None),
                )
                parsed.append(None)
                continue

            try:
                if task_type == "code_func":
                    if rejection:
                        success, parsed_res = parse_hint_message(res, min_hints=3, max_hints=4)
                    else:
                        success, parsed_res = parse_inputs_message(res, num_inputs=num_inputs)
                else:
                    success, parsed_res = parse_code_input_output(
                        res,
                        parse_output=False,
                        reject_multiple_functions=False,
                        remove_after_return=remove_after_return,
                        remove_comments=remove_comments,
                        remove_print=remove_print,
                    )

                if success:
                    parsed_res["skill"] = task["skill"]
                    parsed_res["skill_description"] = task["skill_description"]
                    parsed_res["category"] = task["category"]
                    parsed_res["task_type"] = task_type
                    if task_type == "code_func":
                        parsed_res["code"] = task["code"]
                        parsed_res["imports"] = task["imports"]
                        if rejection:
                            parsed_res["problem"] = task["problem"]
                            parsed_res["inputs"] = task["inputs"]
                            parsed_res["outputs"] = task["outputs"]
                else:
                    parsed_res = task
                    parsed_res["task_type"] = task_type
            except Exception:
                logger.exception(f"Error in parsing {task['skill']} {task['category']} response")
                parsed_res = task
                parsed_res["task_type"] = task_type

            parsed.append(parsed_res)

        return parsed

    def _parse_mutation_responses(
        self,
        raw_responses: Optional[List[Optional[str]]],
        tasks: List[Dict[str, Any]],
        task_type: str,
        num_inputs: Optional[int],
        remove_after_return: Optional[bool],
        remove_comments: Optional[bool],
        remove_print: Optional[bool],
    ) -> List[Optional[Dict[str, Any]]]:
        parsed_variants: List[Optional[Dict[str, Any]]] = []
        raw_responses = raw_responses or []

        if len(raw_responses) != len(tasks):
            logger.warning(
                "Mismatch between requests and responses: %s vs %s",
                len(tasks),
                len(raw_responses),
            )
            raw_responses = list(raw_responses) + [None] * (len(tasks) - len(raw_responses))

        for (task, rejection), res in zip(tasks, raw_responses):
            if not res:
                logger.exception(
                    "Empty response for skill %s category %s",
                    task["skill"],
                    task["category"],
                    exc_info=(ValueError, ValueError("Empty response"), None),
                )
                parsed_variants.append(None)
                continue

            try:
                success, variants_map = parse_mutations(
                    res,
                    parse_input=True,
                    parse_output=False,
                    reject_multiple_functions=False,
                    remove_after_return=remove_after_return,
                    remove_comments=remove_comments,
                    remove_print=remove_print,
                )
                variants = list(variants_map.values()) if success else []

                if success and variants:
                    for content in variants:
                        content["skill"] = task["skill"]
                        content["skill_description"] = task["skill_description"]
                        content["category"] = task["category"]
                        content["task_type"] = task_type
                        parsed_variants.append(content)
                else:
                    parsed_variants.append({"skill": task["skill"], "skill_description": task["skill_description"], "category": task["category"], "task_type": task_type})
            except Exception:
                logger.exception(f"Error in parsing {task['skill']} {task['category']} response")
                parsed_variants.append({"skill": task["skill"], "skill_description": task["skill_description"], "category": task["category"], "task_type": task_type})

        return parsed_variants

    def _parse_crossover_responses(
        self,
        raw_responses: Optional[List[Optional[str]]],
        tasks: List[Dict[str, Any]],
        task_type: str,
        num_inputs: Optional[int],
        remove_after_return: Optional[bool],
        remove_comments: Optional[bool],
        remove_print: Optional[bool],
    ) -> List[Optional[Dict[str, Any]]]:
        parsed: List[Optional[Dict[str, Any]]] = []
        raw_responses = raw_responses or []

        if len(raw_responses) != len(tasks):
            logger.warning(
                "Mismatch between requests and responses: %s vs %s",
                len(tasks),
                len(raw_responses),
            )
            raw_responses = list(raw_responses) + [None] * (len(tasks) - len(raw_responses))

        for (task, rejection), res in zip(tasks, raw_responses):
            if not res:
                logger.exception(
                    "Empty response for skill %s category %s",
                    task["skill"],
                    task["category"],
                    exc_info=(ValueError, ValueError("Empty response"), None),
                )
                parsed.append(None)
                continue

            try:
                success, parsed_res = parse_crossover(
                    res,
                    parse_input=True,
                    parse_output=False,
                    reject_multiple_functions=False,
                    remove_after_return=remove_after_return,
                    remove_comments=remove_comments,
                    remove_print=remove_print,
                )

                if success:
                    parsed_res["skill"] = task["skill"]
                    parsed_res["skill_description"] = task["skill_description"]
                    parsed_res["category"] = task["category"]
                    parsed_res["task_type"] = task_type
                else:
                    parsed_res = {"skill": task["skill"], "skill_description": task["skill_description"], "category": task["category"], "task_type": task_type}
            except Exception:
                logger.exception(f"Error in parsing {task['skill']} {task['category']} response")
                parsed_res = {"skill": task["skill"], "skill_description": task["skill_description"], "category": task["category"], "task_type": task_type}

            parsed.append(parsed_res)

        return parsed

    def _parse_curriculum_responses(
        self,
        raw_responses: Optional[List[Optional[str]]],
        tasks: List[Dict[str, Any]],
        task_type: str,
        num_inputs: Optional[int],
        remove_after_return: Optional[bool],
        remove_comments: Optional[bool],
        remove_print: Optional[bool],
    ) -> List[Optional[Dict[str, Any]]]:
        parsed_tasks: List[Optional[Dict[str, Any]]] = []
        raw_responses = raw_responses or []

        for res in raw_responses:
            if not res:
                logger.exception(
                    "Empty response",
                    exc_info=(ValueError, ValueError("Empty response"), None),
                )
                parsed_tasks.append(None)
                continue

            try:
                success, curriculum_tasks = parse_curriculum(
                    res,
                    parse_input=True,
                    parse_output=False,
                    reject_multiple_functions=False,
                    remove_after_return=remove_after_return,
                    remove_comments=remove_comments,
                    remove_print=remove_print,
                )

                if success and curriculum_tasks:
                    for tasks in curriculum_tasks:
                        tasks["task_type"] = task_type
                        parsed_tasks.append(tasks)
                else:
                    parsed_tasks.append(None)
            except Exception:
                logger.exception("Error in parsing curriculum response")
                parsed_tasks.append(None)

        return parsed_tasks

    def _write_jsonl(self, path: str, records: List[Dict[str, Any]]):
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

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


    def _normalize_reject_info(
        self,
        reject_info: Optional[List[Any]],
        expected_length: int,
    ) -> List[Any]:
        if reject_info is None:
            return list(repeat(None, expected_length))
        if len(reject_info) != expected_length:
            raise ValueError("reject_info must be same length as data list")
        return list(reject_info)