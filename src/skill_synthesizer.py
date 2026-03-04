import logging
import litellm
import json
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from prompts import skill_attribute_prompt, cluster_skill_prompt, cluster_attribute_prompt
from utils.utils import format_messages
from utils.parsers import strip_json

logger = logging.getLogger("SkillSynthesizer")


class SkillSynthesizer:
    def __init__(
        self,
        model: str,
        dataset: List[Dict[str, Any]],
    ):
        self.model = model
        self.dataset = dataset

    def label_skill(
        self,
        call_mode: str,
        skill_out_file_path: str,
        batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self._ensure_thread_mode(call_mode)

        responses = []
        for batch in tqdm(
            self._iter_batches(self.dataset, batch_size),
            total=self._batch_count(self.dataset, batch_size),
            desc="Labeling skills",
        ):
            problem_ids = []
            messages = []
            for sample in batch:
                problem_ids.append(sample["problem_id"])
                messages.append(
                    format_messages(
                        prompt=skill_attribute_prompt.format(
                            problem=sample["problem"],
                            code_solution=sample["solution"],
                        )
                    )
                )

            raw_responses = self._thread_call(messages=messages, max_tokens=max_tokens)
            responses.extend(self._format_skill_responses(problem_ids, raw_responses))

        self._write_jsonl(skill_out_file_path, responses)
        return responses

    def cluster(
        self,
        skills,
        call_mode: str,
        cluster_skill_out_file_path: str,
        cluster_attribute_out_file_path: str,
        max_tokens: Optional[int] = None,
    ):
        self._ensure_thread_mode(call_mode)

        logger.info("Clustering skills...")
        skill_list_str = self._prepare_skill_list(skills)
        self._cluster_and_save(
            prompt_template=cluster_skill_prompt,
            skill_list_str=skill_list_str,
            output_path=cluster_skill_out_file_path,
            max_tokens=max_tokens,
        )

        logger.info("Clustering attributes...")
        attributes = self._cluster_and_save(
            prompt_template=cluster_attribute_prompt,
            skill_list_str=skill_list_str,
            output_path=cluster_attribute_out_file_path,
            max_tokens=max_tokens,
        )

        return attributes

    def _thread_call(
        self,
        messages: List[List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
    ):
        try:
            responses = litellm.batch_completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASE"),
                api_version=os.getenv("AZURE_API_VERSION"),
            )
            return [res["choices"][0]["message"]["content"] for res in responses]
        except Exception:
            logger.exception("Error in thread call")
            return [None] * len(messages)

    
    def _prepare_skill_list(
        self,
        skills,
    ):
        skill_list_str = ""
        for task_skill in skills:
            skill = task_skill["skills"]
            if skill:
                skill_list_str += "\n".join([f"{skill_name}:{description}" for skill_name, description in skill.items()])
        return skill_list_str

    def _format_skill_responses(
        self,
        problem_ids: List[Any],
        responses: Optional[List[Optional[str]]],
    ) -> List[Dict[str, Any]]:
        formatted = []
        responses = responses or []

        if len(responses) != len(problem_ids):
            logger.warning(
                "Mismatch between problems and responses: %s vs %s",
                len(problem_ids),
                len(responses),
            )
            responses = list(responses) + [None] * (len(problem_ids) - len(responses))

        for p_id, res in zip(problem_ids, responses):
            try:
                res = strip_json(res)
                res = json.loads(res)
                res["problem_id"] = p_id
            except Exception:
                logger.exception(f"Error in JSON loading {p_id} response")
                res = {"problem_id": p_id, "skills": None}
            formatted.append(res)

        return formatted

    def _cluster_and_save(
        self,
        prompt_template: str,
        skill_list_str: str,
        output_path: str,
        max_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        messages = [format_messages(prompt=prompt_template.format(skill_list=skill_list_str))]
        response = self._thread_call(messages=messages, max_tokens=max_tokens)

        try:
            if not response:
                logger.exception("Failed to Cluster")
            parsed = json.loads(strip_json(response[0]))
        except Exception:
            logger.exception(f"Error in JSON loading cluster response. {response}")
            raise SystemExit(1)

        self._write_jsonl(output_path, parsed)
        return parsed

    def _write_jsonl(self, path: str, records: List[Dict[str, Any]]):
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _iter_batches(self, items: List[Any], batch_size: Optional[int]):
        if not batch_size:
            if items:
                yield items
            return
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _batch_count(self, items: List[Any], batch_size: Optional[int]) -> int:
        if not batch_size:
            return 1 if items else 0
        return (len(items) + batch_size - 1) // batch_size

    def _ensure_thread_mode(self, call_mode: str):
        if call_mode != "thread":
            raise ValueError("SkillSynthesizer only supports 'thread' call mode.")
