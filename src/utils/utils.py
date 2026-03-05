import argparse
import logging
import litellm
import re
import yaml
import os
import json

from rich.logging import RichHandler
from typing import Union, List, Dict, Any
import random
from typing import Optional

logger = logging.getLogger(__name__)

def str2bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


def str2list(value: str) -> List[float]:
    return [float(x) for x in value.strip('[]').split(',')]


def setup_logger(level: str = "INFO"):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt="[%X]",
    )

    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def format_messages(
    prompt: str
):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    return messages


def check_batch_status(batch_response_id: str):
    retrieved_batch = litellm.retrieve_batch(
        batch_id=batch_response_id,
        custom_llm_provider="openai"
    )

    logger.info(f"retrieved_batch = {retrieved_batch}")
    if retrieved_batch.output_file_id:
        file_id = retrieved_batch.output_file_id
    elif retrieved_batch.error_file_id:
        file_id = retrieved_batch.error_file_id
    file_content = litellm.file_content(
        file_id=file_id, custom_llm_provider="openai"
    )
    logger.info(f"{batch_response_id} content: {file_content.text}")



def format_cluster_skills(cluster_skills: List[Dict[str, Any]]):
    new_cluster_skills = []
    for category in cluster_skills:
        for member in category["members"]:
            new_cluster_skills.append({
                "skill": member["skill"],
                "category": category["category"],
                "skill_description": member["description"],
            })
    return new_cluster_skills


def assign_invalid_skill_task(invalid_tasks: List[Dict[str, Any]], valid_tasks: List[Dict[str, Any]]):
    turned_invalid_tasks = []
    for invalid_task in invalid_tasks:
        for task in valid_tasks:
            if invalid_task["skill"] in task["detected_skill"]:
                invalid_task["code"] = task["code"]
                invalid_task["input"] = task["input"]
                invalid_task["output"] = task["output"]
                turned_invalid_tasks.append(invalid_task)
                break
    valid_tasks += turned_invalid_tasks
    return valid_tasks, turned_invalid_tasks


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def add_hint_to_task(tasks: Dict[str, Any], hint_count: int):
    new_tasks = []
    for task in tasks:
        new_task = task.copy()
        hint_str = "\n".join([f"Hint {i+1}: {hint}" for i, hint in enumerate(task["hints"][:hint_count])])
        new_task["problem"] = f"{task['problem']}\n\n{hint_str}"
        new_task.pop("hints")
        new_tasks.append(new_task)
    return new_tasks

# Difficulty bins: [0,0.3), [0.3,0.6], (0.6,1.0]
def get_level(difficulty: float, bins: List[float] = [0.0, 0.3, 0.6, 1.0]) -> Optional[str]:
    assert len(bins) == 4
    if bins[0] <= difficulty < bins[1]:
        return "easy"
    elif bins[1] <= difficulty <= bins[2]:
        return "medium"
    elif bins[2] < difficulty <= bins[3]:
        return "hard"
    return None


def sample_examples(
    data: List[Dict[str, Any]],
    task_type: str,
    seed: Optional[int] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Read tasks from a JSONL file, bucket by difficulty, and sample examples.
    Buckets: [0,0.3), [0.3,0.6], (0.6,1.0].
    """
    if seed:
        random.seed(seed)

    buckets: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}

    for item in data:
        difficulty = item.get("difficulty")
        level = get_level(float(difficulty))    
        if level:
            buckets[level].append(item)

    sampled: Dict[str, List[Dict[str, Any]]] = {}
    for level, items in buckets.items():
        if not items:
            sampled[level] = []
            continue
        k = min(10, len(items))
        sampled[level] = random.sample(items, k)
    return sampled

def clean_tasks(tasks: List[Dict[str, Any]]):
    removed = 0
    for task in tasks:
        if task["task_type"] in ["code_in", "code_out"]:
            if ("code" not in task) or ("input" not in task) or ("output" not in task) or ("difficulty" not in task):
                removed += 1
                tasks.remove(task)
        elif task["task_type"] == "code_func":
            if ("problem" not in task) or ("inputs" not in task) or ("outputs" not in task) or ("difficulty" not in task):
                removed += 1
                tasks.remove(task)
    return tasks, removed