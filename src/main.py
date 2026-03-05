import argparse
import logging
import datetime
import json
import yaml
import os
import torch

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from skill_synthesizer import SkillSynthesizer
from task_proposer import TaskProposer
from task_verifier import TaskVerifier
from utils.data import load_dataset
from utils.utils import (
    setup_logger, 
    check_batch_status, 
    load_config, 
    format_cluster_skills, 
    assign_invalid_skill_task, 
    save_jsonl,
    add_hint_to_task,
    clean_tasks,
)


logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").propagate = False

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["usaco", "taco"])
    parser.add_argument("--proposer", type=str, required=True)
    parser.add_argument("--solver", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["code_in", "code_out", "code_func"])
    parser.add_argument("--proposer_batch_size", type=int, required=False, default=None)
    parser.add_argument("--solver_batch_size", type=int, required=False, default=None)
    parser.add_argument("--batch_response_id", type=str, required=False, default=None)
    parser.add_argument("--iteration", type=int, required=False, default=0)
    parser.add_argument("--lower_bound", type=float, required=False, default=0.0)
    parser.add_argument("--upper_bound", type=float, required=False, default=1.0)
    return parser.parse_args()


def get_skill(args, config, dataset):    
    skill_synthesizer = SkillSynthesizer(
        model=args.proposer,
        dataset=dataset,
    )
    skills = skill_synthesizer.label_skill(
        batch_size=args.proposer_batch_size,
        skill_out_file_path=os.path.join(config["skill_synthesizer"]["skill_out_file_path"], f"{args.proposer}_{args.dataset}.jsonl"),
        max_tokens=config["skill_synthesizer"]["proposer"][args.proposer]["max_tokens"],
    )

    with open(os.path.join(config["skill_synthesizer"]["skill_out_file_path"], f"{args.proposer}_{args.dataset}.jsonl"), "r") as f:
        label_info = [json.loads(line) for line in f]

    cluster_skills, cluster_attributes = skill_synthesizer.cluster(
        skills=label_info,
        cluster_skill_out_file_path=os.path.join(config["skill_synthesizer"]["cluster_out_file_path"], f"skill_{args.proposer}_{args.dataset}.jsonl"),
        cluster_attribute_out_file_path=os.path.join(config["skill_synthesizer"]["cluster_out_file_path"], f"attribute_{args.proposer}_{args.dataset}.jsonl"),
        max_tokens=config["skill_synthesizer"]["proposer"][args.proposer]["max_tokens"],
    )

    return cluster_skills, cluster_attributes


def get_init_tasks(
    task_proposer,
    task_verifier,
    cluster_skills,
    config,
    args,
    logger,
    lower_bound=0.0,
    upper_bound=1.0,
):
    trial = 0
    valid_task_list = []
    reject_info = None

    skill_list = cluster_skills

    solver_name = config["task_verifier"]["solver"][args.solver].split("/")[-1].lower()

    while trial <= 20:
        tasks =task_proposer.propose_task(
            propose_type="init",
            task_type=args.task_type,
            task_in=skill_list,
            batch_size=args.proposer_batch_size,
            max_tokens=config["task_proposer"]["proposer"][args.proposer]["max_tokens"],
            num_inputs=config["task_proposer"]["num_inputs"],
            banned_keywords=config["task_proposer"]["banned_keywords"],
            remove_after_return=config["task_proposer"]["remove_after_return"],
            remove_comments=config["task_proposer"]["remove_comments"],
            remove_print=config["task_proposer"]["remove_print"],
            reject_info=reject_info,
        )

        valid_task, invalid_task = task_verifier.verify_task(
            tasks=tasks,
            task_type=args.task_type,
            proposer_batch_size=args.proposer_batch_size,
            solver_batch_size=args.solver_batch_size,
            max_tokens=config["task_verifier"]["proposer"][args.proposer]["max_tokens"],
            banned_keywords=config["task_proposer"]["banned_keywords"],
            check_determinism=config["task_verifier"]["check_determinism"],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        save_jsonl(tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/valid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))

        valid_task_list += valid_task

        if invalid_task == []:
            break
        else:
            save_jsonl(invalid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))

        skill_list = [{"skill": task["skill"], "skill_description": task["skill_description"], "category": task["category"]} for task in invalid_task]
        reject_info = [
            (
                {"reason": task.get("reason"), "feedback": task.get("feedback"), "code": task.get("code")}
                if ("reason" in task or "feedback" in task)
                else None
            )
            for task in invalid_task
        ]
        logger.info(f"Trial {trial} | Invalid skill tasks: {[skill['skill'] for skill in skill_list]} | size: {len(skill_list)}")
        trial += 1

    # Reassign invalid skills if they are detected in other skill tasks
    valid_tasks, turned_invalid_tasks = assign_invalid_skill_task(invalid_task, valid_task_list)
    # Save valid tasks
    save_jsonl(valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}.jsonl"))

    return valid_tasks


def get_mutate_tasks(
    task_proposer,
    task_verifier,
    initial_tasks,
    cluster_attributes,
    config,
    args,
    logger,
    lower_bound=0.0,
    upper_bound=1.0,
):
    mutate_valid_task_list = []
    trial = 0
    tasks = initial_tasks
    solver_name = config["task_verifier"]["solver"][args.solver].split("/")[-1].lower()
    while trial <= 3:
        mutate_tasks = task_proposer.propose_task(
            propose_type="mutate",
            task_type=args.task_type,
            task_in=tasks,
            complexity_attributes=cluster_attributes,
            batch_size=args.proposer_batch_size,
            max_tokens=config["task_proposer"]["proposer"][args.proposer]["max_tokens"],
            num_inputs=config["task_proposer"]["num_inputs"],
            banned_keywords=config["task_proposer"]["banned_keywords"],
            remove_after_return=config["task_proposer"]["remove_after_return"],
            remove_comments=config["task_proposer"]["remove_comments"],
            remove_print=config["task_proposer"]["remove_print"]
        )

        valid_task, invalid_task = task_verifier.verify_task(
            tasks=mutate_tasks,
            task_type=args.task_type,
            proposer_batch_size=args.proposer_batch_size,
            solver_batch_size=args.solver_batch_size,
            max_tokens=config["task_verifier"]["proposer"][args.proposer]["max_tokens"],
            banned_keywords=config["task_proposer"]["banned_keywords"],
            check_determinism=config["task_verifier"]["check_determinism"],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        mutate_valid_task_list += valid_task

        save_jsonl(valid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/mutate/iteration={args.iteration}/{args.task_type}/valid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))

        if invalid_task == []:
            break
        else:
            save_jsonl(invalid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/mutate/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))
        
        invalid_skill_set = set([task["skill"] for task in invalid_task])
        valid_skill_set = set([task["skill"] for task in valid_task])
        unsuccessful_skill_set = invalid_skill_set - valid_skill_set
        logger.info(f"Unsuccessful mutate skill tasks: {unsuccessful_skill_set} | size: {len(unsuccessful_skill_set)}")
        if len(unsuccessful_skill_set) == 0:
            break

        tasks = [task for task in initial_tasks if task["skill"] in unsuccessful_skill_set]
        trial += 1

    valid_tasks, turned_invalid_tasks = assign_invalid_skill_task(invalid_task, mutate_valid_task_list)
    
    save_jsonl(valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/mutate/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}.jsonl"))
    
    return valid_tasks


def get_crossover_tasks(
    task_proposer,
    task_verifier,
    initial_tasks,
    mutate_tasks,
    config,
    args,
    logger,
    lower_bound=0.0,
    upper_bound=1.0,
):  
    all_tasks = initial_tasks + mutate_tasks
    all_valid_tasks = []
    solver_name = config["task_verifier"]["solver"][args.solver].split("/")[-1].lower()
    all_skills = task_proposer.skill_list.copy()
    for iter in range(3):
        crossover_valid_task_list = []
        trial = 0
        task_proposer.skill_list = all_skills
        while trial <= 8:
            crossover_tasks = task_proposer.propose_task(
                propose_type="crossover",
                task_type=args.task_type,
                task_in=all_tasks,
                batch_size=args.proposer_batch_size,
                max_tokens=config["task_proposer"]["proposer"][args.proposer]["max_tokens"],
                num_inputs=config["task_proposer"]["num_inputs"],
                banned_keywords=config["task_proposer"]["banned_keywords"],
                remove_after_return=config["task_proposer"]["remove_after_return"],
                remove_comments=config["task_proposer"]["remove_comments"],
                remove_print=config["task_proposer"]["remove_print"],
            )

            valid_task, invalid_task = task_verifier.verify_task(
                tasks=crossover_tasks,
                task_type=args.task_type,
                proposer_batch_size=args.proposer_batch_size,
                solver_batch_size=args.solver_batch_size,
                max_tokens=config["task_verifier"]["proposer"][args.proposer]["max_tokens"],
                banned_keywords=config["task_proposer"]["banned_keywords"],
                check_determinism=config["task_verifier"]["check_determinism"],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            crossover_valid_task_list += valid_task

            save_jsonl(valid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/crossover/iteration={args.iteration}/{args.task_type}/valid_{args.proposer}_{args.dataset}_iter{trial}_explore{iter}.jsonl"))

            if invalid_task == []:
                break
            else:
                save_jsonl(invalid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/crossover/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}_explore{iter}.jsonl"))
            
            invalid_skill_set = set([task["skill"] for task in invalid_task])
            valid_skill_set = set([task["skill"] for task in valid_task])
            unsuccessful_skill_set = invalid_skill_set - valid_skill_set
            logger.info(f"Unsuccessful crossover skill tasks: {unsuccessful_skill_set} | size: {len(unsuccessful_skill_set)}")
            if len(unsuccessful_skill_set) == 0:
                break
            
            # Reset skills for re-crossover
            task_proposer.skill_list = [skill for skill in task_proposer.skill_list if skill["skill"] in unsuccessful_skill_set]
            all_tasks += crossover_valid_task_list
            trial += 1

        valid_tasks, turned_invalid_tasks = assign_invalid_skill_task(invalid_task, crossover_valid_task_list)
        all_valid_tasks += valid_tasks
        save_jsonl(valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/crossover/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}_explore{iter}.jsonl"))
    save_jsonl(all_valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/crossover/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}.jsonl"))
    return all_valid_tasks


def get_induction_tasks(
    task_proposer,
    task_verifier,
    tasks,
    config,
    args,
    logger
):
    logger.info(f"Total tasks: {len(tasks)}")
    solver_name = config["task_verifier"]["solver"][args.solver].split("/")[-1].lower()

    # Propose induction tasks
    valid_tasks = []
    trial = 0
    reject_info = None
    while trial <= 3:
        init_tasks = tasks
        induction_tasks = task_proposer.propose_task(
            propose_type="init",
            task_type=args.task_type,
            task_in=init_tasks,
            batch_size=args.proposer_batch_size,
            max_tokens=config["task_proposer"]["proposer"][args.proposer]["max_tokens"],
            num_inputs=config["task_proposer"]["num_inputs"],
            banned_keywords=config["task_proposer"]["banned_keywords"],
            remove_after_return=config["task_proposer"]["remove_after_return"],
            remove_comments=config["task_proposer"]["remove_comments"],
            remove_print=config["task_proposer"]["remove_print"],
            reject_info=reject_info,
        )

        save_jsonl(induction_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/prevalid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))

        valid_task, invalid_task = task_verifier.verify_task(
            tasks=induction_tasks,
            task_type=args.task_type,
            proposer_batch_size=args.proposer_batch_size,
            solver_batch_size=args.solver_batch_size,
            max_tokens=config["task_verifier"]["proposer"][args.proposer]["max_tokens"],
            banned_keywords=config["task_proposer"]["banned_keywords"],
            check_determinism=config["task_verifier"]["check_determinism"],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        save_jsonl(valid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/valid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))
        valid_tasks += valid_task
        
        if invalid_task == []:
            break
        else:
            save_jsonl(invalid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))

        # Propose Hints
        reject_info = [
            (
                {"reason": task.get("reason"), "feedback": task.get("feedback"), "code": task.get("code")}
            )
            for task in invalid_task
        ]

        save_jsonl(valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/valid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))
        tasks = [task for task in tasks if task["code"] not in [valid_t["code"] for valid_t in valid_tasks]]
        logger.info(f"Trial {trial} | Valid tasks: {len(valid_tasks)} | Invalid tasks: {len(tasks)}")
        trial += 1

    save_jsonl(valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}.jsonl"))
    return valid_tasks

def get_induction_tasks_hint(
    task_proposer,
    task_verifier,
    tasks,
    config,
    args,
    logger,
    lower_bound=0.0,
    upper_bound=1.0,
):
    logger.info(f"Total tasks: {len(tasks)}")
    solver_name = config["task_verifier"]["solver"][args.solver].split("/")[-1].lower()

    # Propose induction tasks
    valid_tasks = []
    invalid_tasks = []
    trial = 0
    while trial <= 3:
        difficult_tasks = []
        task_trial = 0
        init_tasks = tasks
        reject_info = None
        while task_trial <= 3:
            induction_tasks = task_proposer.propose_task(
                propose_type="init",
                task_type=args.task_type,
                task_in=init_tasks,
                batch_size=args.proposer_batch_size,
                max_tokens=config["task_proposer"]["proposer"][args.proposer]["max_tokens"],
                num_inputs=config["task_proposer"]["num_inputs"],
                banned_keywords=config["task_proposer"]["banned_keywords"],
                remove_after_return=config["task_proposer"]["remove_after_return"],
                remove_comments=config["task_proposer"]["remove_comments"],
                remove_print=config["task_proposer"]["remove_print"],
                reject_info=reject_info,
            )

            save_jsonl(induction_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/prevalid_{args.proposer}_{args.dataset}_iter{trial}_task{task_trial}.jsonl"))

            valid_task, invalid_task = task_verifier.verify_task(
                tasks=induction_tasks,
                task_type=args.task_type,
                proposer_batch_size=args.proposer_batch_size,
                solver_batch_size=args.solver_batch_size,
                max_tokens=config["task_verifier"]["proposer"][args.proposer]["max_tokens"],
                banned_keywords=config["task_proposer"]["banned_keywords"],
                check_determinism=config["task_verifier"]["check_determinism"],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            save_jsonl(valid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/valid_{args.proposer}_{args.dataset}_iter{trial}_task{task_trial}.jsonl"))
            valid_tasks += valid_task
            
            if invalid_task == []:
                break
            else:
                invalid_tasks += [task for task in invalid_task if task is not None and "problem" in task and "inputs" in task and "outputs" in task]
                save_jsonl(invalid_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}_task{task_trial}.jsonl"))

            difficult_tasks += [task for task in invalid_task if "difficulty" in task and task["difficulty"] <= lower_bound]
            logger.info(f"Trial {trial} | Task {task_trial} | Too difficult tasks: {len(difficult_tasks)}")
            init_tasks = [task for task in invalid_task if task not in difficult_tasks]
            logger.info(f"Trial {trial} | Task {task_trial} | Invalid tasks: {len(init_tasks)}")
            task_trial += 1

        save_jsonl(difficult_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/difficult_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))

        # Propose Hints
        reject_info = [
            (
                {"reason": task.get("reason"), "feedback": task.get("feedback"), "code": task.get("code")}
            )
            for task in difficult_tasks
        ]
        hint_tasks = task_proposer.propose_task(
            propose_type="init",
            task_type=args.task_type,
            task_in=difficult_tasks,
            batch_size=args.proposer_batch_size,
            max_tokens=config["task_proposer"]["proposer"][args.proposer]["max_tokens"],
            num_inputs=config["task_proposer"]["num_inputs"],
            banned_keywords=config["task_proposer"]["banned_keywords"],
            remove_after_return=config["task_proposer"]["remove_after_return"],
            remove_comments=config["task_proposer"]["remove_comments"],
            remove_print=config["task_proposer"]["remove_print"],
            reject_info=reject_info,
        )
        save_jsonl(hint_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}_iter{trial}_hint.jsonl"))

        
        hint_tasks = [task for task in hint_tasks if task.get("hints", None) is not None]
        # Verify different level of hints
        # hint_trial = 4
        while hint_trial < 4:
            current_hint_tasks = add_hint_to_task(hint_tasks, hint_trial)
            
            valid_hint_task, invalid_hint_task = task_verifier.verify_task(
                tasks=current_hint_tasks,
                task_type=args.task_type,
                proposer_batch_size=args.proposer_batch_size,
                solver_batch_size=args.solver_batch_size,
                max_tokens=config["task_verifier"]["proposer"][args.proposer]["max_tokens"],
                banned_keywords=config["task_proposer"]["banned_keywords"],
                check_determinism=config["task_verifier"]["check_determinism"],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            
            valid_tasks += valid_hint_task
            logger.info(f"Trial {trial} | Hint {hint_trial+1} | Valid tasks: {len(valid_hint_task)} | Invalid tasks: {len(invalid_hint_task)}")
            save_jsonl(invalid_hint_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}_hint{hint_trial}.jsonl"))
            # save_jsonl(invalid_hint_task, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}_hint.jsonl"))
            invalid_tasks += [task for task in invalid_hint_task if task is not None and "problem" in task and "inputs" in task and "outputs" in task and "Hint" in task["problem"]]
            hint_trial += 1

        save_jsonl(valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/valid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))
 
        tasks = [task for task in tasks if task["code"] not in [valid_task["code"] for valid_task in valid_tasks]]
        save_jsonl(invalid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/invalid_{args.proposer}_{args.dataset}_iter{trial}.jsonl"))

        logger.info(f"Trial {trial} | Valid tasks: {len(valid_tasks)} | Invalid tasks: {len(tasks)}")
        trial += 1

    save_jsonl(valid_tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/init/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}.jsonl"))
    return valid_tasks


def main():
    logger = setup_logger()
    logger.info("="*50)
    logger.info(f"Started at {datetime.datetime.now()}")
    logger.info(f"Using GPU: {torch.cuda.device_count()}")

    args = argparser()
    config = load_config("src/proposer_config.yml")

    dataset = load_dataset(args.dataset)
    dataset = dataset[:10]
    logger.info(f"Loaded {len(dataset)} problems")

    if not os.path.exists(os.path.join(config["skill_synthesizer"]["cluster_out_file_path"], f"skill_{args.proposer}_{args.dataset}.jsonl")) or \
        not os.path.exists(os.path.join(config["skill_synthesizer"]["cluster_out_file_path"], f"attribute_{args.proposer}_{args.dataset}.jsonl")):
        # Get skill & attribute
        skills, cluster_attributes = get_skill(args, config, dataset)
    else:
        # ==== load skill & attribute ====
        with open(os.path.join(config["skill_synthesizer"]["cluster_out_file_path"], f"skill_{args.proposer}_{args.dataset}.jsonl"), "r") as f:
            skills = [json.loads(line) for line in f]
        with open(os.path.join(config["skill_synthesizer"]["cluster_out_file_path"], f"attribute_{args.proposer}_{args.dataset}.jsonl"), "r") as f:
            cluster_attributes = [json.loads(line) for line in f]

    cluster_skills = format_cluster_skills(skills)
    skill_candidates = [skill["skill"] for skill in cluster_skills]

    solver_name = config["task_verifier"]["solver"][args.solver].split("/")[-1].lower()
    BASE = 'base' in solver_name
    logger.info(f"Solver name: {solver_name}")
    logger.info(f"BASE: {BASE}")

    # Load artifacts
    solver = LLM(
        model = config["task_verifier"]["solver"][args.solver],
        dtype = "bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        data_parallel_size=1,
        gpu_memory_utilization=0.95,
        seed=config["task_verifier"]["solver"]["seed"],
        enforce_eager=False,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["task_verifier"]["solver"][args.solver], trust_remote_code=True)
    tokenizer.padding_side = "left"

    stop_words = config["task_verifier"]["solver"].get("stop_words", None)

    sampling_params = SamplingParams(
        n=config["task_verifier"]["rollout"],
        max_tokens=config["task_verifier"]["solver"]["max_tokens"],
        temperature=config["task_verifier"]["solver"]["temperature"],
        top_p=config["task_verifier"]["solver"]["top_p"],
        stop=stop_words,
    )

    task_verifier = TaskVerifier(
        proposer=args.proposer,
        solver=solver,
        skill_candidates=skill_candidates,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        BASE=BASE,
    )

    task_proposer = TaskProposer(
        model=args.proposer,
        skill_list=cluster_skills,
        attribute_list=cluster_attributes,
    )

    if args.task_type in ["code_in", "code_out"]:
        # Propose initial tasks based on single skill  
        init_tasks = get_init_tasks(
            task_proposer=task_proposer,
            task_verifier=task_verifier,
            cluster_skills=cluster_skills,
            config=config,
            args=args,
            logger=logger,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
        )

        # Mutate initial tasks by complexity attributes
        mutate_tasks = get_mutate_tasks(
            task_proposer=task_proposer,
            task_verifier=task_verifier,
            initial_tasks=init_tasks,
            cluster_attributes=cluster_attributes,
            config=config,
            args=args,
            logger=logger,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
        )

        all_init_tasks = init_tasks
        if args.iteration > 0:
            for i in range(args.iteration):
                with open(os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/{args.proposer}_{args.dataset}_all_iter={i}.jsonl"), "r") as f:
                    iter_all_tasks = [json.loads(line) for line in f]
                iter_tasks = [task for task in iter_all_tasks if task["task_type"] == args.task_type]
                logger.info(f"Iteration {i} | Total {args.task_type} tasks: {len(iter_tasks)}")

                all_init_tasks += iter_tasks
        
        logger.info(f"Iteration {args.iteration} | Total {args.task_type} tasks: {len(all_init_tasks+mutate_tasks)}")
        # Crossover to find diverse skill combination tasks
        crossover_tasks = get_crossover_tasks(
            task_proposer=task_proposer,
            task_verifier=task_verifier,
            initial_tasks=all_init_tasks,
            mutate_tasks=mutate_tasks,
            config=config,
            args=args,
            logger=logger,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
        )

        tasks = []
        for propose_type in ["init", "mutate", "crossover"]:
            try:
                with open(os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/{propose_type}/iteration={args.iteration}/{args.task_type}/{args.proposer}_{args.dataset}.jsonl"), "r") as f:
                    tasks += [json.loads(line) for line in f]
            except FileNotFoundError:
                continue
        save_jsonl(tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/{args.proposer}_{args.dataset}_{args.task_type}.jsonl"))

    elif args.task_type == "code_func":
        with open(os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/{args.proposer}_{args.dataset}_code_out.jsonl"), "r") as f:
            out_tasks = [json.loads(line) for line in f]
        with open(os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/{args.proposer}_{args.dataset}_code_in.jsonl"), "r") as f:
            in_tasks = [json.loads(line) for line in f]
        tasks = out_tasks + in_tasks


        induction_tasks = get_induction_tasks_hint(
            task_proposer=task_proposer,
            task_verifier=task_verifier,
            tasks=tasks,
            config=config,
            args=args,
            logger=logger,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
        )

        tasks += induction_tasks
        save_jsonl(tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/{args.proposer}_{args.dataset}_all_iter={args.iteration}_v0.jsonl"))
        tasks, removed = clean_tasks(tasks)
        if removed > 0:
            logger.info(f"Removed {removed} improper tasks")
            save_jsonl(tasks, os.path.join(config["task_proposer"]["task_out_file_path"], f"{solver_name}/{args.proposer}_{args.dataset}_all_iter={args.iteration}.jsonl"))


if __name__ == "__main__":
    main()