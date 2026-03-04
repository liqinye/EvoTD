import json
import gdown
import os
import logging
import shutil
import zipfile

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
):
    os.makedirs("data/dataset", exist_ok=True)
    if dataset_name == "usaco":
        if not os.path.exists("data/dataset/usaco.json"):
            logger.info("Downloading USACO dataset...")
            link = "https://drive.google.com/uc?id=1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi"
            output = "data/dataset/usaco.zip"
            gdown.download(link, output, quiet=False)
            logger.info("Extracting USACO dataset...")
            with zipfile.ZipFile(output, "r") as zip_ref:
                zip_ref.extractall("data/dataset")
            logger.info("Removing extra files...")
            os.remove(output)
            shutil.move("data/dataset/data_copy/datasets/usaco_subset307_dict.json", "data/dataset/usaco.json")
            shutil.rmtree("data/dataset/data_copy")
            logger.info("USACO dataset downloaded and extracted successfully")

        with open("data/dataset/usaco.json", "r") as f:
            dataset = json.load(f)

        new_dataset = []
        for problem_id, problem in dataset.items():
            runtime_limit = problem["runtime_limit_sentences"][0] if problem["runtime_limit_sentences"] else f"\nRuntime limit: {problem['runtime_limit']} seconds"
            memory_limit = problem["memory_limit_sentences"][0] if problem["memory_limit_sentences"] else f"\nMemory limit: {problem['memory_limit']} MB"

            entry = {
                "problem_id": problem_id,
                "problem": problem["description"] + runtime_limit + memory_limit,
                "solution": problem["solution_python3"],
            }
            new_dataset.append(entry)

        return new_dataset

    elif dataset_name == "taco":
        raise NotImplementedError(f"{dataset_name} dataset not supported yet")
    else:
        raise NotImplementedError(f"{dataset_name} dataset not supported yet")
    
