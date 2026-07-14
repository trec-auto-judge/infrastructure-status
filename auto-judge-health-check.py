#!/usr/bin/env python3
import click
import requests
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from functools import partial
from tqdm import tqdm
from pathlib import Path
import json
import yaml
import tempfile
import zipfile
import shutil
from os import environ
from subprocess import check_output
from tira.io_utils import parse_prototext_key_values


def track_execution(func, retries=3, timeout=300):
    last_exception = None

    for attempt in range(1, retries + 1):
        start_time = time.perf_counter()

        try:
            # Use ThreadPoolExecutor to enforce the timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func)
                result = future.result(timeout=timeout)
                result["time"] = time.perf_counter() - start_time
                return result

        except TimeoutError:
            print(f"Attempt {attempt} failed: Execution timed out after {timeout}s")
            last_exception = Exception(f"Function timed out after {timeout} seconds")
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            last_exception = e
        finally:
            pass


def download_run(team, run_id, dataset_id, result_dir):
    url = f"https://www.tira.io/task/trec-auto-judge/user/{team}/dataset/{dataset_id}/download/{run_id}.zip"

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".zip") as archive:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    archive.write(chunk)
            archive.flush()

            with tempfile.TemporaryDirectory() as extraction_dir:
                with zipfile.ZipFile(archive.name) as zip_file:
                    zip_file.extractall(extraction_dir)

                run_dir = Path(extraction_dir) / run_id
                if not run_dir.is_dir():
                    raise ValueError(
                        f"Downloaded archive does not contain directory '{run_id}'"
                    )

                size = sum(
                    path.stat().st_size
                    for path in run_dir.rglob("*")
                    if path.is_file()
                )
                shutil.move(run_dir, result_dir / run_id)

    return {
        "team": team,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "size": size,
    }

ALL_TESTS = {}

def load_test_matrix():
    with open(Path(__file__).parent / "test-matrix.yml") as f:
        matrix = yaml.safe_load(f)
        return matrix


def populate_download_run_tests(out_dir):
    for judge_name, judge in load_test_matrix()["judges"].items():
        for e in judge["executions"]:
            test_name = f"download-prompt-cache-{judge_name}-{e['dataset']}-{e['llm-prompt']['name']}"
            test_execution = partial(
                download_run,
                judge_name.split("/")[0],
                e["llm-prompt"]["run_id"],
                e["dataset"],
                out_dir
            )
            ALL_TESTS[test_name] = test_execution


def download_dataset(dataset):
    check_output(["tira-cli", "download", "--dataset", dataset])
    check_output(["tira-cli", "download", "--dataset", dataset, "--truths"])

    return {"dataset_id": dataset}

def populate_download_datasets_from_tira():
    datasets = set()
    for judge_name, judge in load_test_matrix()["judges"].items():
        for e in judge["executions"]:
            datasets.add(e['dataset'])
    for dataset in datasets:
        test_name = f"download-dataset-{dataset}"
        test_execution = partial(
            download_dataset,
            dataset
        )
        ALL_TESTS[test_name] = test_execution  


def run_auto_judge_test(judge, llm_prompt, dataset, prompt_cache_dir):
    cmd = [
        "tira-cli", "run", "local", "--approach",
        "trec-auto-judge/" + judge, "--input", dataset
    ]
    env_to_populate = {
        "OPENAI_API_KEY": "empty",
        "OPENAI_BASE_URL": "empty",
        "OPENAI_MODEL": llm_prompt["name"],
    }
    
    cmd += ["--forward-environment-variable"] + list(env_to_populate.keys())
    cmd += ["--mount-cache", "CACHE_DIR=" + prompt_cache_dir + "/" + llm_prompt["run_id"] + "/CACHE_DIR"]

    for k, v in env_to_populate.items():
        environ[k] = v

    results = check_output(cmd)
    out_dir = results.decode("UTF-8").split("Full evaluation results: ")[1].split("\n")[0]
    ret = {"judge": judge, "llm_prompt": llm_prompt, "dataset_id": dataset}
    
    eval_file = Path(out_dir) / "evaluation.prototext"
    ret["evaluation"] = {
        measure["key"]: measure["value"]
        for measure in parse_prototext_key_values(eval_file)
    }
    
    return ret


def populate_run_auto_judge_tests(run_dir):
    for judge_name, judge in load_test_matrix()["judges"].items():
        for e in judge["executions"]:
            test_name = f"run-{judge_name}-{e['dataset']}-{e['llm-prompt']['name']}"
            test_execution = partial(
                run_auto_judge_test,
                judge_name,
                e["llm-prompt"],
                e["dataset"],
                run_dir
            )
            ALL_TESTS[test_name] = test_execution

@click.command()
@click.argument("output_file")
def main(output_file):
    current_iso = datetime.now().isoformat()
    ret = []

    populate_download_datasets_from_tira()
    prompt_cache_dir = tempfile.mkdtemp()
    populate_download_run_tests(prompt_cache_dir)
    populate_run_auto_judge_tests(prompt_cache_dir)
    
    for test_name, test in tqdm(ALL_TESTS.items()):
        try:
            result = track_execution(test)
        except:
            result = {"status": "failed"}
        result["name"] = test_name
        result["timestamp"] = current_iso
        ret.append(result)

    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    if not Path(output_file).is_file():
        Path(output_file).touch()

    with open(output_file, "a") as f:
        for l in ret:
            f.write(json.dumps(l) + "\n")

if __name__ == '__main__':
    main()
