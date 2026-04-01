import json
import os
import datasets
import random
from utils import PROJ_DIR
import torch
import numpy as np
from tqdm import tqdm
from eval_utils import sub_ans_exact_match_score_with_macnoise_subcontext as macnoise_sub_em
from eval_utils import exact_match_score_with_multiple_candidates as em
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_grouped_hiddens(model_name, hiddens_name, layer_idx):
    load_dir = PROJ_DIR / "cache_data" / model_name / hiddens_name
    label0_hiddens = torch.load(load_dir / f"layer{layer_idx}-use-parameter.pt").cuda()
    label1_hiddens = torch.load(load_dir / f"layer{layer_idx}-use-context.pt").cuda()
    load_files = json.load(open(load_dir / "load_files.json", "r"))
    logger.info("files for hidden states")
    logger.info(json.dumps(load_files, indent=4))
    return {"label0_hiddens": label0_hiddens, "label1_hiddens": label1_hiddens, "load_files": load_files}


def sample_train_data(label0_hiddens=None, label1_hiddens=None, pred_sub_answer_data=None,
                      pred_org_answer_data=None, num_train_examples=None, seed=42):
    if num_train_examples is not None:
        rng = np.random.RandomState(seed)
        cands = list(range(min(len(label0_hiddens), len(label1_hiddens))))
        rng.shuffle(cands)
        selected = torch.tensor(cands[:num_train_examples])
        if pred_sub_answer_data is not None:
            pred_sub_answer_data = [pred_sub_answer_data[idx] for idx in selected]
            pred_org_answer_data = [pred_org_answer_data[idx] for idx in selected]
        return {"label0_hiddens": label0_hiddens[selected], "label1_hiddens": label1_hiddens[selected],
                "pred_sub_answer_data": pred_sub_answer_data, "pred_org_answer_data": pred_org_answer_data}
    else:
        return {"label0_hiddens": label0_hiddens, "label1_hiddens": label1_hiddens}



def unified_em(pred_answer, org_answer):
    org_answer_em = em(pred_answer, org_answer)  # batch_size= 1

    return org_answer_em


def calculate_detailed_em(
        initial_sub_scores,
        initial_org_scores,
        use_context_sub_scores,
        use_parameter_org_scores,
        use_context_org_scores,
        use_parameter_sub_scores,
):
    results = dict()
    if len(use_context_sub_scores) > 0:
        steer_use_c_from_c_to_c = []
        for i_sub, c_sub in zip(initial_sub_scores, use_context_sub_scores):
            if i_sub == 1:
                steer_use_c_from_c_to_c.append(c_sub)
        steer_use_c_from_m_to_c = []
        for i_org, c_sub in zip(initial_org_scores, use_context_sub_scores):
            if i_org == 1:
                steer_use_c_from_m_to_c.append(c_sub)
        steer_use_c_from_c_to_m = []
        for i_sub, c_org in zip(initial_sub_scores, use_context_org_scores):
            if i_sub == 1:
                steer_use_c_from_c_to_m.append(c_org)
        steer_use_c_from_m_to_m = []
        for i_org, c_org in zip(initial_org_scores, use_context_org_scores):
            if i_org == 1:
                steer_use_c_from_m_to_m.append(c_org)
        steer_use_c_from_c_to_c = sum(steer_use_c_from_c_to_c) / len(steer_use_c_from_c_to_c) * 100
        steer_use_c_from_m_to_c = sum(steer_use_c_from_m_to_c) / len(steer_use_c_from_m_to_c) * 100
        steer_use_c_from_c_to_m = sum(steer_use_c_from_c_to_m) / len(steer_use_c_from_c_to_m) * 100
        steer_use_c_from_m_to_m = sum(steer_use_c_from_m_to_m) / len(steer_use_c_from_m_to_m) * 100
        steer_use_c_overall_c = sum(use_context_sub_scores) / len(use_context_sub_scores) * 100
        steer_use_c_overall_m = sum(use_context_org_scores) / len(use_context_org_scores) * 100
        results.update({"SteerUseContext/overall_C": steer_use_c_overall_c,
                        "SteerUseContext/overall_M": steer_use_c_overall_m,
                        "SteerUseContext/from_C_to_C": steer_use_c_from_c_to_c,
                        "SteerUseContext/from_M_to_C": steer_use_c_from_m_to_c,
                        "SteerUseContext/from_C_to_M": steer_use_c_from_c_to_m,
                        "SteerUseContext/from_M_to_M": steer_use_c_from_m_to_m, })
    if len(use_parameter_org_scores) > 0:
        steer_use_m_from_m_to_m = []
        for i_org, m_org in zip(initial_org_scores, use_parameter_org_scores):
            if i_org == 1:
                steer_use_m_from_m_to_m.append(m_org)
        steer_use_m_from_c_to_m = []
        for i_sub, m_org in zip(initial_sub_scores, use_parameter_org_scores):
            if i_sub == 1:
                steer_use_m_from_c_to_m.append(m_org)
        steer_use_m_from_m_to_c = []
        for i_org, m_sub in zip(initial_org_scores, use_parameter_sub_scores):
            if i_org == 1:
                steer_use_m_from_m_to_c.append(m_sub)
        steer_use_m_from_c_to_c = []
        for i_sub, m_sub in zip(initial_sub_scores, use_parameter_sub_scores):
            if i_sub == 1:
                steer_use_m_from_c_to_c.append(m_sub)
        steer_use_m_from_m_to_m = sum(steer_use_m_from_m_to_m) / len(steer_use_m_from_m_to_m) * 100
        steer_use_m_from_c_to_m = sum(steer_use_m_from_c_to_m) / len(steer_use_m_from_c_to_m) * 100
        steer_use_m_from_m_to_c = sum(steer_use_m_from_m_to_c) / len(steer_use_m_from_m_to_c) * 100
        steer_use_m_from_c_to_c = sum(steer_use_m_from_c_to_c) / len(steer_use_m_from_c_to_c) * 100
        steer_use_m_overall_c = sum(use_parameter_sub_scores) / len(use_parameter_sub_scores) * 100
        steer_use_m_overall_m = sum(use_parameter_org_scores) / len(use_parameter_org_scores) * 100

        results.update({"SteerUseParameter/overall_C": steer_use_m_overall_c,
                        "SteerUseParameter/overall_M": steer_use_m_overall_m,
                        "SteerUseParameter/from_M_to_M": steer_use_m_from_m_to_m,
                        "SteerUseParameter/from_C_to_M": steer_use_m_from_c_to_m,
                        "SteerUseParameter/from_M_to_C": steer_use_m_from_m_to_c,
                        "SteerUseParameter/from_C_to_C": steer_use_m_from_c_to_c})
    return results


def load_grouped_prompts(model_name, results_save_dir_name="grouped_prompts",
                         shots=None, seeds=None, files=None):
    load_dir = PROJ_DIR / "cache_data" / model_name / results_save_dir_name
    all_results = []
    load_files = []
    if shots is not None:
        assert seeds is not None
        for shot in shots:
            for seed in seeds:
                cur_path = load_dir / f"{shot}shot-seed{seed}-results.json"
                logger.info(f"load file: {cur_path}")
                load_files.append(cur_path)
                cur_results = json.load(open(cur_path, "r"))
                all_results.extend(cur_results)
        logger.info(f"do not check the duplication")
    else:
        for cur_path in tqdm(files):
            logger.info(f"load file: {cur_path}")
            existed_data = json.load(open(cur_path, "r"))
            all_results.extend(existed_data)
            load_files.append(cur_path)

        logger.info(f"do not check the duplication")

    pred_sub_answer_data, pred_org_answer_data = [], []
    for item in all_results:
        if item["sub_answer_em"] == 1:
            pred_sub_answer_data.append(item)
        if item["org_answer_em"] == 1:
            pred_org_answer_data.append(item)
        if item["sub_answer_em"] == 1 and item["org_answer_em"] == 1:
            raise ValueError("sub_answer == org_answer")
    logger.info(f"loaded {len(pred_sub_answer_data)} use-context-data, "
                f"{len(pred_org_answer_data)} use-parameter-data")
    for idx in range(len(pred_sub_answer_data)):
        pred_sub_answer_data[idx]["use_context_idx"] = idx
    for idx in range(len(pred_org_answer_data)):
        pred_org_answer_data[idx]["use_parameter_idx"] = idx
    return pred_sub_answer_data, pred_org_answer_data, load_files


def load_dataset(data_name):
    if data_name == "nq":
        data = datasets.load_dataset('json', data_files="datasets/nq/test_top10.json")
        data = [_ for _ in data["train"]]
        idx2data = dict()
        for idx, item in enumerate(data):
            item["idx"] = idx
            idx2data[idx] = item
        # cache_path = PROJ_DIR / "cached_data" / f"{data_name}-{model_name}-memorised_set"
        # memorised_set = torch.load(cache_path)
        shots = random.sample(range(len(data)), int(len(data)*0.2))
        return data, shots
    elif data_name == "tqa":
        data = datasets.load_dataset('json', data_files="datasets/tqa/test_top10.json")
        data = [_ for _ in data["train"]]
        idx2data = dict()
        for idx, item in enumerate(data):
            item["idx"] = idx
            idx2data[idx] = item
        # cache_path = PROJ_DIR / "cached_data" / f"{data_name}-{model_name}-memorised_set"
        # memorised_set = torch.load(cache_path)
        shots = random.sample(range(len(data)), int(len(data)*0.2))
        return data, shots
    elif data_name == "hqa":
        data = datasets.load_dataset('json', data_files="datasets/hotpotqa/test_top10.json")
        data = [_ for _ in data["train"]]
        idx2data = dict()
        for idx, item in enumerate(data):
            item["idx"] = idx
            idx2data[idx] = item
        # cache_path = PROJ_DIR / "cached_data" / f"{data_name}-{model_name}-memorised_set"
        # memorised_set = torch.load(cache_path)
        shots = random.sample(range(len(data)), int(len(data)*0.2))
        return data, shots

    
