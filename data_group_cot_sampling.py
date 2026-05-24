import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

# =========================
# logging
# =========================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================
# GSM8K utils
# =========================

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
RE_HASH = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")

# fallback: 全文连续数字，带逗号
RE_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

def extract_gsm8k_answer_robust(text: str):
    """
    Robust GSM8K answer extraction
    1️⃣ 优先 #### <number>
    2️⃣ fallback: last non-empty line中最长数字串
    3️⃣ fallback: 全文最长数字串
    去掉所有逗号
    """
    if text is None:
        return INVALID_ANS

    text = text.strip()

    # 1️⃣ ####
    m = RE_HASH.search(text)
    if m:
        return m.group(1).replace(",", "")

    # 2️⃣ last line fallback
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        nums = RE_NUMBER.findall(last_line)
        if nums:
            longest_num = max(nums, key=len)
            return longest_num.replace(",", "")

    # 3️⃣ full text fallback
    nums = RE_NUMBER.findall(text)
    if nums:
        longest_num = max(nums, key=len)
        return longest_num.replace(",", "")

    return INVALID_ANS

def extract_gsm8k_answer(text):
    match = ANS_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    return INVALID_ANS


# =========================
# Chat templates
# =========================

MODEL_TEMPLATE_DICT = {
    "llama3.1-8b-instruct": {
        "prefix": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant for solving math word problems."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        ),
        "end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    'qwen2.5-7b-instruct':{
        'prefix': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
        'end': '<|im_end|>\n<|im_start|>assistant\n\n'
    },
    'gemma2-9b-it':{
        'prefix': '<bos><start_of_turn>user\n\n',
        'end': '<end_of_turn><start_of_turn>model\n\n'
    },
}


# =========================
# GSM8K Dataset
# =========================

class GSM8KInferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_name, split="train"):
        self.tokenizer = tokenizer
        template = MODEL_TEMPLATE_DICT[model_name]
        self.prefix = template["prefix"]
        self.end = template["end"]

        file_path = os.path.join(data_path, f"{split}.jsonl")
        assert os.path.exists(file_path), file_path

        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                self.data.append({
                    "idx": idx,
                    "question": ex["question"].strip(),
                    "answer": ex["answer"].strip(),
                })

        logger.info(f"Loaded {len(self.data)} GSM8K {split} examples")

    def build_prompt(self, question):
        return (
            self.prefix
            + "Answer the question by briefly explaining your reasoning with one or few sentences, then provide the final answer in the format: #### <number>\n"
              f"Question: {question}\n"
              "Answer:\n"
            + self.end
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = self.build_prompt(ex["question"])
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return {
            "idx": ex["idx"],
            "question": ex["question"],
            "answer": ex["answer"],
            "input_ids": input_ids,
        }


# =========================
# Model
# =========================

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.cuda().eval()
    return model, tokenizer


# =========================
# Generation + Evaluation
# =========================

@torch.inference_mode()
def generate_and_evaluate_with_sampling_gsm8k(
    model,
    tokenizer,
    input_ids,
    gt_answer,
    generation_kwargs,
    num_samples=20
):
    assert input_ids.dim() == 1

    input_ids = input_ids.unsqueeze(0).cuda()

    samples = []
    num_correct = 0

    gt_final = extract_gsm8k_answer(gt_answer)
    if gt_final != INVALID_ANS:
        gt_final = float(gt_final)

    for _ in range(num_samples):
        outputs = model.generate(
            input_ids=input_ids,
            **generation_kwargs
        )

        gen_text = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        pred_final = extract_gsm8k_answer_robust(gen_text)

        if pred_final != INVALID_ANS:
            pred_final = float(pred_final)

        em = int(
            pred_final != INVALID_ANS
            and gt_final != INVALID_ANS
            and pred_final == gt_final
        )

        samples.append({
            "pred_text": gen_text,
            "pred_final": pred_final,
            "em": em
        })

        num_correct += em

    return {
        "gt_final": gt_final,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "sample_accuracy": num_correct / num_samples,
        "samples": samples
    }


def calculate_loss(model, prompt_ids, gold_answer, tokenizer):
    prompt_len = prompt_ids.shape[0]
    answer_ids = tokenizer(gold_answer, add_special_tokens=False).input_ids

    input_ids = torch.tensor(
        prompt_ids.tolist() + answer_ids,
        device="cuda"
    ).unsqueeze(0)

    labels = torch.tensor(
        [-100] * prompt_len + answer_ids,
        device="cuda"
    ).unsqueeze(0)

    with torch.enable_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    return outputs.loss.item()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/wangyujing/llm-aware/models/gemma2-9b-it")
    parser.add_argument("--gsm8k_path", type=str, default="/home/wangyujing/llm-aware/datasets/gsm8k")
    parser.add_argument("--save_dir_name", type=str, default="/home/wangyujing/llm-aware/cashed_data_gsm8k")
    parser.add_argument("--data_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--seeds_to_encode", type=int, nargs="+", default=[42])
    parser.add_argument("--num_sample", type=int, default=10)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    model_family = model_name.split('-')[0]  # gemma2
    base_name = ''.join([c for c in model_family if not c.isdigit()])
    model, tokenizer = load_model(args.model_path)

    dataset = GSM8KInferenceDataset(
        data_path=args.gsm8k_path,
        tokenizer=tokenizer,
        model_name=model_name,
        split=args.split
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    line_break_id = tokenizer.encode("<end_of_turn>", add_special_tokens=False)[-1] if 'gemma' in model_name else tokenizer.eos_token_id

    generation_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,      # 可以试 0.7 / 1.0
        "top_p": 0.9,
        "eos_token_id": line_break_id,
        "pad_token_id": line_break_id,
        "use_cache": True,
    }

    os.makedirs(args.save_dir_name, exist_ok=True)

    for seed in args.seeds_to_encode:
        torch.manual_seed(seed)

        save_path = os.path.join(
            args.save_dir_name,
            f"{args.k_shot}shot-seed{seed}-results_{args.data_name}_{base_name}.json"
        )

        if os.path.exists(save_path):
            logger.info(f"{save_path} exists, skip")
            continue

        results = []
        em_total = 0

        for batch in tqdm(dataloader, desc=f"seed={seed}"):
            gen_res = generate_and_evaluate_with_sampling_gsm8k(
                model=model,
                tokenizer=tokenizer,
                input_ids=batch["input_ids"][0],
                gt_answer=batch["answer"][0],
                generation_kwargs=generation_kwargs,
                num_samples=args.num_sample
            )
            # print(gen_res)

            loss = calculate_loss(
                model=model,
                prompt_ids=batch["input_ids"][0],
                gold_answer=batch["answer"][0],
                tokenizer=tokenizer,
            )

            record = {
                "idx": int(batch["idx"][0]),
                "question": batch["question"][0],
                "gt_answer": batch["answer"][0],
                **gen_res,
                "loss": loss,
            }

            if gen_res["sample_accuracy"] == 1:
                em_total += 1


            results.append(record)

        logger.info(f"[seed={seed}] Avg Sample Accuracy = {em_total / len(results)}")

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
