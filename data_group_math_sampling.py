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
from datasets import load_dataset

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
# MATH utils
# =========================

INVALID_ANS = "[invalid]"

def extract_math_answer(text: str):
    """
    Robust MATH answer extraction.
    Finds the last \boxed{...} and correctly handles nested braces.
    """
    if text is None:
        return INVALID_ANS

    text = text.strip()
    
    # 找到最后一个 \boxed{ 的位置
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Fallback: 如果没有 \boxed，尝试找最后一个 $...$ 
        matches = re.findall(r'\$([^\$]+)\$', text)
        if matches:
            return matches[-1].strip()
        return INVALID_ANS

    # 处理嵌套大括号
    start = idx + len("\\boxed{")
    brace_count = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        
        if brace_count == 0:
            return text[start:i].strip()
            
    return INVALID_ANS

def normalize_math_answer(ans: str):
    """
    基础的数学字符串标准化，用于计算 Exact Match (EM)。
    移除空格、逗号等，使匹配更鲁棒。
    """
    if ans == INVALID_ANS:
        return ans
    # 移除所有空白字符和 LaTeX 逗号/空格
    ans = re.sub(r'\s+', '', ans)
    ans = ans.replace("\\,", "").replace(",", "")
    return ans

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
# MATH Dataset
# =========================

class MathInferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_name, split="test", sample_ratio=1.0):
        self.tokenizer = tokenizer
        template = MODEL_TEMPLATE_DICT[model_name]
        self.prefix = template["prefix"]
        self.end = template["end"]

        # 使用 Hugging Face datasets 加载
        logger.info(f"Loading dataset from {data_path}, split: {split}")
        hf_dataset = load_dataset(data_path, split=split)

        # 数据集采样逻辑
        if 0 < sample_ratio < 1.0:
            sample_size = max(1, int(len(hf_dataset) * sample_ratio))
            logger.info(f"Sampling {sample_ratio*100}% of the dataset: {sample_size} examples.")
            # 使用固定 seed 保证每次抽样的一致性
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(sample_size))
        
        self.data = []
        for idx, ex in enumerate(hf_dataset):
            self.data.append({
                "idx": idx,
                "question": ex["problem"].strip(),
                "answer": ex["solution"].strip(),
                "level": ex.get("level", ""),
                "type": ex.get("type", "")
            })

        logger.info(f"Loaded {len(self.data)} MATH {split} examples")

    def build_prompt(self, question):
        return (
            self.prefix
            + "Answer the question by briefly explaining your reasoning, then provide the final answer enclosed in \\boxed{}.\n"
              f"Question: {question}\n"
              "Answer:\n"
            + self.end
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = self.build_prompt(ex["question"])
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids[0]
        attention_mask = encoded.attention_mask[0]
        return {
            "idx": ex["idx"],
            "question": ex["question"],
            "answer": ex["answer"],
            "level": ex["level"],
            "type": ex["type"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

# =========================
# Model
# =========================

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
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
def generate_and_evaluate_with_sampling_math(
    model,
    tokenizer,
    batch,
    gt_answer,
    generation_kwargs,
    num_samples=20
):
 
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    samples = []
    num_correct = 0

    # 提取 Ground Truth 中的最终答案
    gt_final = extract_math_answer(gt_answer)
    gt_norm = normalize_math_answer(gt_final)

    for _ in range(num_samples):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

        gen_text = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # 提取预测结果中的最终答案
        pred_final = extract_math_answer(gen_text)
        pred_norm = normalize_math_answer(pred_final)

        # Exact Match (基于字符串匹配)
        em = int(
            pred_norm != INVALID_ANS
            and gt_norm != INVALID_ANS
            and pred_norm == gt_norm
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
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local model")
    # 默认使用你提供的 Hugging Face 数据集路径
    parser.add_argument("--math_path", type=str, default="llm-aware-73/datasets/math")
    parser.add_argument("--save_dir_name", type=str, default="./cashed_data_math")
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--seeds_to_encode", type=int, nargs="+", default=[42])
    parser.add_argument("--num_sample", type=int, default=5)
    parser.add_argument("--sample_ratio", type=float, default=0.5, help="Fraction of the dataset to evaluate (e.g., 0.01 for 1%)")
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.model_path))
    model_family = model_name.split('-')[0]  
    base_name = ''.join([c for c in model_family if not c.isdigit()])
    
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model(args.model_path)

    dataset = MathInferenceDataset(
        data_path=args.math_path,
        tokenizer=tokenizer,
        model_name=model_name,
        split=args.split,
        sample_ratio=args.sample_ratio
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    line_break_id = tokenizer.encode("<end_of_turn>", add_special_tokens=False)[-1] if 'gemma' in model_name else tokenizer.eos_token_id

    generation_kwargs = {
        "max_new_tokens": 512, # MATH 通常需要更长的推理过程
        "do_sample": True,
        "temperature": 0.7,      
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
            gen_res = generate_and_evaluate_with_sampling_math(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                gt_answer=batch["answer"][0],
                generation_kwargs=generation_kwargs,
                num_samples=args.num_sample
            )

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
                "level": batch["level"][0],
                "type": batch["type"][0],
                **gen_res,
                "loss": loss,
            }

            if gen_res["sample_accuracy"] == 1:
                em_total += 1

            results.append(record)

        logger.info(f"[seed={seed}] Avg Sample Accuracy = {em_total / len(results):.4f}")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()