import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
from tqdm import tqdm
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

from la_datasets import MMLUMCDataset 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


# ============================================================
# Args
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--num_sample", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--sample_seed", type=int, default=42)

    return parser.parse_args()


# ============================================================
# Load Model
# ============================================================

def load_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    model.cuda().eval()

    return model, tokenizer


# ============================================================
# Generation + Evaluation
# ============================================================

def normalize_mc_prediction(text):
    """
    从生成结果中抽取 A/B/C/D
    """
    text = text.strip().upper()

    for choice in ["A", "B", "C", "D"]:
        if text.startswith(choice):
            return choice

    return text[:1]


@torch.inference_mode()
def generate_and_evaluate(
    model,
    tokenizer,
    input_ids,
    answer,
    generation_kwargs,
    num_samples
):
    samples = []
    num_correct = 0

    for _ in range(num_samples):

        outputs = model.generate(
            input_ids=input_ids.cuda(),
            **generation_kwargs
        )

        pred = tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        pred = normalize_mc_prediction(pred)

        correct = int(pred == answer)

        samples.append({
            "pred": pred,
            "correct": correct
        })

        num_correct += correct

    return {
        "answer": answer,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "sample_accuracy": num_correct / num_samples,
        "samples": samples
    }


# ============================================================
# Loss (Teacher Forcing)
# ============================================================

def calculate_loss(model, input_ids, answer, tokenizer):

    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]

    input_ids = input_ids.cuda()
    labels = input_ids.clone()

    # 只训练最后一个 token
    labels[:, :-1] = -100

    outputs = model(input_ids=input_ids, labels=labels)

    return outputs.loss.item()


# ============================================================
# Main Loop
# ============================================================

@torch.inference_mode()
def run_mmlu(args):

    model_name = os.path.basename(args.model_path)
    model_family = model_name.split('-')[0]  
    base_name = ''.join([c for c in model_family if c.isalpha()])

    model, tokenizer = load_model(args.model_path)

    dataset = MMLUMCDataset(
        parquet_path=args.parquet_path,
        tokenizer=tokenizer,
        model_name=model_name,
        task_type="mc_qa",
        sample_ratio=args.sample_ratio
    )
    
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)

    os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(
            args.save_dir,
            f"results_mmlu_{base_name}.json"
        )

        
    line_break_id = tokenizer.encode("<end_of_turn>", add_special_tokens=False)[-1] if 'gemma' in model_name else tokenizer.eos_token_id

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": line_break_id,
        "eos_token_id": line_break_id,
        "use_cache": True,
    }

    all_results = []
    total_correct = 0
    total = 0

    for batch in tqdm(dataloader):
        # print(batch)

        input_ids = batch["input_ids"]
        answer = batch["label"]

        gen_results = generate_and_evaluate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            answer=answer,
            generation_kwargs=generation_kwargs,
            num_samples=args.num_sample
        )
        # print(gen_results)

        loss = calculate_loss(
            model=model,
            input_ids=input_ids,
            answer=answer,
            tokenizer=tokenizer
        )

        gen_results.update({
            "loss": loss,
            "question": batch["question"],
            "subject": batch["subject"],
            "choices": list(batch["choices"])
        })

        total_correct += gen_results["num_correct"]
        total += args.num_sample

        all_results.append(gen_results)

    logger.info(f"Overall Sample Accuracy: {total_correct / total:.4f}")

    json.dump(all_results, open(save_path, "w"), indent=4)


# ============================================================
# Entry
# ============================================================

def main():
    args = get_args()
    logger.info(json.dumps(vars(args), indent=4))
    run_mmlu(args)


if __name__ == "__main__":
    main()