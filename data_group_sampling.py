import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
from tqdm import tqdm
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from la_datasets import REODQADataset
from data_utils import unified_em, load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/wangyujing/llm-aware/models/qwen2.5-7b-instruct")
    parser.add_argument("--save_dir_name", type=str, default="/home/wangyujing/llm-aware/cashed_data_nq")
    parser.add_argument("--seeds_to_encode", type=int, nargs='+', default=[42])
    parser.add_argument("--data_name", type=str, default="nq")
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--num_sample", type=int, default=10)
    return parser.parse_args()

def load_model(model_path, not_return_model=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     padding_side='left',
    #     truncation_side="left",

    tokenizer.pad_token = tokenizer.eos_token
    if not_return_model:
        model = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            attn_implementation = "flash_attention_2", 
            torch_dtype=torch.bfloat16, 
        )
        model.cuda().eval()
    return model, tokenizer

@torch.inference_mode()
def group_prompts_based_on_behaviours(seeds_to_encode, model_path, k_shot, data_name, save_dir_name, base_name, num_sample):
    model_name = os.path.basename(model_path)
    data, shots = load_dataset(data_name)
    model, tokenizer = load_model(model_path)
    line_break_id = tokenizer.encode("<end_of_turn>", add_special_tokens=False)[-1] if 'gemma' in model_name else tokenizer.eos_token_id
    generation_kwargs = {
        "max_new_tokens": 12,
        "do_sample": True,
        "temperature": 0.7,      # 可以试 0.7 / 1.0
        "top_p": 0.9,
        "eos_token_id": line_break_id,
        "pad_token_id": line_break_id,
        "use_cache": True,
    }


    collect_hiddens_dataset = REODQADataset(
        tokenizer=tokenizer,
        data=data,
        shots_set=shots,
        model_name=model_name,
        demonstration_pool_size=128,
        task="test_prompts"
    )

    record_keys = set()
    save_dir = save_dir_name
    os.makedirs(save_dir, exist_ok=True)


    for cur_seed in seeds_to_encode:

        save_path = save_dir + f"/{k_shot}shot-seed{cur_seed}-results_{data_name}_{base_name}.json"
        if os.path.exists(save_path):
            logger.info(f"the file ``{save_path}`` exists, pass")

        dataloader = collect_hiddens_dataset.collect_hiddens_dataloader(k_shot=k_shot, seed=cur_seed, batch_size=1)
        all_gen_results = []
        tqdm_bar = tqdm(dataloader, desc=f"seed={cur_seed}")
        accuracy = 0
        for batch in tqdm_bar:

            sample_key = str(batch["item_idx"]) + "-" + "-".join([str(idx) for idx in batch["demonstration_ids"]])
            if sample_key in record_keys:
                continue

            for prompt_type in ["no", "c"]: #, "a", "ca"]:
                gen_results = generate_and_evaluate_with_sampling(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=batch[prompt_type],
                    answers=batch["answer"],
                    generation_kwargs=generation_kwargs,
                    num_samples=num_sample
                )

                gen_results.update({
                    "item_idx": batch["item_idx"],
                    "demonstration_ids": batch["demonstration_ids"],
                    "prompt_type": prompt_type
                })

                loss = calculate_loss(model, batch[prompt_type + "_a"])
                gen_results.update({"loss": loss})

                if gen_results["sample_accuracy"] == 1:
                    accuracy += 1
                # else:
                #     logger.info("em = 0")

                all_gen_results.append(gen_results)

        json.dump(all_gen_results, open(save_path, "w"), indent=4, ensure_ascii=False)


def calculate_loss(model, input_ids):
    input_ids = input_ids.cuda()
    outputs = model(input_ids=input_ids, labels=input_ids)
    return outputs.loss.item()




def generate_and_evaluate_with_sampling(
    model,
    tokenizer,
    input_ids,
    answers,
    generation_kwargs,
    num_samples=20
):
    assert len(input_ids) == 1

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

        pred = pred.split(".")[0].strip()
        em = unified_em(pred, answers[0])

        samples.append({
            "pred": pred,
            "em": em
        })

        num_correct += em

    return {
        "answers": answers,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "sample_accuracy": num_correct / num_samples,
        "samples": samples
    }



def main():
    args = get_args()
    logger.info(f"\n{json.dumps(vars(args), indent=4)}")
    model_name = os.path.basename(args.model_path)
    model_family = model_name.split('-')[0]  
    base_name = ''.join([c for c in model_family if c.isalpha()])
    # args.base_name = base_name
    group_prompts_based_on_behaviours(
        seeds_to_encode=args.seeds_to_encode,
        model_path=args.model_path,
        k_shot=args.k_shot,
        data_name=args.data_name,
        save_dir_name=args.save_dir_name,
        base_name=base_name,
        num_sample=args.num_sample
    )


if __name__ == '__main__':
    main()
