import os
import argparse
import torch
from data_utils import load_dataset
import json
import logging
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import REODQADataset, GetGradientDataset
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--save_dir_name", type=str, default="")
    parser.add_argument("--data_name", type=str, default="hqa")
    parser.add_argument("--seeds_to_encode", type=int, nargs='+', default=[42])
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--acc_threshold", type=float, default=0.8)
    return parser.parse_args()

def rank_A_given_B(A, B, delta):
    """
    A: (m, n)
    B: (m, k)
    span(A) ⊆ span(B)

    return: rank(A)
    """
    

    # k x k
    G = B.T @ B

    # Moore–Penrose pseudoinverse (k x k)
    G_pinv = torch.linalg.pinv(G)

    # k x k
    M = B.T @ A @ A.T @ B

    # k x k
    K = G_pinv @ M @ G_pinv

    # eigenvalues (symmetric PSD)
    _, eigvals, _ = torch.linalg.svd(K.float(), full_matrices=False)
    # eigvals = eigvals.abs() 
    # eigvals = eigvals/ (eigvals.max() + 1e-12)
    eigvals, _ = torch.sort(eigvals, descending=True)
    # eigvals = torch.log(1+eigvals + 1e-6)
    rank = stable_rank(eigvals)
    return rank, eigvals.to('cpu')

def stable_rank(singular_vals):
    s2_sum = torch.sum(singular_vals**2)
    smax2 = singular_vals[0] + 1e-12
    return (s2_sum / smax2**2).item()

def participation_rank(singular_vals):
    s2 = singular_vals**2
    return (s2.sum()**2 /
           (s2.pow(2).sum() + 1e-12)).item()

class MLPActivationGradCollector:
    def __init__(self, model):
        """
       
        """
        self.model = model
        self.mlp_act_grads = {}
        self.mlp_hidden_states = {} 
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp

            def forward_hook(module, input, output, layer_idx=layer_idx):
                # output = W2 @ phi(W1 h) + b2
                self.mlp_hidden_states[layer_idx] = input[0].detach()  # 

            def backward_hook(module, grad_input, grad_output, layer_idx=layer_idx):
                # grad_output[0]: dL / d(mlp output)
                self.mlp_act_grads[layer_idx] = grad_output[0].detach()

            self.hooks.append(mlp.down_proj.register_forward_hook(forward_hook))
            self.hooks.append(mlp.register_full_backward_hook(backward_hook))

    def clear(self):
        self.mlp_act_grads.clear()
        self.mlp_hidden_states.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()



def sampling_results_to_QA(results, idx2item,args):
    """
    
    """
    data_lis = []
    for item in results:
        idx = item["item_idx"]
        demo_ids = item["demonstration_ids"]

        new_item = idx2item[idx].copy()
        new_item["prompt_type"] = item["prompt_type"]
        new_item["shots"] = [idx2item[i] for i in demo_ids]
        new_item["answer_em"] = item["answer_em"]

        if item["answer_em"]>= args.acc_threshold:
            new_item["pred_answer"] = [
                                    s["pred"]
                                    for s in item["samples"]
                                    if s["em"] == 1.0
                                ]
        elif item["answer_em"]< 1-args.acc_threshold:
            new_item["pred_answer"] = [
                                    s["pred"]
                                    for s in item["samples"]
                                    if s["em"] == 0.0
                                ]
            
        else:
            new_item["pred_answer"] = [
                                    s["pred"]
                                    for s in item["samples"]
                                ]

        data_lis.append(new_item)

    return data_lis


def find_all_mlp_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) and ".mlp.down_proj" in name:
            
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def get_gradients(dataloader, model, collector, target_modules, gradient_type,args):
    all_ranks = []
    ranks = {}
    gradient = {}
    # for name, param in model.named_parameters():
    #     for t in target_modules:
    #         if t in name and 'weight' in name:
    #             gradient[name] = 0
                # ranks[name] = [0]

    base_dir = f"./cashed_data_sampling_{args.data_name}_gflat"

    rank_dir = os.path.join(
        base_dir,
        f"g_rank_sampling_{args.delta}delta_{args.base_name}_entloss_1",
        gradient_type
    )
    sim_dir = os.path.join(
        base_dir,
        f"alignment_sampling_{args.delta}delta_{args.base_name}_entloss_1",
        gradient_type
    )

    os.makedirs(rank_dir, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)
    step = 0
    f_cnt = 0
    ratio_n = 0
    avg_ratio = 0
    for epochs in range(1):
        for inputs in tqdm(dataloader):
            input_ids, labels = inputs["input_ids"].to(model.device), inputs["labels"]
            output = model(input_ids)
            next_token_logits = output.logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            entropy_loss = -torch.sum(probs * log_probs)
            model.zero_grad()
            entropy_loss.backward()

            grad_v = collector.mlp_act_grads  

            cnt = torch.sum(labels != -100)
            step_similarities = {}
            # step_eigvals = {}
            ranks = {}

            for n, lp in model.named_parameters():
                if not any(t in n for t in target_modules):
                    continue
                if ".mlp.down_proj.weight" not in n or lp.grad is None:
                    continue
                gradient[n] = (lp.grad).detach().cpu()
                param = lp.grad.detach().reshape(-1, lp.grad.shape[-1])
                Q_h_norm = torch.linalg.norm(param.float(), ord='fro')

                act_grads_list = []
                param_grad_projected_list = []
                

                lname = int(n.split('.')[2])
                g = grad_v[lname]
                
                hidden_states = collector.mlp_hidden_states[lname]  # [batch, seq_len, hidden_dim]
                g_flat = g.detach().reshape(-1, g.shape[-1])  # flatten token/batch
                hs_flat = hidden_states.detach().reshape(-1, hidden_states.shape[-1])

                act_grads_list.extend([g_flat[i] for i in range(g_flat.shape[0])])
                                                                            

                
                if len(act_grads_list) > 0:
                    ratio_n += 1
                    rank_param, eigvals = rank_A_given_B(param.t().float(), hs_flat.t().float(), args.delta*2.0)
                    _, sigma_h, _ = torch.linalg.svd((hs_flat@hs_flat.t()).float(), full_matrices=False)
                    # singular_
                    singular_vals, _ = torch.sort(sigma_h, descending=True)
                    rank_hs = stable_rank(singular_vals)
                    alignment = rank_param / ( rank_hs+ 1e-12) 
                    step_similarities[n] = (alignment, hs_flat.shape[0]) 
                    ranks[n] = rank_param

                lp.grad = None
            collector.clear()

            f_cnt += cnt
            step += 1
            # print(gradient_type)

            # 保存结果
            torch.save(
                ranks,
                os.path.join(rank_dir, f"{step}.pt")
            )

            torch.save(
                step_similarities,
                os.path.join(sim_dir, f"{step}.pt")
            )
            # all_ranks.append(ranks)

    # return all_ranks


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
        model.cuda().train()
    for name, param in model.named_parameters():
        if ".mlp.down_proj" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return model, tokenizer


def split_sampling_results_into_4_groups(
    args,
    all_results,
    acc_threshold=0.8
):
    """
    """

    answer_data = []
    not_answer_data = []
    doc_answer_data = []
    doc_not_answer_data = []

    for item in all_results:
        acc = item["sample_accuracy"]
        prompt_type = item["prompt_type"]
        if acc >= acc_threshold:
            item["answer_em"] = 1 
        elif acc < 1.0-acc_threshold:
            item["answer_em"] = 0
            preds = [
                        s["pred"]
                        for s in item["samples"]
                    ]
            if len(set(preds)) == 1:
                continue
        else:
            continue

        if prompt_type == "no":
            if item["answer_em"] == 1:
                answer_data.append(item)
            else:
                not_answer_data.append(item)

        elif prompt_type == "c":
            if item["answer_em"] == 1:
                doc_answer_data.append(item)
            else:
                doc_not_answer_data.append(item)

    logger.info(
        f"split sampling results:\n"
        f"  answer_data: {len(answer_data)}\n"
        f"  not_answer_data: {len(not_answer_data)}\n"
        f"  doc_answer_data: {len(doc_answer_data)}\n"
        f"  doc_not_answer_data: {len(doc_not_answer_data)}"
    )

    return (
        answer_data,
        not_answer_data,
        doc_answer_data,
        doc_not_answer_data
    )


def gradient_weighted_pca(
    hs_flat: torch.Tensor,   # (T, d)
    g_flat: torch.Tensor,    # (T, d)
    eps: float = 1e-12,
):
    """
    Gradient-weighted PCA in hidden-state space (d x d).
    """

    weights = torch.linalg.norm(g_flat, dim=1)  # (T,)

    H = hs_flat  # (T, d)
    C = weights[:, None]* (H @ H.T)  # (T, T)


    U, s, Vh = torch.linalg.svd(C, full_matrices = False)

    # descending order
    # idx = torch.argsort(eigvals, descending=True)
    # eigvals = eigvals[idx]
    # eigvecs = eigvecs[:, idx]

    return s

def select_by_top_relative(
    singular_vals: torch.Tensor,
    log_drop_thresh: float = 1.0,
    max_k: int = None,
):
    """
    Select principal directions by log-relative cutoff.

    Keeps directions whose singular values are within
    exp(-log_drop_thresh) of the top one.

    Args:
        singular_vals: (r,) descending
        log_drop_thresh: delta in log-domain
            e.g. 1.0 ~ exp(-1) ~ 0.37
                 2.0 ~ exp(-2) ~ 0.14
                 3.0 ~ exp(-3) ~ 0.05
        max_k: optional hard cap

    Returns:
        k: number of principal directions
    """
    assert singular_vals.ndim == 1
    assert singular_vals.numel() > 0

    log_sigma = torch.log(singular_vals + 1e-12)
    log_sigma0 = log_sigma[0]

    mask = log_sigma >= (log_sigma0 - log_drop_thresh)
    k = int(mask.sum().item())

    if max_k is not None:
        k = min(k, max_k)

    k = max(k, 1)
    # print((k,singular_vals.size()))

    return k


def cap_dataset_size(data, max_size=600, seed=42):
    """
    """
    if len(data) <= max_size:
        return data
    random.seed(seed)
    return random.sample(data, k=max_size)



def main():
    args = get_args()
    logger.info(f"\n{json.dumps(vars(args), indent=4)}")

    model_name = os.path.basename(args.model_path)
    model_family = model_name.split('-')[0]
    base_name = ''.join([c for c in model_family if c.isalpha()])
    args.base_name = base_name

    # 1. load sampling results
    all_results = []
    for seed in args.seeds_to_encode:
        path = (
            args.save_dir_name +
            f"/{args.k_shot}shot-seed{seed}-results_{args.data_name}_{args.base_name}.json"
        )
        logger.info(f"load sampling file: {path}")
        all_results.extend(json.load(open(path)))

    # 2. split into four groups
    answer_data, not_answer_data, doc_answer_data, doc_not_answer_data = \
        split_sampling_results_into_4_groups(
            args,
            all_results,
            acc_threshold=args.acc_threshold
        )
    answer_data = cap_dataset_size(answer_data, max_size=1000, seed=42)
    not_answer_data = cap_dataset_size(not_answer_data, max_size=1000, seed=42)
    doc_answer_data = cap_dataset_size(doc_answer_data, max_size=1000, seed=42)
    doc_not_answer_data = cap_dataset_size(doc_not_answer_data, max_size=1000, seed=42)

    # 3. load model / dataset
    data, shots = load_dataset(args.data_name)
    model, tokenizer = load_model(args.model_path)
    collector = MLPActivationGradCollector(model)

    collect_dataset = REODQADataset(
        tokenizer=tokenizer,
        data=data,
        shots_set=shots,
        model_name=os.path.basename(args.model_path),
        demonstration_pool_size=128,
        task="test_prompts"
    )
    idx2item = collect_dataset.idx2item
    target_modules = find_all_mlp_linear_names(model)

    groups = [
        ("doc_not_answer", doc_not_answer_data),
        ("doc_answer", doc_answer_data),
        ("answer", answer_data),
        ("not_answer", not_answer_data),
    ]

    for gradient_type, results in groups:
        if len(results) == 0:
            continue

        logger.info(f"processing {gradient_type}, size={len(results)}")

        data_lis = sampling_results_to_QA(results, idx2item,args)
        dataset = GetGradientDataset(
            os.path.basename(args.model_path),
            tokenizer,
            data_lis
        )
        dataloader = dataset.get_dataloader(
            batch_size=1,
            num_workers=8,
            shuffle=False
        )

        get_gradients(
            dataloader,
            model,
            collector,
            target_modules,
            gradient_type,
            args
        )




if __name__ == '__main__':
    main()
