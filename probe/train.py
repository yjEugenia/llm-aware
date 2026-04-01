import argparse
import sys
import os
from torch.utils.data import random_split, DataLoader, ConcatDataset
from utils import AlignmentDataset_0
from alignment_probing import AlignmentProbeTrainer_0
from config import Config
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alignment Probe")
    parser.add_argument("--task_name", type=str, default="mmlu", help="Name of the task (e.g., hqa, math)")
    parser.add_argument("--model_name", type=str, default="gemma", help="Name of the model (e.g., llama, qwen)")
    parser.add_argument("--base_save_dir", type=str, default="./checkpoints", help="Base directory to save models")
    
    # ============================
    parser.add_argument("--doc_mode", type=str, default="no_doc", choices=["with_doc", "no_doc"], 
                        help="Train probe on data 'with_doc' or 'no_doc'")
    # ================================================

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    
    config = Config.from_args()
    
    #
    config.save_dir = os.path.join(args.base_save_dir, args.task_name, args.model_name, args.doc_mode, "gen")
    print(f"[*] Model and config will be saved to: {config.save_dir}")

    random.seed(42)
    data_path = f"./cashed_data_sampling_{args.task_name}_gflat/alignment_sampling_0.0delta_{args.model_name}_gen/"
    
    # =============== ===============
    datasets_to_concat = []

    if args.doc_mode == "only_doc":
        if args.task_name == 'mmlu':
            raise ValueError("MMLU does not have doc data folders configured in this setup.")
        
        print("[*] Loading WITH DOC datasets...")
        doc_in_dataset = AlignmentDataset_0(data_dir=data_path + "doc_answer", label=1)
        doc_out_dataset = AlignmentDataset_0(data_dir=data_path + "doc_not_answer", label=0)
        datasets_to_concat = [doc_in_dataset, doc_out_dataset]

    elif args.doc_mode == "no_doc":
        print("[*] Loading NO DOC datasets...")
        in_dataset = AlignmentDataset_0(data_dir=data_path + "answer", label=1)
        out_dataset = AlignmentDataset_0(data_dir=data_path + "not_answer", label=0)
        datasets_to_concat = [in_dataset,out_dataset,]

    elif args.doc_mode == "with_doc":
        print("[*] Loading NO + WITH DOC datasets...")
        in_dataset = AlignmentDataset_0(data_dir=data_path + "answer", label=1)
        out_dataset = AlignmentDataset_0(data_dir=data_path + "not_answer", label=0)
        doc_in_dataset = AlignmentDataset_0(data_dir=data_path + "doc_answer", label=1)
        doc_out_dataset = AlignmentDataset_0(data_dir=data_path + "doc_not_answer", label=0)
        # print(len(in_dataset),len(out_dataset),len(doc_in_dataset) ,len(doc_out_dataset) )
        datasets_to_concat = [doc_in_dataset, doc_out_dataset, in_dataset, out_dataset]
    # ==============================================================

    full_dataset = ConcatDataset(datasets_to_concat)
    print(f"[*] Total valid samples for {args.doc_mode}: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    trainer = AlignmentProbeTrainer_0(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    trainer.setup_model()
    trainer.train()
