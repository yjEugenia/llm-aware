import argparse
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_curve, auc

# 假设这些可以从你的模块中正常导入
from utils import AlignmentDataset_0, AlignmentProbe_1 
from config import Config

def load_test_data(task_name, model_name, batch_size, num_workers):
    """根据测试任务名称加载数据"""
    data_path = f"./cashed_data_paraphrased_{task_name}/alignment_sampling_0.0delta_{model_name}_paraphrased/"
    print(f"[*] Loading test data from: {data_path}")
    
    in_dataset = AlignmentDataset_0(data_dir=data_path + "answer", label=1)
    out_dataset = AlignmentDataset_0(data_dir=data_path + "not_answer", label=0)
    
    
    if task_name == 'mmlu' or args.doc_mode == 'no_doc':
        full_dataset = ConcatDataset([out_dataset,in_dataset])
    else:
        doc_in_dataset = AlignmentDataset_0(data_dir=data_path + "doc_answer", label=1)
        doc_out_dataset = AlignmentDataset_0(data_dir=data_path + "doc_not_answer", label=0)
        full_dataset = ConcatDataset([doc_in_dataset,doc_out_dataset ,in_dataset, out_dataset]) #,in_dataset, out_dataset
        
    # 测试时不需要打乱 (shuffle=False)
    test_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return test_loader

def evaluate_model(model, test_loader, device, boundary_threshold=0.5):
    """评估模型并返回 Accuracy 和 AUROC"""
    model.eval()
    
    preds_continuous = []
    preds_binary = []
    labels = []
    criterion = torch.nn.BCELoss()
    val_losses = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            val_losses.append(loss.item())

            # 提取预测值
            outputs = outputs.squeeze(dim=-1) # 避免 batch_size=1 时把全维度 squeeze 掉
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            preds_continuous.extend(outputs.cpu().numpy().tolist())
            preds_binary.extend((outputs >= boundary_threshold).int().cpu().numpy().tolist())
            labels.extend(y_batch.cpu().numpy().tolist())

    labels = np.array(labels)
    preds_continuous = np.array(preds_continuous)
    preds_binary = np.array(preds_binary)

    # 计算指标
    TP = np.sum((preds_binary == 1) & (labels == 1))
    FP = np.sum((preds_binary == 1) & (labels == 0))
    FN = np.sum((preds_binary == 0) & (labels == 1))
    TN = np.sum((preds_binary == 0) & (labels == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    fpr, tpr, thresholds = roc_curve(labels, preds_continuous)
    roc_auc = auc(fpr, tpr)

    return accuracy, roc_auc, np.mean(val_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Alignment Probe Generalization")
    parser.add_argument("--train_task", type=str, required=True, help="Task the model was trained on (e.g., hqa)")
    parser.add_argument("--test_task", type=str, required=True, help="Task to evaluate the model on (e.g., mmlu)")
    parser.add_argument("--model_name", type=str, default="llama", help="Name of the LLM model (e.g., gemma)")
    parser.add_argument("--base_save_dir", type=str, default="./checkpoints", help="Base directory where models are saved")
    parser.add_argument("--ckpt_name", type=str, default="model_best_auroc.pth", help="Checkpoint file name")
    parser.add_argument("--doc_mode", type=str, default="with_doc", choices=["with_doc", "no_doc"], 
                        help="Train probe on data 'with_doc' or 'no_doc'")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    config = Config.from_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 确定 checkpoint 路径并加载 Config (可选，确保参数一致性)
    model_dir = Path(args.base_save_dir) / args.train_task / args.model_name / args.doc_mode /"gen"
    ckpt_path = model_dir / args.ckpt_name
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"\n[{args.train_task} -> {args.test_task}]")
    print(f"[*] Loading model from: {ckpt_path}")

    # 2. 加载测试数据
    test_loader = load_test_data(
        task_name=args.test_task, 
        model_name=args.model_name, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers
    )

    # 3. 初始化模型结构
    sample_X, _ = next(iter(test_loader))
    input_dim = sample_X.shape[1]
    model = AlignmentProbe_1(input_dim=input_dim).to(device)

    # 4. 加载预训练权重
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("[*] Model weights loaded successfully.")

    # 5. 进行评估
    acc, auroc, test_loss = evaluate_model(model, test_loader, device, config.boundary_threshold)
    
    print("-" * 30)
    print(f"Train on  : {args.train_task}")
    print(f"Test on   : {args.test_task}")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {auroc:.4f}")
    print("-" * 30)



#python collect_hidden_states/probe/evaluate.py --train_task hqa --test_task nq --ckpt_name model_best_acc.pth