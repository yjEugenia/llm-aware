# -*- coding: utf-8 -*-
import os
import json
import logging
from typing import Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class AlignmentProbeTrainer:

    def __init__(self, train_loader, val_loader, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.criterion = None
        self.optimizer = None

    # ---------------------------
    # Setup
    # ---------------------------
    def setup_model(self):

        sample_X, _ = next(iter(self.train_loader))
        input_dim = sample_X.shape[1]

        self.model = AlignmentProbe(input_dim=input_dim).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            # weight_decay=self.config.weight_decay
        )

    # ---------------------------
    # Train loop
    # ---------------------------
    def train(self):

        best_val_acc = 0.0

        for epoch in range(self.config.num_epochs):

            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()

            self._log_metrics(epoch, train_loss, val_metrics)

            if val_metrics["Accuracy"] > best_val_acc:
                best_val_acc = val_metrics["Accuracy"]
                self.save_model()

    # ---------------------------
    # Train one epoch
    # ---------------------------
    def _train_epoch(self):

        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).long()   

            self.optimizer.zero_grad()

            outputs = self.model(X_batch)              # (B, 2)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ---------------------------
    # Validation
    # ---------------------------
    def _validate_epoch(self):

        self.model.eval()

        all_preds = []
        all_probs = []
        all_labels = []
        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in self.val_loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).long()

                outputs = self.model(X_batch)              # (B, 2)
                loss = self.criterion(outputs, y_batch)
                val_losses.append(loss.item())

                probs = torch.softmax(outputs, dim=1)[:, 1]  
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs)

        metrics = {
            "val_loss": np.mean(val_losses),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
        }

        return metrics

    # ---------------------------
    # Save model
    # ---------------------------
    def save_model(self):

        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / "model.pth")

        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=4)

    # ---------------------------
    # Logging
    # ---------------------------
    def _log_metrics(self, epoch, train_loss, val_metrics):

        logger.info(f"\nEpoch {epoch}")
        logger.info(f"Train Loss: {train_loss:.4f}")

        for k, v in val_metrics.items():
            logger.info(f"{k}: {v:.4f}")
            
class AlignmentProbeTrainer_0:
    """
    Trainer class for Alignment Probe.
    The probe takes a layer-wise alignment vector (1 x L) as input
    and predicts whether the sample lies within the knowledge boundary.
    """

    def __init__(self, train_loader, val_loader, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = None

    def setup_model(self):
        """Initialize model, loss, optimizer, and scheduler."""
        input_dim = next(iter(self.train_loader))[0].shape[1]
        self.model = AlignmentProbe_1(input_dim=input_dim).to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience
        )

    def train(self):
        """Main training loop."""
        
        best_val_acc = 0.0
        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()

            self._log_metrics(epoch, train_loss, val_metrics)

            if val_metrics["Accuracy"] > best_val_acc:
                best_val_acc = val_metrics["Accuracy"]
                logger.info(f"--> New best Accuracy: {best_val_acc:.4f}. Saving model...")
                self.save_model(filename="model_best_acc.pth")
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_model(filename="model_best_loss.pth")

            self.scheduler.step(val_metrics["val_loss"])

    def save_model(self, filename="model.pth"):
        """Save trained model and configuration."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / filename)

        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=4)

    def _train_epoch(self):
        """One training epoch."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch.unsqueeze(1)) #.unsqueeze(1)
            loss.backward()
            self.optimizer.step()
            # self.model.zero_grad()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        """Validation loop with detailed metrics."""
        self.model.eval()

        val_losses = []
        preds_binary = []
        preds_continuous = []
        labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.unsqueeze(1))
                val_losses.append(loss.item())

                outputs = outputs.squeeze()
                preds_continuous.extend(outputs.cpu().numpy().tolist())
                preds_binary.extend((outputs >= self.config.boundary_threshold).int().cpu().numpy().tolist())
                labels.extend(y_batch.cpu().numpy().tolist())

        labels = np.array(labels)
        preds_continuous = np.array(preds_continuous)
        preds_binary = np.array(preds_binary)

        TP = np.sum((preds_binary == 1) & (labels == 1))
        FP = np.sum((preds_binary == 1) & (labels == 0))
        FN = np.sum((preds_binary == 0) & (labels == 1))
        TN = np.sum((preds_binary == 0) & (labels == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        f1 = f1_score(labels, preds_binary)

        fpr, tpr, thresholds = roc_curve(labels, preds_continuous)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        pcc = np.corrcoef(labels, preds_continuous)[0, 1]
        mean_neg_pred = preds_continuous[labels == 0].mean()
        mean_pos_pred = preds_continuous[labels == 1].mean()

        metrics = {
            "val_loss": np.mean(val_losses),
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PCC": pcc,
            "optimal_threshold": optimal_threshold,
            "mean_neg_pred": mean_neg_pred,
            "mean_pos_pred": mean_pos_pred,
        }

        return metrics

    def _log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict):
        """Logging helper."""
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        for k, v in val_metrics.items():
            logger.info(f"{k}: {v:.4f}")

class CoTAlignmentProbeTrainer:
    def __init__(self, train_loader, val_loader, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader

    def setup_model(self):
        sample_X, _ = next(iter(self.train_loader))
        input_dim = sample_X.shape[-1]

        self.model = CoTAlignmentProbe(
            input_dim=input_dim
        ).to(self.device)


        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience
        )
    def train(self):
        """Main training loop."""
        best_val_loss = float("inf")
        best_val_acc = 0.0

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()

            self._log_metrics(epoch, train_loss, val_metrics)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_model(filename="model_best_loss.pth")

            if val_metrics["Accuracy"] > best_val_acc:
                best_val_acc = val_metrics["Accuracy"]
                logger.info(f"--> New best Accuracy: {best_val_acc:.4f}. Saving model...")
                self.save_model(filename="model_best_acc.pth")

            self.scheduler.step(val_metrics["val_loss"])

    def save_model(self, filename="model.pth"):
        """Save trained model and configuration."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / filename)

        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=4)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        """Validation loop with detailed metrics."""
        self.model.eval()

        val_losses = []
        preds_binary = []
        preds_continuous = []
        labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.unsqueeze(1))
                val_losses.append(loss.item())

                outputs = outputs.squeeze()
                preds_continuous.extend(outputs.cpu().numpy().tolist())
                preds_binary.extend((outputs >= self.config.boundary_threshold).int().cpu().numpy().tolist())
                labels.extend(y_batch.cpu().numpy().tolist())

        labels = np.array(labels)
        preds_continuous = np.array(preds_continuous)
        preds_binary = np.array(preds_binary)

        TP = np.sum((preds_binary == 1) & (labels == 1))
        FP = np.sum((preds_binary == 1) & (labels == 0))
        FN = np.sum((preds_binary == 0) & (labels == 1))
        TN = np.sum((preds_binary == 0) & (labels == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        f1 = f1_score(labels, preds_binary)

        fpr, tpr, thresholds = roc_curve(labels, preds_continuous)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        pcc = np.corrcoef(labels, preds_continuous)[0, 1]
        mean_neg_pred = preds_continuous[labels == 0].mean()
        mean_pos_pred = preds_continuous[labels == 1].mean()

        metrics = {
            "val_loss": np.mean(val_losses),
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PCC": pcc,
            "optimal_threshold": optimal_threshold,
            "mean_neg_pred": mean_neg_pred,
            "mean_pos_pred": mean_pos_pred,
        }

        return metrics

    def _log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict):
        """Logging helper."""
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        for k, v in val_metrics.items():
            logger.info(f"{k}: {v:.4f}")
