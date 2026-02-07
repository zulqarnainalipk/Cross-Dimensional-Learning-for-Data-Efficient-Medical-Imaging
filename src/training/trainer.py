"""
Training utilities and trainer class.
"""

import os
import time
from typing import Dict, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm


class Trainer:
    """
    Training class for cross-dimensional knowledge transfer.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        config,
        device: torch.device = torch.device('cuda'),
        log_dir: str = 'logs',
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            Scheduler for learning rate
            config: Configuration object
            device: Device to train on
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # Move data to device
            if isinstance(batch, (tuple, list)):
                data = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                data = batch.to(self.device)
                targets = None

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)

            # Compute loss
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs

            # Backward pass
            loss.backward()

            # Gradient clipping
            if hasattr(self.config, 'gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            # Compute accuracy
            if targets is not None:
                if hasattr(outputs, 'logits'):
                    preds = outputs.logits.argmax(dim=1)
                else:
                    preds = outputs.argmax(dim=1)

                correct = (preds == targets).sum().item()
                total_correct += correct
                total_samples += targets.size(0)

            total_loss += loss.item()

        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_acc': total_correct / total_samples if total_samples > 0 else 0.0
        }

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                data = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                data = batch.to(self.device)
                targets = None

            outputs = self.model(data)

            if isinstance(outputs, tuple):
                loss = outputs[0]
                logits = outputs[1] if len(outputs) > 1 else None
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                logits = outputs.logits if hasattr(outputs, 'logits') else None

            total_loss += loss.item()

            if targets is not None:
                if logits is not None:
                    preds = logits.argmax(dim=1)
                else:
                    preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                correct = (preds == targets).sum().item()
                total_correct += correct
                total_samples += targets.size(0)

        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': total_correct / total_samples if total_samples > 0 else 0.0
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None
    ) -> Dict:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])

            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])

            # Learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            history['lr'].append(current_lr)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Check for improvement
            if val_metrics['val_acc'] > self.best_metric:
                self.best_metric = val_metrics['val_acc']
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print(f"  New best model saved! Acc: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if (early_stopping_patience is not None and
                    self.patience_counter >= early_stopping_patience):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return history

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str) -> int:
        """
        Load model checkpoint.

        Args:
            filename: Name of checkpoint file

        Returns:
            Epoch number
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)

        return self.current_epoch


def create_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config,
    device: torch.device = torch.device('cuda'),
    log_dir: str = 'logs',
    checkpoint_dir: str = 'checkpoints'
) -> Trainer:
    """
    Factory function to create trainer.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        config: Configuration object
        device: Device to train on
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints

    Returns:
        Trainer instance
    """
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
