"""Lightning Module for the VinDr-PCXR DenseNet121 baseline classifier."""
from __future__ import annotations

from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchxrayvision as xrv

from core.config import cfg
from core.metrics import compute_metrics, format_metrics_table, mean_auroc


class VinDrClassifier(pl.LightningModule):
    """DenseNet121 fine-tuned for 6-class multilabel VinDr-PCXR classification.

    Two-phase training schedule:
      Phase 1 (epochs 0 … warmup_epochs-1): backbone frozen (lr = 0.0),
        only the new 6-class head is trained at ``lr_head``.
      Phase 2 (epoch warmup_epochs onward): full network at ``lr_backbone``.
        Early stopping begins counting from here.

    Loss: BCEWithLogitsLoss with per-class ``pos_weight`` to counteract the
    severe class imbalance in VinDr-PCXR.
    """

    def __init__(
        self,
        num_classes: int = 6,
        warmup_epochs: int = 3,
        lr_head: float = 1e-4,
        lr_backbone: float = 1e-5,
        class_names: Optional[List[str]] = None,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weights"])
        self.class_names = class_names or cfg.viable_classes

        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        self.model.op_threshs = None

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weights.clone() if pos_weights is not None else None
        )

        # ---- epoch-level buffers for metric aggregation --------------
        self._val_probs: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._test_probs: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        imgs, labels = batch
        loss = self.criterion(self(imgs), labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch: Any, batch_idx: int) -> None:
        imgs, labels = batch
        logits = self(imgs)
        self.log(
            "val/loss",
            self.criterion(logits, labels),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self._val_probs.append(torch.sigmoid(logits).cpu())
        self._val_targets.append(labels.cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return
        probs = torch.cat(self._val_probs).float().numpy()
        targets = torch.cat(self._val_targets).float().numpy()
        metrics = compute_metrics(targets, probs, self.class_names)
        self.log("val/auroc", mean_auroc(metrics), prog_bar=True)
        for name, m in metrics.items():
            self.log(f"val/auroc_{name}", m["auroc"])
        self._val_probs.clear()
        self._val_targets.clear()


    def test_step(self, batch: Any, batch_idx: int) -> None:
        imgs, labels = batch
        self._test_probs.append(torch.sigmoid(self(imgs)).cpu())
        self._test_targets.append(labels.cpu())

    def on_test_epoch_end(self) -> None:
        probs = torch.cat(self._test_probs).float().numpy()
        targets = torch.cat(self._test_targets).float().numpy()
        metrics = compute_metrics(targets, probs, self.class_names)
        self.log("test/auroc", mean_auroc(metrics))
        for name, m in metrics.items():
            self.log(f"test/auroc_{name}", m["auroc"])
            self.log(f"test/f1_{name}", m["f1"])
            self.log(f"test/sensitivity_{name}", m["sensitivity"])
            self.log(f"test/specificity_{name}", m["specificity"])
        print("\n" + format_metrics_table(metrics))
        self._test_probs.clear()
        self._test_targets.clear()


    def configure_optimizers(self) -> Any:
        # Phase 1: backbone hard-frozen (no grad computation), head only.
        # Phase 2: backbone unfrozen + optimiser rebuilt in on_train_epoch_start.
        for p in self.model.features.parameters():
            p.requires_grad = False
        return torch.optim.AdamW(
            self.model.classifier.parameters(),
            lr=self.hparams.lr_head,
        )

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.warmup_epochs:
            # Hard-unfreeze backbone and rebuild optimiser with two param groups.
            for p in self.model.features.parameters():
                p.requires_grad = True
            self.trainer.optimizers = [
                torch.optim.AdamW(
                    [
                        {"params": self.model.features.parameters(), "lr": self.hparams.lr_backbone},
                        {"params": self.model.classifier.parameters(), "lr": self.hparams.lr_head},
                    ]
                )
            ]
            self.print(
                f"\n[Phase 2] Epoch {self.current_epoch}: "
                f"backbone unfrozen — lr → {self.hparams.lr_backbone}"
            )
