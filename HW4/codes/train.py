

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.dataset_utils import PromptTrainDataset
from utils.loss_utils import edge_loss, freq_loss
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim

from net.model import PromptIR
from options import options as opt


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.net = PromptIR(decoder=True)

        self.loss_fn = nn.L1Loss()
        self.edge_loss_fn = edge_loss
        self.freq_loss_fn = freq_loss

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):

        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        # Lightning already moves tensors to correct device
        restored = self.net(degrad_patch)

        # =========================
        # Loss
        # =========================
        l1 = self.loss_fn(restored, clean_patch)

        edge = self.edge_loss_fn(restored, clean_patch)

        freq = self.freq_loss_fn(restored, clean_patch)

        loss = l1 + 0.1 * edge + 0.05 * freq

        # =========================
        # Metrics
        # =========================
        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)

        # =========================
        # Logging
        # =========================
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "l1_loss",
            l1,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "edge_loss",
            edge,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "freq_loss",
            freq,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "psnr",
            psnr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "ssim",
            ssim,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        # =========================
        # TensorBoard Image Logging
        # =========================
        if self.global_step % 500 == 0:

            grid = torchvision.utils.make_grid(
                torch.cat([
                    degrad_patch[:4],   # input
                    restored[:4],       # output
                    clean_patch[:4],    # ground truth
                ], dim=0),
                nrow=4,
                normalize=True,
                scale_each=True
            )

            self.logger.experiment.add_image(
                "Sample_Input_Output_GT",
                grid,
                global_step=self.global_step
            )

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            self.parameters(),
            lr=2e-4
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )

        return [optimizer], [scheduler]


def main():

    print("Options")
    print(opt)

    # =========================
    # TensorBoard Logger
    # =========================
    logger = TensorBoardLogger(
        save_dir="logs",
        name=opt.wandb_name
    )

    # =========================
    # Dataset
    # =========================
    trainset = PromptTrainDataset(opt)

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    # =========================
    # Checkpoint
    # =========================
    checkpoint_callback = ModelCheckpoint(
        monitor="psnr",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=opt.ckpt_dir,
        filename=opt.ckpt_name + "-{epoch:02d}-{psnr:.2f}"
    )

    # =========================
    # Model
    # =========================
    model = PromptIRModel()

    # =========================
    # Trainer
    # =========================
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],

        # useful defaults
        log_every_n_steps=10,
        precision="16-mixed"
    )

    # =========================
    # Train
    # =========================
    # trainer.fit(
    #     model=model,
    #     train_dataloaders=trainloader
    # )

    trainer.fit(
    model=model,
    train_dataloaders=trainloader,
    ckpt_path=opt.resume_ckpt
)

    # =========================
    # Best checkpoint
    # =========================
    print(f"\n✅ Best PSNR: {checkpoint_callback.best_model_score.item():.4f}")

    print(f"📦 Best model path:")
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    main()



# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --ckpt_dir train_ckpt_freq \
#   --ckpt_name best_rainsnow_freq \
#   --wandb_name RainySnow_freq \
#   --resume_ckpt train_ckpt_aug20/last.ckpt
# --num_aug 20