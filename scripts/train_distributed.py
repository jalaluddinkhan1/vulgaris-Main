"""
Distributed VULGARIS training entry-point.

This script is designed to be launched via torchrun, which handles process
spawning, rank assignment, and the rendezvous handshake automatically.

Quick-start
-----------
  # Single node, 4 parallel workers (e.g. 4 CPU cores or 4 GPUs):
  torchrun --nproc_per_node=4 scripts/train_distributed.py \\
      --config configs/base.yaml --data_dir /data/training

  # Two nodes, 4 workers each (8 total):
  # Run on node-0 (the master):
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
      --master_addr=<NODE0_IP> --master_port=29500 \\
      scripts/train_distributed.py --config configs/base.yaml

  # Run on node-1 simultaneously:
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
      --master_addr=<NODE0_IP> --master_port=29500 \\
      scripts/train_distributed.py --config configs/base.yaml

How it works
------------
  1. init_process_group() — each worker registers with the rendezvous server.
  2. broadcast_parameters() — rank 0 sends its weights to all other ranks so
     all workers start from exactly the same initialisation.
  3. DistributedSampler — each rank receives a disjoint shard of the dataset.
  4. TrainingPipeline.train_step() — after backward, allreduce_gradients()
     sums gradients across all ranks before the optimizer update.
  5. save_checkpoint() — only rank 0 writes to disk.

Resuming from a checkpoint
---------------------------
  Pass --resume checkpoints/checkpoint_latest.npz — all ranks load the same
  checkpoint (broadcast_parameters is still called to ensure consistency).
"""

import argparse
import os
import sys

import numpy as np

# Add project root to path so imports work regardless of cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.distributed import (
    init_process_group, destroy_process_group,
    broadcast_parameters, barrier,
    get_rank, get_world_size, is_main_process,
)
from training.distributed_sampler import DistributedSampler


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Distributed VULGARIS training")
    p.add_argument("--config",      required=True,
                   help="Path to ModelConfig YAML file")
    p.add_argument("--data_dir",    required=True,
                   help="Directory containing pre-tokenised .npz shard files")
    p.add_argument("--output_dir",  default="checkpoints",
                   help="Where to write checkpoints (rank 0 only)")
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=32,
                   help="Per-rank batch size (global = batch_size × world_size)")
    p.add_argument("--resume",      default=None,
                   help="Path to .npz checkpoint to resume from")
    p.add_argument("--domain_idx",  type=int, default=0)
    p.add_argument("--log_interval",type=int, default=50,
                   help="Print metrics every N steps (rank 0 only)")
    p.add_argument("--backend",     default="gloo",
                   choices=["gloo", "nccl"],
                   help="torch.distributed backend (gloo=CPU, nccl=GPU)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str):
    """
    Load training data from .npz shards in data_dir.

    Each shard is expected to contain arrays 'x' (B, C, T) and 'y' (B, out_dim).
    Returns two stacked arrays: X (N, C, T) and Y (N, out_dim).
    """
    shards_x, shards_y = [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        d = np.load(os.path.join(data_dir, fname), allow_pickle=False)
        if "x" not in d or "y" not in d:
            continue
        shards_x.append(d["x"])
        shards_y.append(d["y"])

    if not shards_x:
        raise FileNotFoundError(
            f"No .npz shards with 'x'/'y' arrays found in {data_dir}"
        )

    X = np.concatenate(shards_x, axis=0)
    Y = np.concatenate(shards_y, axis=0)
    return X, Y


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Initialise distributed process group ──────────────────────────────
    init_process_group(backend=args.backend)
    rank       = get_rank()
    world_size = get_world_size()

    if is_main_process():
        print(f"[dist] world_size={world_size}, backend={args.backend}")
        print(f"[dist] per-rank batch={args.batch_size}, "
              f"global batch={args.batch_size * world_size}")

    # ── 2. Build model + training components ────────────────────────────────
    from config import ModelConfig
    from model.vulgaris import Vulgaris
    from training.loss import VulgarisLoss
    from training.optimizer import SpectralAdamW, CosineSchedule
    from training.pipeline import TrainingPipeline

    config = ModelConfig.from_yaml(args.config)
    config.training.checkpoint_dir = args.output_dir

    model = Vulgaris(config)

    optimizer = SpectralAdamW(
        list(model.parameters()),
        lr=getattr(config.training, "lr", 3e-4),
        weight_decay=getattr(config.training, "weight_decay", 0.01),
    )
    total_steps = args.epochs * 1000   # refined after dataset load
    scheduler = CosineSchedule(
        optimizer,
        warmup_steps=getattr(config.training, "warmup_steps", 200),
        max_steps=total_steps,
        min_lr=getattr(config.training, "min_lr", 1e-5),
    )
    loss_fn = VulgarisLoss(config)

    pipeline = TrainingPipeline(
        model=model,
        config=config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # ── 3. Sync initial weights from rank 0 to all ranks ────────────────────
    if args.resume:
        if is_main_process():
            pipeline.load_checkpoint(args.resume)
        barrier()                      # wait for rank 0 to finish loading
    broadcast_parameters(model)        # ensure all ranks have identical weights

    # ── 4. Load dataset and build sampler ───────────────────────────────────
    X, Y = load_dataset(args.data_dir)
    N = X.shape[0]
    total_steps = args.epochs * (N // (args.batch_size * world_size))
    scheduler.max_steps = total_steps  # refine now that we know dataset size

    sampler = DistributedSampler(
        dataset_size=N,
        shuffle=True,
        seed=42,
    )

    if is_main_process():
        print(f"[dist] dataset N={N}, per-rank samples/epoch={len(sampler)}")

    # ── 5. Training loop ─────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        indices = sampler.get_indices(epoch)
        batches = DistributedSampler.build_batches(
            indices, batch_size=args.batch_size, drop_last=True
        )

        epoch_losses = []
        for step, batch_idx in enumerate(batches):
            x_batch = X[batch_idx]    # (B, C, T)
            y_batch = Y[batch_idx]    # (B, out_dim)

            metrics = pipeline.train_step(
                x_batch, y_batch, domain_idx=args.domain_idx
            )
            epoch_losses.append(metrics.get("total_loss", float("nan")))

            if is_main_process() and (step + 1) % args.log_interval == 0:
                avg = float(np.nanmean(epoch_losses[-args.log_interval:]))
                print(f"[epoch {epoch+1}/{args.epochs}  "
                      f"step {pipeline.step_count}]  "
                      f"loss={avg:.6f}  lr={optimizer.lr:.2e}")

        if is_main_process():
            epoch_avg = float(np.nanmean(epoch_losses))
            print(f"[epoch {epoch+1}/{args.epochs}] avg_loss={epoch_avg:.6f}")
            pipeline.save_checkpoint(tag=f"epoch_{epoch+1:04d}")

        barrier()   # all ranks wait before starting the next epoch

    # ── 6. Cleanup ───────────────────────────────────────────────────────────
    if is_main_process():
        pipeline.save_checkpoint(tag="final")
        print("[dist] Training complete.")

    destroy_process_group()


if __name__ == "__main__":
    main()
