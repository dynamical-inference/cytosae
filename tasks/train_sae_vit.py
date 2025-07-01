import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ['HF_HOME'] = '/home/nsml/tmp/huggingface_cache'

import argparse
from typing import Any
import multiprocessing
import torch
import wandb
from PIL import Image
from datasets import load_dataset

from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.sae_trainer import SAETrainer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.utils import get_scheduler
from src.sae_training.vit_activations_store import ViTActivationsStore
from tasks.utils import (
    DATASET_INFO,
    get_classnames,
    load_hooked_vit,
)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_token", action="store_true", default=None)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument(
        "--model_name", type=str, default="dinov2_vitb14"
    )
    parser.add_argument("--module_name", type=str, default="resid")
    parser.add_argument("--block_layer", type=int, default=-2)
    parser.add_argument("--clip_dim", type=int, default=768)

    parser.add_argument("--dataset", type=str, default="mll23")
    parser.add_argument("--use_cached_activations", action="store_true", default=None)
    parser.add_argument("--cached_activations_path", type=str)
    parser.add_argument("--expansion_factor", type=int, default=64)
    parser.add_argument("--b_dec_init_method", type=str, default="geometric_median")
    parser.add_argument("--gated_sae", action="store_true", default=None)
    # Training Parameters
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--l1_coefficient", type=float, default=0.00008)
    parser.add_argument("--lr_scheduler_name", type=str, default="constantwithwarmup")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_warm_up_steps", type=int, default=500)
    parser.add_argument("--total_training_tokens", type=int, default=1_000_000)
    parser.add_argument("--n_batches_in_store", type=int, default=15)
    parser.add_argument("--mse_cls_coefficient", type=float, default=1.0)
    # Dead Neurons and Sparsity
    parser.add_argument("--use_ghost_grads", action="store_true", default=False)
    parser.add_argument("--feature_sampling_method")
    parser.add_argument("--feature_sampling_window", type=int, default=64)
    parser.add_argument("--dead_feature_window", type=int, default=64)
    parser.add_argument("--dead_feature_threshold", type=float, default=1e-6)
    # WANDB
    parser.add_argument("--log_to_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="patch_sae_test")
    parser.add_argument("--wandb_entity", type=str, default="m-f-dasdelen-helmholtz-munich")
    parser.add_argument("--wandb_log_frequency", type=int, default=20)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_checkpoints", type=int, default=20)
    parser.add_argument("--checkpoint_path", type=str, default="out/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    # resume
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--run_name", type=str, default="train")
    parser.add_argument("--start_training_steps", type=int, default=0)
    parser.add_argument("--pt_name", type=str)

    parser.add_argument(
        "--vit_type", type=str, default="custom", help="choose between [base, maple]"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default = './DinoBloom-B.pth',
        help="CLIP model path in the case of not using the default",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="CLIP config path in the case of using maple",
    )
    args = parser.parse_args()

    print(f"Using {args.device}")
    cfg = ViTSAERunnerConfig(
        class_token=args.class_token,
        image_width=args.image_width,
        image_height=args.image_height,
        model_name= args.model_name,
        module_name=args.module_name,
        block_layer=args.block_layer,
        dataset_path=DATASET_INFO[args.dataset]["path"],
        image_key="image",
        label_key="label",
        use_cached_activations=args.use_cached_activations,
        cached_activations_path=args.cached_activations_path,
        d_in=args.clip_dim,
        expansion_factor=args.expansion_factor,
        b_dec_init_method=args.b_dec_init_method,
        gated_sae=args.gated_sae,
        lr=args.lr,
        l1_coefficient=args.l1_coefficient,
        lr_scheduler_name=args.lr_scheduler_name,
        batch_size=args.batch_size,
        lr_warm_up_steps=args.lr_warm_up_steps,
        total_training_tokens=args.total_training_tokens,
        n_batches_in_store=args.n_batches_in_store,
        mse_cls_coefficient=args.mse_cls_coefficient,
        use_ghost_grads=args.use_ghost_grads,
        feature_sampling_method=args.feature_sampling_method,
        feature_sampling_window=args.feature_sampling_window,
        dead_feature_window=args.dead_feature_window,
        dead_feature_threshold=args.dead_feature_threshold,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_log_frequency=args.wandb_log_frequency,
        device=args.device,
        seed=args.seed,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        dtype=torch.float32,
    )

    print("Loading dataset")
    classnames = get_classnames(args.dataset)
    dataset = load_dataset(**DATASET_INFO[args.dataset])
    num_proc = multiprocessing.cpu_count()
    print(f"Using {num_proc} processes for mapping.")
    if args.dataset in ['custom_data']:
        def transform_function(example):
            # This function will be called on each example when it is accessed.
            try:
                example["image"] = [Image.open(img).convert("RGB") for img in example["image"]]
            except Exception as e:
                # Optionally handle errors, e.g., mark the example as invalid or return None.
                print(f"Error processing {example['image']}: {e}")
                example["image"] = [None]
            try:
                # print(example["label"])
                example["label"] = [int(label) for label in example["label"]]
            except Exception as e:
                print(f"Error processing {example['label']}: {e}")
                example["label"] = [None]
            return example 
        dataset.set_transform(transform_function)
        
    dataset = dataset.shuffle(seed=args.seed)
    print("Loading SAE and ViT models")
    sae = SparseAutoencoder(cfg, args.device)

    vit = load_hooked_vit(
        cfg,
        args.vit_type,
        args.model_name,
        args.device,
        args.model_path,
        args.config_path,
        classnames,
    )

    print("Initializing ViTActivationsStore")
    activation_store = ViTActivationsStore(
        dataset,
        args.batch_size,
        args.device,
        args.seed,
        vit,
        args.block_layer,
        cfg.module_name,
        args.class_token,
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=sae.cfg.lr)
    scheduler = get_scheduler(args.lr_scheduler_name, optimizer=optimizer)

    print("Initializing SAE b_dec using activation_store")
    sae.initialize_b_dec(activation_store)
    sae.train()

    if cfg.log_to_wandb:
        print('wandb logged in')
        wandb.init(project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    sae_trainer = SAETrainer(sae, vit, activation_store, cfg, optimizer, scheduler, args.device)
    sae_trainer.fit()
