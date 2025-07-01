import json
import os
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple
from PIL import Image

import torch
import torchvision.transforms as T
from datasets import Dataset, load_dataset

from src.models.utils import get_adapted_clip, get_base_clip
from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# Dataset configurations
DATASET_INFO = {
    "mll23": {
        "path": "dataset/mll23",
        "split": "train",
    },
    "matek": {
        "path": "dataset/PKG - AML-Cytomorphology_LMU",
        "split": "train",
    },
    "acevedo": {
        "path": "dataset/acevedo",
        "split": "train",
    },
    "hehr": {
        "path": "dataset/PKG - AML-Cytomorphology_MLL_Helmholtz_v1",
        "split": "train",
    },
    "bmc": {
        "path": "dataset/bmc", 
        "split": "train",
    },
}

SAE_DIM = 49152 # 768*64 [vit_dimension * expansion_factor]


def load_sae(sae_path: str, device: str) -> tuple[SparseAutoencoder, Config]:
    """Load a sparse autoencoder model from a checkpoint file."""
    checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)

    try:
        cfg = Config(checkpoint["cfg"])
    except:
        cfg = Config(checkpoint["config"])
    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval().to(device)

    return sae, cfg


def load_hooked_vit(
    cfg: Config,
    vit_type: str,
    backbone: str,
    device: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> HookedVisionTransformer:
    """Load a vision transformer model with hooks."""
    if vit_type == "base":
        model, processor = get_base_clip(backbone)
    elif vit_type == "custom":
        model = get_dino_bloom(
            modelpath=model_path, 
            modelname=backbone    
        )

        custom_transforms = T.Compose([
            T.Resize((cfg.image_height, cfg.image_width)),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

        dummy = type("Dummy", (), {})()
        dummy.image_std = [0.229, 0.224, 0.225]
        dummy.image_mean = [0.485, 0.456, 0.406]
        custom_transforms.image_processor = dummy

        processor = custom_transforms
    else:
        model, processor = get_adapted_clip(
            cfg, vit_type, model_path, config_path, backbone, classnames
        )

    return HookedVisionTransformer(model, processor, device=device)

def get_dino_bloom(modelpath="/content/dinobloom-s.pth",modelname="dinov2_vits14"):
    embed_sizes={"dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536}
    # load the original DINOv2 model with the correct architecture and parameters.
    model=torch.hub.load('facebookresearch/dinov2', modelname)
    # load finetuned weights
    pretrained = torch.load(modelpath, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    #corresponds to 224x224 image. patch size=14x14 => 16*16 patches
    pos_embed = torch.nn.Parameter(torch.zeros(1, 257, embed_sizes[modelname]))
    model.pos_embed = pos_embed

    model.load_state_dict(new_state_dict, strict=True)
    return model

def get_sae_and_vit(
    sae_path: str,
    vit_type: str,
    device: str,
    backbone: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> tuple[SparseAutoencoder, HookedVisionTransformer, Config]:
    """Load both SAE and ViT models."""
    sae, cfg = load_sae(sae_path, device)
    vit = load_hooked_vit(
        cfg, vit_type, backbone, device, model_path, config_path, classnames
    )
    return sae, vit, cfg


def load_and_organize_dataset(dataset_name: str) -> Tuple[list, Dict]:
    # TODO: ERR for imagenet (gets killed after 75%)
    """
        Load dataset and organize data by class. 
        Return classnames and data by class.
        Requried for classification_with_top_k_masking.py and compute_class_wise_sae_activation.py   
    """
    dataset = load_dataset(**DATASET_INFO[dataset_name])
    classnames = get_classnames(dataset_name, dataset)

    data_by_class = defaultdict(list)
    for data_item in tqdm(dataset):
        classname = classnames[data_item["label"]]
        data_by_class[classname].append(data_item)

    return classnames, data_by_class

def load_and_organize_dataset_csv(dataset_name: str) -> Tuple[list, Dict]:
    """
        Load dataset and organize data by class. 
        Return classnames and data by class.
        Requried for classification_with_top_k_masking.py and compute_class_wise_sae_activation.py   
    """
    dataset = load_dataset(**DATASET_INFO[dataset_name])

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
            example["label"] = [label for label in example["label"]]
        except Exception as e:
            print(f"Error processing {example['label']}: {e}")
            example["label"] = [None]
        return example 
    dataset.set_transform(transform_function)

    #classnames = get_classnames(dataset_name, dataset)
    classnames = []
    data_by_class = defaultdict(list)
    for data_item in tqdm(dataset):
        classname = data_item["label"]
        if classname not in classnames:
            classnames.append(classname) #can be used dict to make it more efficient
        data_by_class[classname].append(data_item)

    return classnames, data_by_class


def get_classnames(
    dataset_name: str, dataset: Dataset = None, data_root: str = "./configs/classnames"
) -> list[str]:
    """Get class names for a dataset."""
    filename = f"{data_root}/{dataset_name}_classnames"
    txt_filename = filename + ".txt"
    json_filename = filename + ".json"
    print(f'Class names loaded from {txt_filename}')

    if not os.path.exists(txt_filename) and not os.path.exists(json_filename):
        raise ValueError(f"Dataset {dataset_name} not supported")

    filename = json_filename if os.path.exists(json_filename) else txt_filename

    with open(filename, "r") as file:
        if dataset_name in ['acevedo', 'mll23', 'matek','hehr','bmc']:
            class_names = [line.strip() for line in file.readlines()]
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    return class_names


def setup_save_directory(
    root_dir: str, save_name: str, sae_path: str, vit_type: str, dataset_name: str
) -> str:
    """Set and create the save directory path."""
    sae_run_name = sae_path.split("/")[-2]
    save_directory = (
        f"{root_dir}/{save_name}/sae_{sae_run_name}/{vit_type}/{dataset_name}"
    )
    os.makedirs(save_directory, exist_ok=True)
    return save_directory


def get_sae_activations(
    model_activations: torch.Tensor, sae: SparseAutoencoder
) -> torch.Tensor:
    """Extract and process activations from the sparse autoencoder."""
    hook_name = "hook_hidden_post"

    # Run SAE forward pass and get activations from cache
    _, cache = sae.run_with_cache(model_activations)
    sae_activations = cache[hook_name]

    # Average across sequence length dimension if needed
    if len(sae_activations.size()) > 2:
        sae_activations = sae_activations.mean(dim=1)

    return sae_activations


def process_batch(vit, batch_data, device):
    """Process a single batch of images."""
    images = [data["image"] for data in batch_data]
    processed_images = []
    for img in images:
        # Convert image to RGB if it has an alpha channel or is not already in RGB mode.
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        processed_img = vit.processor(img)
        processed_images.append(processed_img)

    # Stack into a single batch of shape [batch_size, C, H, W]
    inputs = torch.stack(processed_images, dim=0).to(device)
    '''
    inputs = vit.processor(
        images=images, text="", return_tensors="pt", padding=True
    ).to(device)
    '''
    return inputs


def get_max_acts_and_images(
    datasets: dict, feat_data_root: str, sae_runname: str, vit_name: str
) -> tuple[dict, dict]:
    """Load and return maximum activations and mean activations for each dataset."""
    max_act_imgs = {}
    mean_acts = {}

    for dataset_name in datasets:
        # Load max activating image indices
        max_act_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "max_activating_image_indices.pt",
        )
        max_act_imgs[dataset_name] = torch.load(max_act_path, map_location="cpu").to(
            torch.int32
        )

        # Load mean activations
        mean_acts_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "sae_mean_acts.pt",
        )
        mean_acts[dataset_name] = torch.load(mean_acts_path, map_location="cpu").numpy()

    return max_act_imgs, mean_acts

'''
def load_datasets(seed: int = 1):
    """Load multiple datasets from HuggingFace."""
    return {
        "imagenet": load_dataset(
            "evanarlian/imagenet_1k_resized_256", split="train"
        ).shuffle(seed=seed),
        "imagenet-sketch": load_dataset(
            "clip-benchmark/wds_imagenet_sketch", split="test"
        ).shuffle(seed=seed),
        "caltech101": load_dataset(
            "HuggingFaceM4/Caltech-101",
            "with_background_category",
            split="train",
        ).shuffle(seed=seed),
    }
'''
def load_datasets(seed: int = 1):
    """Load multiple datasets from HuggingFace."""
    return {
        "mll23": load_dataset(**DATASET_INFO['mll23']).shuffle(seed=seed),
        "acevedo": load_dataset(**DATASET_INFO['acevedo']).shuffle(seed=seed),
        "matek": load_dataset(**DATASET_INFO['matek']).shuffle(seed=seed),
        "bmc": load_dataset(**DATASET_INFO['bmc']).shuffle(seed=seed),
        #"hehr": load_dataset(**DATASET_INFO['hehr']).shuffle(seed=seed),
    }


def get_all_classnames(datasets, data_root):
    """Get class names for all datasets."""
    class_names = {}
    for dataset_name, dataset in datasets.items():
        class_names[dataset_name] = get_classnames(dataset_name, dataset, data_root)
    return class_names
