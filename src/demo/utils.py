from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)


def load_sae_tester(sae_path):
    datasets = load_datasets()
    classnames = get_all_classnames(datasets, data_root="./configs/classnames")

    root = "./out/feature_data"
    sae_runname = "sae_8jsxk3co"
    vit_name = "custom"

    max_act_imgs, mean_acts = get_max_acts_and_images(
        datasets, root, sae_runname, vit_name
    )

    sae_tester = {}

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="custom",
        device="cpu",
        backbone="dinov2_vitb14",
        model_path="./DinoBloom-B.pth",
        classnames=classnames["wbc"],
    )
    sae_clip = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="custom",
        device="cpu",
        model_path="./DinoBloom-B.pth",
        config_path=None,
        backbone="dinov2_vitb14",
        classnames=classnames["wbc"],
    )
    sae_maple = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)
    sae_tester["CLIP"] = sae_clip
    sae_tester["MaPLE-imagenet"] = sae_maple
    return sae_tester
