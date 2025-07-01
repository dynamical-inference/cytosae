## Table of Contents
- [Train SAE](#train-sae)
- [Extract and save SAE latent data](#extract-and-save-sae-latent-data)
- [Compute patient-level SAE latents](#compute-patient-level-sae-latents)


## Train SAE
**Requirements**
* DinoBloom-B checkpoint (place it inside the root folder)
* Training data (only images)

**Run**
- We used following configurations:
```bash
PYTHONPATH=./ python -u tasks/train_sae_vit.py \
--model_name dinov2_vitb14 \
--model_path ./DinoBloom-B.pth \
--seed 42 \
--expansion_factor 64 \
--b_dec_init_method geometric_median \
--batch_size 64 \
--n_checkpoints 20 \
--l1_coefficient 0.00008 \
--lr 0.0004 \
--block_layer -2 \
--dataset mll23 \
--use_ghost_grads \
--total_training_tokens 1000000 \
--log_to_wandb --wandb_project cytosae --wandb_entity m-f-dasdelen-helmholtz-munich
```

**Outputs**
* SAE checkpoint (.pt)

**Analysis**
- Training log can be found [here](https://api.wandb.ai/links/m-f-dasdelen-helmholtz-munich/jpqzn9xg)


## Extract and save SAE latent data
**Requirements**
* DinoBloom-B checkpoint
* SAE checkpoint
* Dataset (can be different from training dataset)
  
**Run**
```bash
PYTHONPATH=./ python -u tasks/compute_sae_feature_data.py \
    --root_dir ./ \
    --dataset_name {dataset_name} \
    --sae_path {PATH/TO/SAE_CKPT}.pt \
    --vit_type custom \
    --model_path ./DinoBloom-B.pth
```

**Outputs** 

Under `{root_dir}/out/feature_data/{checkpoint_name}/{vit_type}/{dataset_name}/`
* `max_activating_image_indices.pt`
* `max_activating_image_label_indices.pt`
* `max_activating_image_values.pt`
* `sae_mean_acts.pt`
* `sae_sparsity.pt`

**Analyze the output**
- See [analysis.ipynb](../analysis/analysis.ipynb) for analysis

## Compute patient-level SAE latents
**Requirements**
* DinoBloom-B checkpoint
* SAE checkpoint
* SAE feature data (latent data)
* Dataset (to compute patient-level SAE latents; should be SAME with dataset used in [Extract and save SAE latent data](#extract-and-save-sae-latent-data))

**Run**
```bash
PYTHONPATH=./ python tasks/compute_patient_wise_sae_activation.py \
    --root_dir ./ \
    --dataset_name hehr \
    --threshold 0.2 \
    --sae_path ${PATH/TO/SAE_CKPT}.pt \
    --vit_type custom
```

**Outputs**
* `cls_sae_cnt.npy` size of `(num_of_patients, num_sae_latents)`

**Patient-wise analysis**
- See [patient_analysis.ipynb](../analysis/patient_analysis.ipynb) for analysis
