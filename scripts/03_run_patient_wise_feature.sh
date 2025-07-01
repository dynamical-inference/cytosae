PYTHONPATH=./ nohup python -u tasks/compute_patient_wise_sae_activation.py \
    --root_dir ./ \
    --dataset_name hehr \
    --sae_path out/checkpoints/8jsxk3co/final_sparse_autoencoder_dinov2_vitb14_-2_resid_49152.pt \
    --vit_type custom \
    --threshold 0.2 \
    > logs/03_class_wise-hehr.txt