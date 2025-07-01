PYTHONPATH=./ nohup python -u tasks/compute_sae_feature_data.py \
    --root_dir ./ \
    --dataset_name mll23 \
    --sae_path out/checkpoints/8jsxk3co/final_sparse_autoencoder_dinov2_vitb14_-2_resid_49152.pt \
    --vit_type custom > logs/02_test_extract_mll23.txt
