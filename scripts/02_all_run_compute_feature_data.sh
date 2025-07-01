DATASETS=("mll23" "acevedo" "bmc" "matek" "hehr")  # Add all dataset names here

for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    PYTHONPATH=./ nohup python -u tasks/compute_sae_feature_data.py \
        --root_dir ./ \
        --dataset_name "$DATASET" \
        --sae_path out/checkpoints/8jsxk3co/final_sparse_autoencoder_dinov2_vitb14_-2_resid_49152.pt \
        --vit_type custom \
        > logs/extract_"$DATASET".txt

    echo "Finished processing dataset: $DATASET"
done