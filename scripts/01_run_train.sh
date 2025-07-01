PYTHONPATH=./ nohup python -u tasks/train_sae_vit.py \
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
--log_to_wandb --wandb_project cytosae --wandb_entity m-f-dasdelen-helmholtz-munich \
> logs/01_test_training-mll23.txt