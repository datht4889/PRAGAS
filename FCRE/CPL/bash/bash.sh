for t in FewRel Tacred
do
    CUDA_VISIBLE_DEVICES=1 TOKENIZER_PARALELISM=True python train.py --task_name $t \
        --model bge \
        --output-size 768 \
        --max-length 256 \
        --num_k 5 \
        --num_gen 5 \
        --base_optimizer AdamW \
        --decay 0.01 \
        --mixup \
        --SAM \
        --sam_optimizer ASAM \
        --rho 0.1 \
        --rho_weight 6 \
        --distill \
        --distill_type RKD \
        --distill_loss_weight 0 \
        --distill_top_k 10 \
        --batch-size 16  
done