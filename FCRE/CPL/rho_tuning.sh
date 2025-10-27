for t in Tacred
do
    for r in 0.05 0.08 0.1 0.12
    do
        CUDA_VISIBLE_DEVICES=0 python train.py --task_name $t \
            --num_k 5 \
            --num_gen 5 \
            --base_optimizer AdamW \
            --decay 0.01 \
            --mixup \
            --SAM \
            --sam_optimizer ASAM \
            --rho $r \
            --batch-size 16
    done
done