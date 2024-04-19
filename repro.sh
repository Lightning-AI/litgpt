
CUDA_VISIBLE_DEVICES=0, litgpt pretrain \
    --config tinyllama.yaml --devices 1 \
    --train.max_seq_length 512 \
    --use_full_mha True \
    --out_dir out/mha \
& \
CUDA_VISIBLE_DEVICES=1, \
litgpt pretrain \
    --config tinyllama.yaml --devices 1 \
    --train.max_seq_length 512 \
    --use_full_mha False \
    --out_dir out/gqa



# CUDA_VISIBLE_DEVICES=0, litgpt pretrain --config tinyllama.yaml --use_full_mha True --out_dir out/mha