CUDA_VISIBLE_DEVICES="3" \
    python -m torch.distributed.launch \
    --nproc_per_node 4 example/train.py \
    --config_file example/expriments/vit.yaml \
    --gpus 1
