CUDA_VISIBLE_DEVICES="0,1,2,3" \
    python -m torch.distributed.launch \
    --nproc_per_node 4 example/train.py \
    --config_file example/expriments/vit.yaml \
    --gpus 4
    --data_path /home/dxm/dxm/fastmri_train    
