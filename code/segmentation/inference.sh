export CUDA_VISIBLE_DEVICES=3
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export OMP_NUM_THREADS=4

# please change the dataset name, eval_set and the eval_model path
# the following is an example of evaluating on the gref_umd test dataset
python -m torch.distributed.launch --nproc_per_node=1 --master_port=10030 --use_env inference.py \
--num_workers 4 \
--bert_enc_num 12 --detr_enc_num 6 --backbone resnet101 --detr_model ./checkpoints/detr-r101-referit.pth \
--img_path /tmp2/b10705005/cvpdlfinal/1.png \
--query "chair" \
--max_query_len 40 \
--eval_model ./checkpoints/referit/best_checkpoint.pth
