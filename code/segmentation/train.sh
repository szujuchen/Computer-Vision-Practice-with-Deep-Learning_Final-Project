export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export OMP_NUM_THREADS=8

# please change your output_dir and tensorboard according to the dataset name
# if you want to resume training, please set the resume path

# for unc dataset
# detr_model = detr-r101-unc.pth
# dataset = unc

# for unc+ dataset
# detr_model = detr-r101-unc.pth
# dataset = unc+

# for gref dataset
# detr_model = detr-r101-gref.pth
# dataset = gref

# for gref_umd dataset
# detr_model = detr-r101-gref.pth
# dataset = gref_umd

# for referit dataset
# detr_model = detr-r101-referit.pth
# dataset = referit

# The following is an example of training on the unc dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port=10033 --use_env train.py \
--epochs 175 --batch_size 8 --aug_scale --aug_translate --aug_crop --backbone resnet101 \
--detr_model ./checkpoints/detr-r101-referit.pth \
--lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --num_workers 16 \
--dataset customized --max_query_len 40 --freeze_epochs 10 --lr_drop 60 --clip_max_norm 0.1 \
--data_root ./data/coco \
--output_dir ./output/refer_coco_nop/ \
--tensorboard ./output/refer_coco_nop/tensorboard/ \
--resume ./checkpoints/referit/best_checkpoint.pth