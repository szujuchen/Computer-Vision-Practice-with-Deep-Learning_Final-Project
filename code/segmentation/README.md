# Environment
看你用conda還是pip
conda 就用 req.txt 
pip 就用 requirements.txt
# inference
./inference.sh
可以動的變數 
1. img_path
2. query
3. eval_model: load checkpoints in `output/`\
    option: \
    base- checkpoints/referit/best_checkpoint.pth,\
    finetune refcoco- output/refer_coco/best_checkpoint.pth, \
    finetune our data- output/refer_customized/best_checkpoint.pth

mask 生出來會存在 `outputimg/`

# inference generate mask
那個model生出來的segmentation mask 是在 `engine.py #161`
