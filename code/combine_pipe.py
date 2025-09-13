import argparse
import os
import torch
import json

from segmentation.inference import test_seg, get_seg_args_parser
from powerpaint.app_command_v1 import test_powerpaint, get_powerpaint_args_parser

parser = argparse.ArgumentParser("combined pipeline", parents=[get_seg_args_parser(), get_powerpaint_args_parser()])
args = parser.parse_args()

output_dir = f"./results/OUTPUT/nearest-exact/10"
mask_dir = f"./results/MASK/nearest-exact/10"
SEG_MODEL = f"/tmp2/b10705005/cvpdlfinal/SegVG/output/refer_coco/best_checkpoint.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
# mask_dir = os.path.join(output_dir, "mask")

os.environ['HF_HOME'] = '/tmp2/b10705005/cache/hf'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(mask_dir):
    os.mkdir(mask_dir)

seg_args = args
paint_args = args

with open("results/data_20.json", "r") as f:
    tests = json.load(f)

for test in tests:
    seg_args.img_path = os.path.join("results/INPUT", test["image"])
    seg_args.query = test["remove_prompt"]
    seg_args.eval_model = SEG_MODEL
    seg_args.output_dir = mask_dir
    mask = test_seg(seg_args)

    paint_args.input_image = os.path.join("results/INPUT", test["image"])
    paint_args.output_folder = output_dir
    paint_args.prompt = test["new_object_prompt"]
    outfile = test_powerpaint(paint_args, mask)
    print("result at: ", outfile)