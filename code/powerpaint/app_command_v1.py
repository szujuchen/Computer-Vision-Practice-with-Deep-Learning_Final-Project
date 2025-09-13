import argparse
import os
import random
import cv2

import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation

from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel

import sys
from os.path import dirname, abspath
# Add the directory containing `subtask1` to sys.path
current_dir = dirname(abspath(__file__))  # Directory of helper.py
sys.path.append(current_dir)

from powerpaint.models.brushnet import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_powerpaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_powerpaint_brushnet import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.pipelines.pipeline_powerpaint_controlnet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_grad_enabled(False)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def select_tab_text_guided():
    return "text-guided"


def select_tab_object_removal():
    return "object-removal"


def select_tab_image_outpainting():
    return "image-outpainting"


def select_tab_shape_guided():
    return "shape-guided"


class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version) -> None:
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only

        # ppt-v1
        self.pipe = Pipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                torch_dtype=weight_dtype, 
                local_files_only=local_files_only, 
            )
        self.pipe.tokenizer = TokenizerWrapper(
            from_pretrained="runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer",
            revision=None,
            local_files_only=local_files_only,
        )

        # add learned task tokens into the tokenizer
        add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )

        # loading pre-trained weights
        load_model(self.pipe.unet, os.path.join(checkpoint_dir, "unet/unet.safetensors"), strict=False)
        load_model(self.pipe.text_encoder, os.path.join(checkpoint_dir, "text_encoder/text_encoder.safetensors"), strict=False)
        self.pipe = self.pipe.to("cuda")

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def load_controlnet(self, control_type):
        if self.current_control != control_type:
            if control_type == "canny" or control_type is None:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", 
                    torch_dtype=weight_dtype, 
                    local_files_only=self.local_files_only,
                )
            elif control_type == "pose":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=weight_dtype,
                    local_files_only=self.local_files_only,
                )
            elif control_type == "depth":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth", 
                    torch_dtype=weight_dtype, 
                    local_files_only=self.local_files_only,
                )
            else:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-hed", 
                    torch_dtype=weight_dtype, 
                    local_files_only=self.local_files_only,
                )
            self.control_pipe = self.control_pipe.to("cuda")
            self.current_control = control_type

    def predict(
        self,
        input_image,
        prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
    ):
        size1, size2 = input_image["image"].convert("RGB").size

        if task != "image-outpainting":
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        else:
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))

        if vertical_expansion_ratio is not None and horizontal_expansion_ratio is not None:
            o_W, o_H = input_image["image"].convert("RGB").size
            c_W = int(horizontal_expansion_ratio * o_W)
            c_H = int(vertical_expansion_ratio * o_H)

            expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
            original_img = np.array(input_image["image"])
            expand_img[
                int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                :,
            ] = original_img

            blurry_gap = 10

            expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
            if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                    :,
                ] = 0

            input_image["image"] = Image.fromarray(expand_img)
            input_image["mask"] = Image.fromarray(expand_mask)

        if self.version != "ppt-v1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print("Prompt A: ", promptA)
        print("Prompt B: ", promptB)
        print("Negative Prompt A: ", negative_promptA)
        print("Negative Prompt B: ", negative_promptB)


        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))
        set_seed(seed)
        
        # ppt-v1
        result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                width=H,
                height=W,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
            ).images[0]

        mask_np = np.array(input_image["mask"].convert("RGB"))
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )
        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        dict_res = [input_image["mask"].convert("RGB"), result_m]

        result_paste = Image.fromarray(np.uint8(ours_np * 255))
        dict_out = [input_image["image"].convert("RGB"), result_paste]
        # dict_out = [result]
        return dict_out, dict_res


    def infer(
        self,
        input_image,
        text_guided_prompt,
        text_guided_negative_prompt,
        shape_guided_prompt,
        shape_guided_negative_prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        outpaint_prompt,
        outpaint_negative_prompt,
        removal_prompt,
        removal_negative_prompt,
    ):
        print("Task: ", task)
        if task == "text-guided":
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt
        elif task == "shape-guided":
            prompt = shape_guided_prompt
            negative_prompt = shape_guided_negative_prompt
        elif task == "object-removal":
            
            prompt = removal_prompt
            negative_prompt = removal_negative_prompt
        elif task == "image-outpainting":
            prompt = outpaint_prompt
            negative_prompt = outpaint_negative_prompt
            return self.predict(
                input_image,
                prompt,
                fitting_degree,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                task,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
            )
        else:
            task = "text-guided"
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt

        
        return self.predict(
            input_image, prompt, fitting_degree, ddim_steps, scale, seed, negative_prompt, task, None, None
        )

def test_powerpaint(args, mask_arr=None):
    assert args.input_image is not None
    assert args.output_folder is not None
    assert mask_arr or args.mask_image != ""

    set_seed(args.seed_paint)

    weight_dtype = torch.float16
    checkpoint_dir = os.path.join(current_dir, "checkpoints/ppt-v1")
    local_files_only = False
    version = "ppt-v1"

    input_dict = {
        "image": Image.open(args.input_image),
        "mask": Image.open(args.mask_image) if mask_arr is None else mask_arr
    }

    print("Image size:", input_dict["image"].size)
    print("mask size:", input_dict["mask"].size)

    controller = PowerPaintController(weight_dtype, checkpoint_dir, local_files_only, version)

    result, mask_result = controller.infer(
        input_image=input_dict,
        text_guided_prompt=args.prompt,
        text_guided_negative_prompt=args.negative_prompt,

        shape_guided_prompt="",
        shape_guided_negative_prompt="",

        fitting_degree=args.fitting_degree,
        ddim_steps=args.ddim_steps,
        scale=args.scale,
        seed=args.seed_paint,
        task=args.task,

        vertical_expansion_ratio=args.vertical_expansion_ratio,
        horizontal_expansion_ratio=args.horizontal_expansion_ratio,

        outpaint_prompt="",
        outpaint_negative_prompt="",
        removal_prompt="",
        removal_negative_prompt=""
    )

    # os.makedirs(args.output_folder, exist_ok=True)
    # result[0].save(args.output_folder+"_result_v1.jpg")
    result[1].save(os.path.join(args.output_folder, f"{args.input_image.split('/')[-1].split('.')[0]}_result.jpg"))
    # mask_result[1].save(os.path.join(args.output_folder, f"{args.input_image.split('/')[-1].split('.')[0]}_mask.jpg"))
    return os.path.join(args.output_folder, f"{args.input_image.split('/')[-1].split('.')[0]}_result.jpg")

def get_powerpaint_args_parser():
    parser = argparse.ArgumentParser('Set powerpaint', add_help=False)
    parser.add_argument("--input_image", type=str, help="input image path")
    parser.add_argument("--mask_image", type=str, default="", help="mask image path")
    parser.add_argument("--prompt", default="", help="prompt")
    parser.add_argument("--negative_prompt", default="", help="negative prompt")
    parser.add_argument("--task", type=str, default="text-guided", help="task")
    parser.add_argument("--fitting_degree", type=float, default=0.95, help="fitting degree") # tradeoff
    parser.add_argument("--ddim_steps", type=int, default=50, help="ddim steps") # num_inference_steps
    parser.add_argument("--scale", type=float, default=8.5, help="scale") # guiden level
    parser.add_argument("--seed_paint", type=int, default=12873, help="seed")
    parser.add_argument("--vertical_expansion_ratio", type=float, default=0.0, help="vertical expansion ratio")
    parser.add_argument("--horizontal_expansion_ratio", type=float, default=0.0, help="horizontal expansion ratio")
    parser.add_argument("--output_folder", type=str, help="output folder")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Powerpaint inference script', parents=[get_powerpaint_args_parser()])
    args = parser.parse_args()
    test_powerpaint(args)