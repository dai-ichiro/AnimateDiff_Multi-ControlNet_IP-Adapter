import torch
from diffusers import DiffusionPipeline, ControlNetModel, MotionAdapter
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image
import os
import argparse
import yaml
import datetime
import shutil

def gif2images(gif_filename):
    gif=Image.open(gif_filename)
    frames=[]
    for i in range(gif.n_frames):
        gif.seek(i)
        img = gif.copy()
        frames.append(img)
    return frames

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    required=True,
    help="path to yaml file"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config_dict = yaml.load(f, Loader=yaml.SafeLoader)

time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(os.path.join("outputs", time_str), exist_ok=False)
shutil.copyfile(args.config, os.path.join("outputs",time_str, "config.yaml"))

adapter = MotionAdapter.from_pretrained(
    config_dict["motion_module_path"]
)

controlnet_list = config_dict["controlnet"]

controlnet = MultiControlNetModel(
    [ 
        ControlNetModel.from_pretrained(
            x["model_path"],
            torch_dtype=torch.float16
        ) 
        for x in controlnet_list
    ]
)

controlimage = [gif2images(x["image_path"]) for x in controlnet_list]
n_frames = 32 if min([len(x) for x in controlimage])>32 else min([len(x) for x in controlimage])
controlimage = [x[0:n_frames] for x in controlimage]

controlnet_conditioning_scale = [x["conditioning_scale"] for x in controlnet_list]

if config_dict["vae"]["enable"]:
    from diffusers import AutoencoderKL
    if config_dict["vae"]["single_file"]:
        vae = AutoencoderKL.from_single_file(
            config_dict["vae"]["model_path"],
            torch_dtype=torch.float16
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            config_dict["vae"]["model_path"],
            torch_dtype=torch.float16
        )

model_id = config_dict["pretrained_model_path"]
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    controlnet=controlnet,
    custom_pipeline="custom-pipeline/pipeline_animatediff_controlnet.py",
    torch_dtype=torch.float16
)

if config_dict["vae"]["enable"]:
    if config_dict["vae"]["single_file"]:
        pipe.vae = AutoencoderKL.from_single_file(
            config_dict["vae"]["model_path"],
            torch_dtype=torch.float16
        )
    else:
        pipe.vae = AutoencoderKL.from_pretrained(
            config_dict["vae"]["model_path"],
            torch_dtype=torch.float16
        )

pipe.to("cuda")

use_ipadapter = config_dict["ip_adapter"]["enable"]
if use_ipadapter:
    pipe.load_ip_adapter(
        config_dict["ip_adapter"]["folder"],
        subfolder=config_dict["ip_adapter"]["subfolder"],
        weight_name=config_dict["ip_adapter"]["weight_name"],
        torch_dtype=torch.float16
    )
    pipe.set_ip_adapter_scale(config_dict["ip_adapter"]["scale"])
    
use_lcmlora = config_dict["lcm_lora"]["enable"]
if use_lcmlora:
    from diffusers import LCMScheduler
    pipe.scheduler = LCMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear"
    )
    pipe.load_lora_weights(config_dict["lcm_lora"]["model_path"], adapter_name="lcm")
    pipe.set_adapters(["lcm"], adapter_weights=[config_dict["lcm_lora"]["weight"]])
else:
    from diffusers import DDIMScheduler
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
    )

pipe.enable_vae_slicing()

if config_dict["freeu"]["enable"]:
    pipe.enable_freeu(
        s1=config_dict["freeu"]["s1"],
        s2=config_dict["freeu"]["s2"],
        b1=config_dict["freeu"]["b1"],
        b2=config_dict["freeu"]["b2"]
    )

prompt = config_dict["prompt"]
negative_prompt = config_dict["negative_prompt"]
seed = config_dict["seed"]
steps = config_dict["steps"]
guidance_scale = 1.0 if use_lcmlora else config_dict["guidance_scale"]
width = config_dict["width"]
height = config_dict["height"]
clip_skip = config_dict["clip_skip"] if isinstance(config_dict["clip_skip"], int) else None

if use_ipadapter:
    ip_image = Image.open(config_dict["ip_adapter"]["image_path"])
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=ip_image,
        num_frames=n_frames,
        width=width,
        height=height,
        conditioning_frames=controlimage,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(seed),
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        clip_skip=clip_skip
    ).frames[0]
else:
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=n_frames,
        width=width,
        height=height,
        conditioning_frames=controlimage,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(seed),
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        clip_skip=clip_skip
    ).frames[0]

from diffusers.utils import export_to_gif
export_to_gif(result, os.path.join("outputs", time_str, "result.gif"))
