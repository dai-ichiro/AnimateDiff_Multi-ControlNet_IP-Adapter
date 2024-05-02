import torch
from diffusers import DiffusionPipeline, AutoencoderKL, ControlNetModel, MotionAdapter
from diffusers.utils import export_to_gif
from PIL import Image
import yaml
import datetime
import shutil
from typer import Typer, Option
from pathlib import Path

def gif2images(gif_filename):
    gif=Image.open(gif_filename)
    frames=[]
    for i in range(gif.n_frames):
        gif.seek(i)
        img = gif.copy()
        frames.append(img)
    return frames

app = Typer()

@app.command()
def load_yaml(
    config: str=Option(..., help="config file")
):
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    Path("outputs", time_str).mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config, Path("outputs",time_str, "config.yaml").as_posix())

    result = main(**config_dict)

    export_to_gif(result, Path("outputs", time_str, "result.gif").as_posix())

def main(
    pretrained_model_path,
    vae,
    motion_module_path,
    controlnet_list,
    prompt, negative_prompt,
    seed,
    steps,
    guidance_scale,
    width, height,
    clip_skip,
    ip_adapter,
    lcm_lora,
    freeu
):
    adapter = MotionAdapter.from_pretrained(motion_module_path)

    controlnet = [
        ControlNetModel.from_pretrained(
            x["model_path"],
            torch_dtype=torch.float16
        ) 
        for x in controlnet_list
    ]

    controlimage = [gif2images(x["image_path"]) for x in controlnet_list]
    n_frames = 32 if min([len(x) for x in controlimage])>32 else min([len(x) for x in controlimage])
    controlimage = [x[0:n_frames] for x in controlimage]

    controlnet_conditioning_scale = [x["conditioning_scale"] for x in controlnet_list]

    pipe = DiffusionPipeline.from_pretrained(
        pretrained_model_path,
        motion_adapter=adapter,
        controlnet=controlnet,
        custom_pipeline="pipeline_animatediff_controlnet",
        torch_dtype=torch.float16
    )
    
    if vae["enable"]:
        if vae["single_file"]:
            _vae = AutoencoderKL.from_single_file(
                vae["model_path"],
                torch_dtype=torch.float16
            )
        else:
            _vae = AutoencoderKL.from_pretrained(
                vae["model_path"],
                torch_dtype=torch.float16
            )
        pipe.vae = _vae

    use_ipadapter = ip_adapter["enable"]
    if use_ipadapter:
        pipe.load_ip_adapter(
            ip_adapter["folder"],
            subfolder=ip_adapter["subfolder"],
            weight_name=ip_adapter["weight_name"],
            torch_dtype=torch.float16
        )
        pipe.set_ip_adapter_scale(ip_adapter["scale"])
    
    use_lcmlora = lcm_lora["enable"]
    if use_lcmlora:
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear"
        )
        pipe.load_lora_weights(lcm_lora["model_path"], adapter_name="lcm")
        pipe.set_adapters(["lcm"], adapter_weights=[lcm_lora["weight"]])
    else:
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear"
        )

    pipe.to("cuda")

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    if freeu["enable"]:
        pipe.enable_freeu(
            s1 = freeu["s1"],
            s2 = freeu["s2"],
            b1 = freeu["b1"],
            b2 = freeu["b2"]
        )

    guidance_scale = 1.0 if use_lcmlora else guidance_scale
    clip_skip = clip_skip if isinstance(clip_skip, int) else None

    if use_ipadapter:
        ip_image = Image.open(ip_adapter["image_path"])
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

    return result

if __name__=="__main__":
    app()
    
    
