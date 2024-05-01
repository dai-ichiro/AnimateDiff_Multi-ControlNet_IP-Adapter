import cv2              
from PIL import Image   
from pathlib import Path
from controlnet_aux.processor import Processor
from typer import Typer, Option

app = Typer()

@app.command()
def main(
    video: str=Option(..., help="path to video file"),
    processor_type: str=Option(..., "--type", help="preprocess type"),
    gif: bool=Option(default=False, help="gif or not")
):
    if processor_type == "tile":
        cap = cv2.VideoCapture(video)
        post_images = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            post_images.append(img)
    else:
        processor = Processor(processor_type)      
        cap = cv2.VideoCapture(video)
        post_images = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            result = processor(img, to_pil=True)
            post_images.append(result)
    
    if gif:
        from diffusers.utils import export_to_gif
        export_to_gif(post_images, f"{processor_type}.gif")
    else:
        Path(processor_type).mkdir(exist_ok=False)
        for i, image in enumerate(post_images):
            image.save(Path(processor_type, f"{i}.png").as_posix())

if __name__ == "__main__":
    app()