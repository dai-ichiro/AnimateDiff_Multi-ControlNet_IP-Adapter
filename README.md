## Environment
~~~
Windows 11
CUDA 11.8
Python 3.12
~~~
## Install
~~~
pip install -r requirements.txt
~~~
## Video
"walk.mp4" was downloaded from [here](https://pixabay.com/ja/videos/%E5%A5%B3%E6%80%A7-%E3%83%A2%E3%83%87%E3%83%AB-%E6%A9%8B%E8%84%9A-%E6%B5%B7-85303/)

## How to make ControlNet images
"openpose.gif" was created by "preprocess_typer.py".

~~~
python preprocess_typer.py --video walk.mp4 --type openpose --gif
~~~

## How to use
~~~
python run.py --config settings.yaml
~~~

## Link to my blog
https://touch-sp.hatenablog.com/entry/2023/12/13/183401
