# Regional Attention

Cross-attend to different prompt based on region of image

![](input/blob-bigger-0.png)
![](input/blob-bigger-1.png)
![](input/blob-bigger-2.1.png)
> - illustration of ice dragon fighting in a fantasy battleground  
> - illustration of fire dragon fighting in a fantasy battleground  
> - illustration of sakura tree in full bloom, petals falling, spring day, calm sky, rolling hills, grass, flowers masterpiece, dramatic, highly detailed, high dynamic range  

![00075_base_illustration of dragons fighting in a fantasy battleground_36](https://github.com/user-attachments/assets/7b846a09-4f7d-43d9-a349-7edde2f1f048)

Generated via [`regional_attn.py`, at commit `bddf6de`](https://github.com/Birch-san/regional-attn/blob/bddf6def6b2744976f102c84beb07c86aefaac70/script/regional_attn.py)

## Setup

Linux/CUDA:

```bash
python3 -m venv venv
. venv/bin/activate
pip install wheel
pip install -r requirements.txt
# optional dep for playing around with high-pass noise interpolation
pip install dctorch --no-deps
```

## Run

_Ensure virtualenv has been activated._

```bash
python -m script.play
```