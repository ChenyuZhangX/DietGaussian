import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
from torchvision.utils import save_image
from PIL import Image
import os

path = 'outputs/toy_wo_clip_r4_ep10w/train/ours_100000/renders'
names = [f"{idx:05d}.png" for idx in range(38, 100)]
imgs = [Image.open(os.path.join(path, name)) for name in names]
imgs[0].save('ours4.gif', save_all=True, append_images=imgs[1:], duration=300, loop=0)