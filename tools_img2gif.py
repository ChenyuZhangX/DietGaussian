import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
from torchvision.utils import save_image
from PIL import Image
import os

path = 'outputs/toy_wo_clip_r4_ep10w/train/ours_100000/renders'

ranges = set(range(38, 90))
to_del = set(range(73, 80))
ranges = ranges.difference(to_del)

print(ranges)


names = [f"{idx:05d}.png" for idx in ranges]
trans = Compose([
    ToTensor(),
])
imgs = [trans(Image.open(os.path.join(path, name)).convert('RGB')) for name in names]



imgs = [ToPILImage()(img) for img in imgs]
imgs[0].save('workspace/ours.gif', save_all=True, append_images=imgs[1:], duration=100, loop=0)