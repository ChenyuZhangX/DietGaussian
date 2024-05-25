import torch
import clip
from PIL import Image

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(preprocess)


Processor = Compose([
    Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None), 
    CenterCrop(size=(224, 224)), 
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ]
)

image = Image.open("submodules/CLIP/CLIP.png").convert("RGB")
# as tensor
image = ToTensor()(image)
image = Processor(image).to(device).unsqueeze(0)

print(image.max(), image.min())

prod = preprocess(Image.open("submodules/CLIP/CLIP.png")).unsqueeze(0).to(device)
print(prod.shape)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)


with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)