# Diet Gaussian


## New Codes

```bash
train_wo_clip.sh # original training script

train.sh # training script with clip loss

convert.py # convert the data to colmap format

dataset_colmap.py # simple dataset transformation version

ablation_clip.ipynb # ablation study for clip model

SAM_automatic_mask_generator_example.ipynb # example of automatic mask generation

tools_img2gif.py # convert images to gif
```

## Requirements

### Install CLIP Model

```bash
cd submodules/CLIP
pip install -e .
```