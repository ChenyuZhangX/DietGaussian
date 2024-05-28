export CUDA_VISIBLE_DEVICES=1
python train.py -s './data/toy/undistort' -m 'outputs/toy_clip_r4_fbeg_ep3w_0.1' -r 4 --port 6022 --with_clip --iterations 30000