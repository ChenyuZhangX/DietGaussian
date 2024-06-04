export CUDA_VISIBLE_DEVICES=1
# python train.py -s './data/toy/undistort' -m 'outputs/toy_clip_r4_af10w_ep20w_0.1' -r 4 --port 6022 --with_clip --iterations 200000
 python train.py -s './data/toy/undistort' -m 'debug' -r 4 --port 6021 --with_clip --iterations 200000