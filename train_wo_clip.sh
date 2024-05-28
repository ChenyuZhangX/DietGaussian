export CUDA_VISIBLE_DEVICES=2
python train.py -s './data/toy/undistort' -m 'outputs/toy_wo_clip_r4_ep10w' --port 6010 -r 4 --iterations 100000