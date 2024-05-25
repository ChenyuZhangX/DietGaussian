export CUDA_VISIBLE_DEVICES=0
python train.py -s './data/toy/undistort' -m 'outputs/toy_clip_af3w_bothloss_0.5_neighbour' -r 4 --port 6010 --with_clip --iterations 30000