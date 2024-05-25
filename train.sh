export CUDA_VISIBLE_DEVICES=2
python train.py -s './data/toy/undistort' -m 'outputs/toy_clip_af10w_bothloss_0.02_neighbour' -r 1 --port 6013 --with_clip --iterations 100000