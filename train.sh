export CUDA_VISIBLE_DEVICES=0
python train.py -s './data/toy/undistort' -m 'outputs/toy_clip_af3k_bothloss' -r 8 --port 6011 --with_clip --iterations 100000