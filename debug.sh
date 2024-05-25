export CUDA_VISIBLE_DEVICES=1
python debug.py -s './data/toy/undistort' -m 'outputs/debug' -r 8 --port 6001 --with_clip --iterations 30000