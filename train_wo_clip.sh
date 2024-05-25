export CUDA_VISIBLE_DEVICES=3
python train.py -s './data/toy/undistort' -m 'outputs/toy_wo_clip' --port 6009 -r 8 --iterations 30000