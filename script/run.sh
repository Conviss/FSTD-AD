python main.py --dataset PSM --epoch 64
python main.py --dataset SMD --p 1
python main.py --dataset SWaT --window_size 32 --step_size 8 --window_length 125 --kernel_size 8 --stride 4 --hidden_dim 12
python main.py --dataset MSL --epoch 64 --p 1
python main.py --dataset SMAP --epoch 64 --p 1 --each_entity True
