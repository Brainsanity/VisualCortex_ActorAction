#!/bin/sh

# running now by Ruitao...
echo "Running net_77ep_8_2 => net_82ep_8_2 ..."
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=8 --speed=2 --num_epochs=5 --cont=1 --load_name=net_77ep_8_2 --save_name=net_82ep_8_2 &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_82ep_8_2 --nframes=8 --speed=2 --note=None  --data_list=val	2>&1 | tee -a param_R_2plus1_D_100ep/eval_82ep_8_2.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_82ep_8_2 --nframes=8 --speed=2 --note=None --data_list=train 	2>&1 | tee -a param_R_2plus1_D_100ep/eval_82ep_8_2.txt &&
echo "Finished net_77ep_8_2 => net_82ep_8_2"


echo "Running net_82ep_8_2 => net_87ep_16_1 ..."
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=5 --cont=1 --load_name=net_82ep_8_2 --save_name=net_87ep_16_1 &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_87ep_16_1 --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a param_R_2plus1_D_100ep/net_87ep_16_1.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_87ep_16_1 --nframes=16 --speed=1 --note=None --data_list=train 	2>&1 | tee -a param_R_2plus1_D_100ep/net_87ep_16_1.txt &&
echo "Finished net_82ep_8_2 => net_87ep_16_1"