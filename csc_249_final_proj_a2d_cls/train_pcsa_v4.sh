#!/bin/sh

# # net_46ep => net_47ep
# python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_46ep --save_name=net_47ep

# echo 'net_47ep => net_48ep ...'
# # python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_47ep --save_name=net_48ep &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_48ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_48ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt

# echo 'net_48ep => net_49ep ...'
# # python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_48ep --save_name=net_49ep
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_49ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_49ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt

# running now... | finished
echo 'net_50ep => net_55ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_50ep --save_name=net_55ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_55ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_55ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_55ep => net_60ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_55ep --save_name=net_60ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_60ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_60ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_60ep => net_65ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_60ep --save_name=net_65ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_65ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_65ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_65ep => net_70ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_65ep --save_name=net_70ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_70ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_70ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_70ep => net_75ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_70ep --save_name=net_75ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_75ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_75ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_75ep => net_80ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_75ep --save_name=net_80ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_80ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_80ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt