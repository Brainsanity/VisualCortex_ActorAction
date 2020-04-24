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
# echo 'net_50ep => net_55ep ...'
# python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_50ep --save_name=net_55ep &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_55ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_55ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

# echo 'net_55ep => net_60ep ...'
# python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_55ep --save_name=net_60ep &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_60ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_60ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

# echo 'net_60ep => net_65ep ...'
# python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_60ep --save_name=net_65ep &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_65ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_65ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

# echo 'net_65ep => net_70ep ...'
# python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_65ep --save_name=net_70ep &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_70ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_70ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

# echo 'net_70ep => net_75ep ...'
# python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_70ep --save_name=net_75ep &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_75ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_75ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

# echo 'net_75ep => net_80ep ...'
# python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_75ep --save_name=net_80ep &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_80ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_80ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt


# waiting to run...
echo 'net_50ep => net_51ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_50ep --save_name=net_51ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_51ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_51ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_51ep => net_52ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_51ep --save_name=net_52ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_52ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_52ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_52ep => net_53ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_52ep --save_name=net_53ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_53ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_53ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_53ep => net_54ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_53ep --save_name=net_54ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_54ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_54ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_55ep => net_56ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_50ep --save_name=net_56ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_56ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_56ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_56ep => net_57ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_56ep --save_name=net_57ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_57ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_57ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_57ep => net_58ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_57ep --save_name=net_58ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_58ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_58ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_58ep => net_59ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_58ep --save_name=net_59ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_59ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_59ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_60ep => net_61ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_60ep --save_name=net_61ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_61ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_61ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_61ep => net_62ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_61ep --save_name=net_62ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_62ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_62ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_62ep => net_63ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_62ep --save_name=net_63ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_63ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_63ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt &&

echo 'net_63ep => net_64ep ...'
python train.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --num_epochs=1 --cont=1 --load_name=net_63ep --save_name=net_64ep &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_64ep --note=None --data_list=val 		2>&1 | tee -a print_pcsa_v4.txt &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_64ep --note=None --data_list=train 	2>&1 | tee -a print_pcsa_v4.txt