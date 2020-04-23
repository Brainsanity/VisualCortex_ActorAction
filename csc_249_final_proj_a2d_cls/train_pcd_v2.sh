#!/bin/sh

# counter=0
# curH=$(date +%H)	# current hour (00)
# while [ $curH -ne 11 ] #[ ! -f "net_20ep.ckpt" ]
# do
# 	echo "$counter. Not ready yet..."
# 	#((counter+=1))
# 	sleep 5s
# 	curH=$(date +%H)
# done

echo 'net_lr001_20ep => net__lr0001_25ep ...'
python train.py 	  --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.0001 --cont=1 --crop=1 --load_name=net_lr001_20ep --save_name=net_lr0001_25ep
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_25ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_25ep --note=None --data_list=train 	2>&1 | tee -a print_pcd_v2.txt

echo 'net_lr0001_25ep => net_lr0001_30ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.0001 --cont=1 --crop=1 --load_name=net_lr0001_25ep --save_name=net_lr0001_30ep
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_30ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_30ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt

echo 'net_lr0001_30ep => net_lr00001_35ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr0001_30ep --save_name=net_lr00001_35ep
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_35ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_35ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt

echo 'net_lr00001_35ep => net_lr00001_40ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_35ep --save_name=net_lr00001_40ep
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_40ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_40ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt

echo 'net_lr00001_40ep => net_lr00001_45ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_40ep --save_name=net_lr00001_45ep
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_45ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_45ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt