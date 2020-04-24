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


## crop

# echo 'net_lr001_20ep => net__lr0001_25ep ...'
# python train.py 	  --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.0001 --cont=1 --crop=1 --load_name=net_lr001_20ep --save_name=net_lr0001_25ep
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_25ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_25ep --note=None --data_list=train 	2>&1 | tee -a print_pcd_v2.txt

# echo 'net_lr0001_25ep => net_lr0001_30ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.0001 --cont=1 --crop=1 --load_name=net_lr0001_25ep --save_name=net_lr0001_30ep
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_30ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_30ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt

# echo 'net_lr0001_30ep => net_lr00001_35ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr0001_30ep --save_name=net_lr00001_35ep
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_35ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_35ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt

# echo 'net_lr00001_35ep => net_lr00001_40ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_35ep --save_name=net_lr00001_40ep
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_40ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_40ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt

# echo 'net_lr00001_40ep => net_lr00001_45ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_40ep --save_name=net_lr00001_45ep
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_45ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_45ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt


# waiting to run... | running now... | finished
# echo 'net_lr00001_45ep => net_lr00001_50ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_45ep --save_name=net_lr00001_50ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_50ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_50ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_50ep => net_lr00001_55ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_50ep --save_name=net_lr00001_55ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_55ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_55ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&


# waiting to run...
# echo 'net_lr00001_55ep => net_lr00001_60ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_60ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_60ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_60ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_60ep => net_lr00001_65ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_65ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_65ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_65ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_65ep => net_lr00001_70ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_70ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_70ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_70ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_70ep => net_lr00001_75ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_75ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_75ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_75ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_75ep => net_lr00001_80ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_80ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_80ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_80ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_80ep => net_lr00001_85ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_85ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_85ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_85ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_85ep => net_lr00001_90ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_90ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_90ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_90ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_90ep => net_lr00001_95ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_95ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_95ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_95ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr00001_95ep => net_lr00001_100ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --crop=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_100ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_100ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_100ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&


# running now... | finished
## no crop
# echo 'scratch => net_lr001_5ep ...'
# python train.py 	  --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.001 --save_name=net_lr001_5ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_5ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_5ep --note=None --data_list=train 	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr001_5ep => net_lr001_10ep ...'
# python train.py 	  --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.001 --cont=1 --load_name=net_lr001_5ep --save_name=net_lr001_10ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_10ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_10ep --note=None --data_list=train 	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr001_10ep => net_lr001_15ep ...'
# python train.py 	  --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.001 --cont=1 --load_name=net_lr001_10ep --save_name=net_lr001_15ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_15ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_15ep --note=None --data_list=train 	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr001_15ep => net_lr001_20ep ...'
# python train.py 	  --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.001 --cont=1 --load_name=net_lr001_15ep --save_name=net_lr001_20ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_20ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr001_20ep --note=None --data_list=train 	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr001_20ep => net_lr0001_25ep ...'
# python train.py 	  --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.0001 --cont=1 --load_name=net_lr001_20ep --save_name=net_lr0001_25ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_25ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_25ep --note=None --data_list=train 	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr0001_25ep => net_lr0001_30ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.0001 --cont=1 --load_name=net_lr0001_25ep --save_name=net_lr0001_30ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_30ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr0001_30ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# echo 'net_lr0001_30ep => net_lr00001_35ep ...'
# python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr0001_30ep --save_name=net_lr00001_35ep &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_35ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
# python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_35ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

# running now...
echo 'net_lr00001_35ep => net_lr00001_40ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_35ep --save_name=net_lr00001_40ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_40ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_40ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_40ep => net_lr00001_45ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_40ep --save_name=net_lr00001_45ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_45ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_45ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_45ep => net_lr00001_50ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_45ep --save_name=net_lr00001_50ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_50ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_50ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_50ep => net_lr00001_55ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_50ep --save_name=net_lr00001_55ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_55ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_55ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_55ep => net_lr00001_60ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_60ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_60ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_60ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_60ep => net_lr00001_65ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_65ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_65ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_65ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_65ep => net_lr00001_70ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_70ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_70ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_70ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_70ep => net_lr00001_75ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_75ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_75ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_75ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&

echo 'net_lr00001_75ep => net_lr00001_80ep ...'
python train.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --num_epochs=5 --lr=0.00001 --cont=1 --load_name=net_lr00001_55ep --save_name=net_lr00001_80ep &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_80ep --note=None --data_list=val		2>&1 | tee -a print_pcd_v2.txt &&
python eval_on_val.py --net=per_class_detection --version=2 --model_path=param_pcd_v2 --load_name=net_lr00001_80ep --note=None --data_list=train	2>&1 | tee -a print_pcd_v2.txt &&