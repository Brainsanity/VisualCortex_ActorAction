#!/bin/sh

## crop
# echo "net_crop_60ep => net_crop_65ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_60ep --save_name=net_crop_65ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_65ep --note=v4_65ep 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_65ep --note=v4_65ep_train --data_list=train	2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_crop_65ep => net_crop_70ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_65ep --save_name=net_crop_70ep
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_70ep --note=v4_70ep 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_70ep --note=v4_70ep_train --data_list=train	2>&1 | tee -a print_pcd_v1.txt

# echo "net_crop_70ep => net_crop_75ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_70ep --save_name=net_crop_75ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_75ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_75ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_crop_75ep => net_crop_80ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_75ep --save_name=net_crop_80ep
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_80ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_80ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt


# running now... | finished
## no crop
# echo "net_56ep => net_60ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=4 --cont=1 --load_name=net_56ep --save_name=net_60ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_60ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_60ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&


# echo "net_60ep => net_65ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --load_name=net_60ep --save_name=net_65ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_65ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_65ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&


# running now... | finished
# echo "net_65ep => net_70ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --load_name=net_65ep --save_name=net_70ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_70ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_70ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_70ep => net_75ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --load_name=net_70ep --save_name=net_75ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_75ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_75ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_75ep => net_80ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --load_name=net_75ep --save_name=net_80ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_80ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_80ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&



## crop
# echo "net_crop_80ep => net_crop_85ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_80ep --save_name=net_crop_85ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_85ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_85ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_crop_85ep => net_crop_90ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_85ep --save_name=net_crop_90ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_90ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_90ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# waiting to run...
# echo "net_crop_90ep => net_crop_95ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_90ep --save_name=net_crop_95ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_95ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_95ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_crop_95ep => net_crop_100ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_95ep --save_name=net_crop_100ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_100ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_100ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt



## no crop
# running now...
# echo "scratch => net_5ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --save_name=net_5ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_5ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# # python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_5ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_5ep => net_10ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --load_name=net_5ep --save_name=net_10ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_10ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# # python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_10ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

# echo "net_10ep => net_15ep ..."
# python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --load_name=net_10ep --save_name=net_15ep &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_15ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# # python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_15ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&


## crop
echo "scratch => net_crop_5ep ..."
python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --crop=1 --save_name=net_crop_5ep &&
python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_70ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_5ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

echo "net_crop_5ep => net_crop_10ep ..."
python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_5ep --save_name=net_crop_10ep &&
python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_10ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_10ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

echo "net_crop_10ep => net_crop_15ep ..."
python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_10ep --save_name=net_crop_15ep &&
python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_15ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_15ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&

echo "net_crop_15ep => net_crop_20ep ..."
python train.py --net=per_class_detection --model_path=param_pcd_100ep --num_epochs=5 --cont=1 --crop=1 --load_name=net_crop_20ep --save_name=net_crop_20ep &&
python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_20ep --note=None 		 --data_list=val	2>&1 | tee -a print_pcd_v1.txt &&
# python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --load_name=net_crop_20ep --note=None --data_list=train 		2>&1 | tee -a print_pcd_v1.txt &&