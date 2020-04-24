#!/bin/sh
# exec > print_pcd_v1.txt

python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --crop=1 --load_name=net_crop_65ep --note=v4_65ep 		 --data_list=val &&
python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --crop=1 --load_name=net_crop_65ep --note=v4_65ep_train --data_list=train &&

python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --crop=1 --load_name=net_crop_70ep --note=v4_70ep 		 --data_list=val &&
python eval_on_val.py --net=per_class_detection --model_path=param_pcd_100ep --crop=1 --load_name=net_crop_70ep --note=v4_70ep_train --data_list=train