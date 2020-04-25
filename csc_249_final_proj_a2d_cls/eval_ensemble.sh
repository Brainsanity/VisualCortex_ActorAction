#!/bin/sh

while [ ! -f 'Predict_val_R_2plus1_D_72ep.txt' ]
do
	echo "Not ready yet..."
	sleep 5s
done

python eval_on_val.py --net=ensemble --pcd_file=param_pcd_100ep/net_70ep.ckpt --pcsa_file=param_pcsa_v4_100ep/net_55ep.ckpt --pcd3d_file=param_pcd3d/net_95ep.ckpt --pcd_f1=59.4 --pcsa_f1=58.9 --pcd3d_f1=61.7 --nframes=16 --speed=1 --data_list=val

python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_72ep --nframes=16 --speed=1 --data_list=val --note=72ep_16_1