#!/bin/sh


echo "Running net_96ep => net_98ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_96ep --save_name=net_98ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_98ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_98ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_96ep => net_98ep"

echo "Running net_98ep => net_100ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_98ep --save_name=net_100ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_100ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_100ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_98ep => net_100ep"

echo "Running net_100ep => net_102ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_100ep --save_name=net_102ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_102ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_102ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_100ep => net_102ep"

echo "Running net_102ep => net_104ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_102ep --save_name=net_104ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_104ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_104ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_102ep => net_104ep"

echo "Running net_104ep => net_106ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_104ep --save_name=net_106ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_106ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_106ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_104ep => net_106ep"

echo "Running net_106ep => net_108ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_106ep --save_name=net_108ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_108ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_108ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_106ep => net_108ep"

echo "Running net_108ep => net_110ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_108ep --save_name=net_110ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_110ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_110ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_108ep => net_110ep"

echo "Running net_110ep => net_112ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_110ep --save_name=net_112ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_112ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_112ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_110ep => net_112ep"

echo "Running net_112ep => net_114ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_112ep --save_name=net_114ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_114ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_114ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_112ep => net_114ep"

echo "Running net_114ep => net_116ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_114ep --save_name=net_116ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_116ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_116ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_114ep => net_116ep"

echo "Running net_116ep => net_118ep"
python train.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_116ep --save_name=net_118ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_118ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_R_2plus1_D_100ep --load_name=net_118ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_116ep => net_118ep"

