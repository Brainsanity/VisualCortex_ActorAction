#!/bin/sh


# echo "Running net_91ep => net_92ep"
# python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_91ep --save_name=net_92ep &&
# python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_92ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
# python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_92ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
# echo "Finished net_91ep => net_92ep"

# echo "Running net_92ep => net_93ep"
# python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_92ep --save_name=net_93ep &&
# python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_93ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
# python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_93ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
# echo "Finished net_92ep => net_93ep"

# echo "Running net_93ep => net_94ep"
# python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_93ep --save_name=net_94ep &&
# python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_94ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
# python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_94ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
# echo "Finished net_93ep => net_94ep"

echo "Running net_96ep => net_97ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_96ep --save_name=net_97ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_97ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_97ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_96ep => net_97ep"

echo "Running net_97ep => net_98ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_97ep --save_name=net_98ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_98ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_98ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_97ep => net_98ep"

echo "Running net_98ep => net_99ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_98ep --save_name=net_99ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_99ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_99ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_98ep => net_99ep"

echo "Running net_99ep => net_100ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_99ep --save_name=net_100ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_100ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_100ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_99ep => net_100ep"

echo "Running net_100ep => net_101ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_100ep --save_name=net_101ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_101ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_101ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_100ep => net_101ep"

echo "Running net_101ep => net_102ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_101ep --save_name=net_102ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_102ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_102ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_101ep => net_102ep"

echo "Running net_102ep => net_103ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_102ep --save_name=net_103ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_103ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_103ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_102ep => net_103ep"

echo "Running net_103ep => net_104ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_103ep --save_name=net_104ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_104ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_104ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_103ep => net_104ep"

echo "Running net_104ep => net_105ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_104ep --save_name=net_105ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_105ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_105ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_104ep => net_105ep"

echo "Running net_105ep => net_106ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_105ep --save_name=net_106ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_106ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_106ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_105ep => net_106ep"

echo "Running net_106ep => net_107ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_106ep --save_name=net_107ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_107ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_107ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_106ep => net_107ep"

echo "Running net_107ep => net_108ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_107ep --save_name=net_108ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_108ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_108ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_107ep => net_108ep"

echo "Running net_108ep => net_109ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_108ep --save_name=net_109ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_109ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_109ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_108ep => net_109ep"

echo "Running net_109ep => net_110ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_109ep --save_name=net_110ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_110ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_110ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_109ep => net_110ep"

echo "Running net_110ep => net_111ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_110ep --save_name=net_111ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_111ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_111ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_110ep => net_111ep"

echo "Running net_111ep => net_112ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_111ep --save_name=net_112ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_112ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_112ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_111ep => net_112ep"

echo "Running net_112ep => net_113ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_112ep --save_name=net_113ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_113ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_113ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_112ep => net_113ep"

echo "Running net_113ep => net_114ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_113ep --save_name=net_114ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_114ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_114ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_113ep => net_114ep"

echo "Running net_114ep => net_115ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_114ep --save_name=net_115ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_115ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_115ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_114ep => net_115ep"

echo "Running net_115ep => net_116ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_115ep --save_name=net_116ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_116ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_116ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_115ep => net_116ep"

echo "Running net_116ep => net_117ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_116ep --save_name=net_117ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_117ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_117ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_116ep => net_117ep"

echo "Running net_117ep => net_118ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_117ep --save_name=net_118ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_118ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_118ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_117ep => net_118ep"

echo "Running net_118ep => net_119ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_118ep --save_name=net_119ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_119ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_119ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_118ep => net_119ep"

echo "Running net_119ep => net_120ep"
python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_119ep --save_name=net_120ep &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_120ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&
python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_120ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&
echo "Finished net_119ep => net_120ep"

