#!/bin/sh
# exec > print_pcsa_v4.txt

python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_43ep --note=v4_43ep 		--data_list=val &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_43ep --note=v4_43ep_train --data_list=train &&

python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_44ep --note=v4_44ep 		--data_list=val &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_44ep --note=v4_44ep_train --data_list=train &&

python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_47ep --note=v4_47ep 		--data_list=val &&
python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_47ep --note=v4_47ep_train --data_list=train &&

# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_48ep --note=v4_48ep 		--data_list=val &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_48ep --note=v4_48ep_train --data_list=train

# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_49ep --note=v4_49ep 		--data_list=val &&
# python eval_on_val.py --net=per_class_soft_attention --version=4 --model_path=param_pcsa_v4_100ep --load_name=net_49ep --note=v4_49ep_train --data_list=train