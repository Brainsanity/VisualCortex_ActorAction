fname = 'train_pcd3d.sh';
f = fopen(fname,'w');

fprintf( f, '#!/bin/sh\n\n\n' );

preEPs = [91:93, 96:119];
postEPs = [92:94, 97:120];
for( i = 1 : size(preEPs,2) )
	fprintf( f, ['echo "Running net_', num2str(preEPs(i)), 'ep => net_', num2str(postEPs(i)) 'ep"\n'] );
	fprintf( f, ['python train.py --net=R_2plus1_D --model_path=param_pcd3d --nframes=16 --speed=1 --num_epochs=1 --cont=1 --load_name=net_', num2str(preEPs(i)), 'ep --save_name=net_', num2str(postEPs(i)), 'ep &&\n'] );
	fprintf( f, ['python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_', num2str(postEPs(i)), 'ep --nframes=16 --speed=1 --note=None  --data_list=val	2>&1 | tee -a print_pcd3d.txt &&\n'] );
	fprintf( f, ['python eval_on_val.py --net=R_2plus1_D --model_path=param_pcd3d --load_name=net_', num2str(postEPs(i)), 'ep --nframes=16 --speed=1 --note=None  --data_list=train	2>&1 | tee -a print_pcd3d.txt &&\n'] );
	fprintf( f, ['echo "Finished net_', num2str(preEPs(i)), 'ep => net_', num2str(postEPs(i)) 'ep"\n'] );
	fprintf( f, '\n' );
end

fclose(f);