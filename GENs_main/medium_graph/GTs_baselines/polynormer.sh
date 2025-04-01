
for layer in 1 2 3 4 5 6 7 8 9 10
do
for hidden_channels in 32 64 256
do

## heterophilic datasets
python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --local_layers $layer --global_layers $layer --weight_decay 0.0 --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --device $1 --beta 0.5  --save_result --model polynormer
python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 2 --device $1  --save_result--model polynormer
python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 2 --local_layers $layer --global_layers $layer --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 8 --metric rocauc --device $1 --save_result --model polynormer
python main.py --dataset tolokers --hidden_channels $hidden_channels --epochs 800 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.5 --in_dropout 0.2 --num_heads 16 --metric rocauc --device $1 --beta 0.1 --save_result --model polynormer
python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.2 --global_dropout 0.5 --num_heads 8 --metric rocauc --device $1 --in_dropout 0.15 --beta 0.4 --pre_ln --save_result --model polynormer

## homophilic datasets
python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1200 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device $1 --save_result --model polynormer
python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device $1 --save_result --model polynormer
python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-4 --dropout 0.3 --in_dropout 0.1 --num_heads 8 --device $1 --save_result --model polynormer
python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.1 --num_heads 8 --device $1 --save_result --model polynormer
python main.py --dataset wikics --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --device $1 --save_result --model polynormer


done 
done 
