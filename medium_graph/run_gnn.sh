## homophilic datasets

#python main.py --gnn gens --dataset amazon-photo --lr 0.0006 --hidden_channels 256 --local_layers 3  --dropout 0.7  --ln --weight_decay 0.001  --epochs 500   --K 1 --gamma 1.0  --heads 4  --base_model gat --hop_att True --fea_drop normal --device $1 --runs 3

python main.py --gnn gens --dataset wikics --lr 0.00035 --hidden_channels 1024 --local_layers 2  --dropout 0.9  --ln --weight_decay 1e-05  --epochs 1000   --K 2  --gamma 0.8 --heads 4 --base_model gat --hop_att True --fea_drop simple --device $1 --runs 3

#python main.py --gnn gens --dataset cora --lr 0.001 --hidden_channels 1024 --local_layers 2  --dropout 0.8  --weight_decay 0.0 --K 3  --gamma 1.0  --base_model gat --hop_att True --fea_drop simple --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5

python main.py --gnn gens --dataset citeseer --lr 0.004 --hidden_channels 512 --local_layers 2  --dropout 0.9 --ln --bn  --weight_decay 0.1 --K 4  --gamma 1.0 --heads 4   --base_model gcn --hop_att True --fea_drop none --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5

#python main.py --gnn gens --dataset pubmed --lr 0.001 --hidden_channels 1024 --local_layers 2  --dropout 0.7  --weight_decay 0.0005 --K 6  --gamma 0.4  --base_model gat --hop_att True --fea_drop none --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5

## heterophilic datasets

#python main.py --gnn gens --dataset amazon-ratings --lr 0.0015 --hidden_channels 300 --local_layers 7  --dropout 0.55  --bn --weight_decay 1e-06 --epochs 2500   --K 2  --gamma 0.8  --base_model gat --hop_att True --fea_drop simple --device $1 --runs 3

python main.py --gnn gens --dataset roman-empire --lr 0.004 --hidden_channels 192 --local_layers 9  --dropout 0.2  --bn --weight_decay 0.0001 --epochs 2000   --K 1  --gamma 1.0 --heads 4   --base_model gat --hop_att True --fea_drop simple --device $1 --runs 3

#python main.py --gnn gens --dataset questions --lr 0.0035 --hidden_channels 256 --local_layers 10  --dropout 0.4  --bn --weight_decay 1e-05 --epochs 1500   --K 1  --gamma 1.0  --base_model gcn --hop_att True --fea_drop simple --device $1 --runs 3

python main.py --gnn gens  --dataset squirrel --lr 0.002 --hidden_channels 64 --local_layers 7  --dropout 0.2 --weight_decay 1e-5   --K 2  --gamma 1.0 --heads 4   --base_model gcn --hop_att False --fea_drop simple --device $1 --runs 10

python main.py --gnn gens --dataset chameleon --lr 0.005 --hidden_channels 16 --local_layers 9  --dropout 0.2 --ln --res --weight_decay 0.0001   --K 2  --gamma 1.0 --heads 4   --base_model gcn --hop_att False --fea_drop simple --device $1 --runs 10