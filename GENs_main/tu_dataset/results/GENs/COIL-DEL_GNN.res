{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 0, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'gens', 'num_layers': 6, 'num_fc': 1, 'hidden_dim': 128, 'node_encoder': 1, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'simple', 'gens_hop_att': True, 'gens_heads': 1, 'gens_base_model': 'gat', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.8535897135734558
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 0, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'gens', 'num_layers': 6, 'num_fc': 1, 'hidden_dim': 128, 'node_encoder': 1, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'simple', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.8664102554321289
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 1, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'gcn', 'num_layers': 6, 'num_fc': 1, 'hidden_dim': 128, 'node_encoder': 1, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'simple', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.7976922988891602
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 0, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'dgmlp', 'num_layers': 1, 'num_fc': 1, 'hidden_dim': 128, 'node_encoder': 1, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'simple', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.8169230222702026
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 1, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'gat', 'num_layers': 6, 'num_fc': 1, 'hidden_dim': 128, 'node_encoder': 1, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'simple', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.7835897207260132
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 0, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'gcn2', 'num_layers': 6, 'num_fc': 1, 'hidden_dim': 128, 'node_encoder': 1, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'simple', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.766410231590271
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 0, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'ongnn', 'num_layers': 9, 'num_fc': 2, 'hidden_dim': 128, 'node_encoder': 1, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'simple', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.8112820386886597
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 1, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'gcon', 'num_layers': 1, 'num_fc': 2, 'hidden_dim': 128, 'node_encoder': 2, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'normal', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.8476922416687012
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 1, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'gin', 'num_layers': 6, 'num_fc': 1, 'hidden_dim': 128, 'node_encoder': 2, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'normal', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.84333336353302
{'dataset': 'COIL-DEL', 'out_dir': './results/GENs/', 'eval_mad': False, 'feature_params': {'degree': True, 'onehot_maxdeg': 10, 'AK': 1, 'centrality': False, 'remove_edges': 'none', 'edge_noises_add': 0, 'edge_noises_delete': 0, 'group_degree': 0, 'virtual_node': False}, 'params': {'gpu': 1, 'seed': 41, 'net': 'GNN', 'epoch_select': 'test_max', 'with_eval_mode': True, 'data_root': 'datasets', 'epochs': 100, 'batch_size': 128, 'lr': 0.0005, 'scheduler': False, 'weight_decay': 0.0005}, 'net_params': {'gnn_type': 'dagnn', 'num_layers': 1, 'num_fc': 2, 'hidden_dim': 128, 'node_encoder': 2, 'residual': False, 'global_pool': 'sum', 'dropout': 0.0, 'batch_norm': True, 'concat': False, 'sum_x': False, 'gens_K': 2, 'gens_gamma': 1, 'gens_fea_drop': 'normal', 'gens_hop_att': False, 'gens_heads': 1, 'gens_base_model': 'gcn', 'gens_edge_dim': -1, 'gens_att_dropout': 0, 'gens_att_concat': False}} 0.818461537361145
