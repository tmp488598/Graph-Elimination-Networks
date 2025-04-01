import datetime
import os
import torch
import logging
import graphgps  # noqa, register custom modules
import optuna
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             load_cfg,
                                             makedirs_rm_exist)
from graphgps.yaml_config import set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
import argparse
from graphgps.utils import save_result

import subprocess

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)



def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


def objective(trial):
    """The objective function for Optuna optimization."""
    # Overriding the config values based on the trial parameters
    # cfg.optim.base_lr = trial.suggest_categorical('graph_pooling',  ["add", "mean"])
    cfg.train.batch_size = trial.suggest_categorical('batch_size', [500, 300, 50])
    cfg.optim.base_lr = trial.suggest_loguniform('base_lr', 0.0005, 0.005)
    cfg.optim.weight_decay = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4])
    cfg.optim.max_epoch = trial.suggest_int('max_epoch', 300, 500, step=50)
    cfg.optim.num_warmup_epochs = trial.suggest_categorical('num_warmup_epochs', [5, 10, 20])
    cfg.gnn.dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)

    cfg.gnn.dim_inner = trial.suggest_categorical('dim_inner', [64, 80, 96])
    cfg.gnn.gens.edge_dim = cfg.gnn.dim_inner

    cfg.gnn.layers_mp = trial.suggest_categorical('layers_mp', [8, 9,10,11])
    cfg.gnn.layers_post_mp = trial.suggest_categorical('layers_post_mp', [1,2,3])
    # cfg.gnn.residual = trial.suggest_categorical('residual', [True, False])
    # cfg.gnn.batchnorm = trial.suggest_categorical('batchnorm', [True, False])
    cfg.gnn.gens.K = trial.suggest_categorical('K', [1, 3, 5])
    cfg.gnn.gens.gamma = trial.suggest_float('gamma', 0.5, 0.8, step=0.1)
    # cfg.gnn.gens.fea_drop = trial.suggest_categorical('fea_drop', ['simple'])
    cfg.gnn.gens.base_model = trial.suggest_categorical('base_model', ['gat'])
    cfg.gnn.gens.heads = trial.suggest_categorical("heads", [1, 4])
    # cfg.gnn.gens.hop_att = trial.suggest_categorical('hop_att', [True, False])
    cfg.gnn.gens.use_ffN = trial.suggest_categorical('use_ffN', [True, False])
    cfg.gnn.gens.norm_type = trial.suggest_categorical('norm_type', ['batch', 'layer', 'None'])

    out_max_best_test = 0.0
    num_run = 0
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run

        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        # cfg.accelerator = 'cuda:{}'.format(0)

        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()

        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )

        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

        logging.info(cfg.gnn)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)

        # Start training
        if cfg.train.mode == 'standard':
            datamodule = GraphGymDataModule()
            max_best_test = train(model, datamodule, logger=True)
        else:
            max_best_test = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

        max_best_test = float(max_best_test.split(': ')[1])

        out_max_best_test += max_best_test

        num_run += 1

    save_result(out_max_best_test/num_run)

    return out_max_best_test/num_run

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    parser.add_argument('--search', type=bool, default=False, help='Whether to search by parameters.')

    return parser.parse_args()
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    # warnings.simplefilter(action='ignore', category=FutureWarning)


    # Load cmd line args
    args = parse_args()

    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    auto_select_device()

    if args.search:
        study = optuna.create_study(direction="maximize",
                                    study_name= "pepfunc",
                                    storage=f"sqlite:///optuna_{cfg.wandb.project}.db",
                                    load_if_exists=True)

        study.optimize(objective, n_trials=100)  # Run the optimization for 500 trials

        # Output the best hyperparameters
        print("Best hyperparameters: ", study.best_params)

    else:
        for run_id, seed, split_index in zip(*run_loop_settings()):
            # Set configurations for each run

            custom_set_run_dir(cfg, run_id)
            set_printing()
            cfg.dataset.split_index = split_index
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            # cfg.accelerator = 'cuda:{}'.format(1)

            if cfg.pretrained.dir:
                cfg = load_pretrained_model_cfg(cfg)
            logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                         f"split_index={cfg.dataset.split_index}")
            logging.info(f"    Starting now: {datetime.datetime.now()}")
            # Set machine learning pipeline
            loaders = create_loader()
            loggers = create_logger()
            model = create_model()
            if cfg.pretrained.dir:
                model = init_model_from_pretrained(
                    model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                    cfg.pretrained.reset_prediction_head, seed=cfg.seed
                )
            optimizer = create_optimizer(model.parameters(),
                                         new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            # Print model info
            logging.info(model)
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info('Num parameters: %s', cfg.params)
            # Start training
            if cfg.train.mode == 'standard':
                if cfg.wandb.use:
                    logging.warning("[W] WandB logging is not supported with the "
                                    "default train.mode, set it to `custom`")
                datamodule = GraphGymDataModule()
                train(model, datamodule, logger=True)
            else:
                train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                           scheduler)




        # Aggregate results from different seeds
        try:
            agg_runs(cfg.out_dir, cfg.metric_best)
        except Exception as e:
            logging.info(f"Failed when trying to aggregate multiple runs: {e}")
        # When being launched in batch mode, mark a yaml as done
        if args.mark_done:
            os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
