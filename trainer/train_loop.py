import gc
import wandb, optuna
import torch
import numpy as np
from tqdm.auto import tqdm
from optuna.trial import TrialState
from optuna.integration.wandb import WeightsAndBiasesCallback

from torch.optim.swa_utils import update_bn
from configuration import CFG
from trainer import FBPTrainer, MPLTrainer
from trainer.trainer_utils import get_name
from utils.helper import class2dict

g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: any) -> None:
    """ Base Trainer Loop Function """
    fold_list = [i for i in range(cfg.n_folds)]
    for fold in tqdm(fold_list):
        print(f'============== {fold}th Fold Train & Validation ==============')
        wandb.init(project=cfg.name,
                   name=f'[{cfg.model_arch}]' + f'fold{fold}/' + cfg.model,
                   config=class2dict(cfg),
                   group=cfg.model,
                   job_type='train',
                   entity="qcqced")
        val_score_max, fold_swa_loss = np.inf, []
        train_input = FBPTrainer(cfg, g)  # init object
        loader_train, loader_valid, train = train_input.make_batch(fold)
        model, swa_model, criterion, optimizer,\
            lr_scheduler, swa_scheduler, awp, save_parameter = train_input.model_setting(len(train))

        for epoch in range(cfg.epochs):
            print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
            train_loss, grad_norm, lr = train_input.train_fn(
                loader_train, model, criterion, optimizer, lr_scheduler,
                epoch, awp, swa_model, cfg.swa_start, swa_scheduler
            )
            valid_loss = train_input.valid_fn(
                loader_valid, model, criterion
            )
            wandb.log({
                '<epoch> Train Loss': train_loss,
                '<epoch> Valid Loss': valid_loss,
                '<epoch> Gradient Norm': grad_norm,
                '<epoch> lr': lr
            })
            print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Valid Loss: {np.round(valid_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Gradient Norm: {np.round(grad_norm, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] lr: {lr}')
            if val_score_max >= valid_loss:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                print(f'Best Score: {valid_loss}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}{cfg.state_dict}fold{fold}_{get_name(cfg)}_state_dict.pth')
                val_score_max = valid_loss

            del train_loss, valid_loss, grad_norm, lr
            gc.collect(), torch.cuda.empty_cache()

        update_bn(loader_train, swa_model)
        swa_loss = train_input.swa_fn(loader_valid, swa_model, criterion)
        print(f'Fold[{fold}/{fold_list[-1]}] SWA Loss: {np.round(swa_loss, 4)}')

        if val_score_max >= swa_loss:
            print(f'[Update] Valid Score : ({val_score_max:.4f} => {swa_loss:.4f}) Save Parameter')
            print(f'Best Score: {swa_loss}')
            torch.save(model.state_dict(),
                       f'{cfg.checkpoint_dir}{cfg.state_dict}SWA_fold{fold}_{get_name(cfg)}_state_dict.pth')

    wandb.finish()

