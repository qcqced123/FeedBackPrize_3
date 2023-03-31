import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model # 여기 지우고
        self.criterion = criterion # 여기 지우고
        self.metric_ftns = metric_ftns # 여기 지우고
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def train_with_validation(self):
        # This is Test Code for Appending Cross-Validaton Strategy (fold to epoch)
        cfg_list = [CFG]
        for cfg in cfg_list:
            # init wandb
            wandb.init(project="[Append Ver 2]UPPPM Token Classification",
                       name='[Append Version 2.2]' + 'Fold 0' + cfg.model_name,
                       config=class2dict(cfg),
                       group=cfg.model_name,
                       job_type="train",
                       entity="qcqced")
            wandb_config = wandb.config
            print(f'========================= Retriever Model :{cfg.model_name} =========================')
            fold_list, swa_score_max = [i for i in range(cfg.n_folds)], -np.inf

            for fold in tqdm(fold_list):
                print(f'============== {fold + 1}th Fold Train & Validation ==============')
                val_score_max = -np.inf
                fold_train_loss_list, fold_valid_loss_list, fold_score_list = [], [], []
                fold_swa_loss, fold_swa_score = [], []

                train_input = TrainInput(cfg, train_df)  # init object
                model, swa_model, criterion, optimizer, save_parameter = train_input.model_setting()
                loader_train, loader_valid, train, valid, valid_labels = train_input.make_batch(fold)

                # Scheduler Setting
                if cfg.swa:
                    swa_start = cfg.swa_start
                    swa_scheduler = SWALR(
                        optimizer,
                        swa_lr=cfg.swa_lr,  # Later Append
                        anneal_epochs=cfg.anneal_epochs,
                        anneal_strategy=cfg.anneal_strategy
                    )

                if cfg.scheduler == 'cosine':
                    scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=int(
                            len(train) / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps) * cfg.warmup_ratio,
                        num_training_steps=int(
                            len(train) / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps),
                        num_cycles=cfg.num_cycles
                    )
                elif cfg.scheduler == 'cosine_annealing':
                    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=int(
                            len(train) / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps) * cfg.warmup_ratio,
                        num_training_steps=int(
                            len(train) / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps),
                        num_cycles=8
                    )
                else:
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=int(len(train) / cfg.batch_size * cfg.epochs) * cfg.warmup_ratio,
                        num_training_steps=int(len(train) / cfg.batch_size * cfg.epochs),
                        num_cycles=cfg.num_cycles
                    )

                for epoch in range(cfg.epochs):
                    print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
                    if cfg.swa:
                        train_loss, valid_loss, score, grad_norm, lr = train_fn(
                            cfg,
                            loader_train,
                            loader_valid,
                            model,
                            criterion,
                            optimizer,
                            scheduler,
                            valid,
                            valid_labels,
                            int(epoch),
                            swa_model=swa_model,
                            swa_start=swa_start,
                            swa_scheduler=swa_scheduler,
                        )
                    else:
                        train_loss, valid_loss, score = train_fn(
                            cfg,
                            loader_train,
                            loader_valid,
                            model,
                            criterion,
                            optimizer,
                            scheduler,
                            valid,
                            valid_labels,
                            int(epoch),
                        )

                    train_loss = train_loss.detach().cpu().numpy()
                    valid_loss = valid_loss.detach().cpu().numpy()
                    grad_norm = grad_norm.detach().cpu().numpy()

                    fold_train_loss_list.append(train_loss)
                    fold_valid_loss_list.append(valid_loss)
                    fold_score_list.append(score)

                    wandb.log({
                        '<epoch> Train Loss': train_loss,
                        '<epoch> Valid Loss': valid_loss,
                        '<epoch> Pearson_Score': score,
                        '<epoch> Gradient Norm': grad_norm,
                        '<epoch> lr': lr
                    })

                    print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
                    print(f'[{epoch + 1}/{cfg.epochs}] Valid Loss: {np.round(valid_loss, 4)}')
                    print(f'[{epoch + 1}/{cfg.epochs}] Pearson Score: {np.round(score, 4)}')
                    print(f'[{epoch + 1}/{cfg.epochs}] Gradient Norm: {np.round(grad_norm, 4)}')
                    print(f'[{epoch + 1}/{cfg.epochs}] lr: {lr}')

                    if val_score_max <= score:
                        print(f'[Update] Valid Score : ({val_score_max:.4f} => {score:.4f}) Save Parameter')
                        print(f'Best Score: {score}')
                        torch.save(model.state_dict(),
                                   f'Ver2-3_Token_Classification_Fold{fold}_DeBERTa_V3_Large.pth')
                        val_score_max = score

                del train_loss, valid_loss
                gc.collect()
                torch.cuda.empty_cache()

                print(f'================= {fold + 1}th Train & Validation =================')
                fold_train_loss = np.mean(fold_train_loss_list)
                fold_valid_loss = np.mean(fold_valid_loss_list)
                fold_score = np.mean(fold_score_list)
                wandb.log({f'<Fold{fold + 1}> Train Loss': fold_train_loss,
                           f'<Fold{fold + 1}> Valid Loss': fold_valid_loss,
                           f'<Fold{fold + 1}> Pearson_Score': fold_score, })
                print(f'Fold[{fold + 1}/{fold_list[-1] + 1}] Train Loss: {np.round(fold_train_loss, 4)}')
                print(f'Fold[{fold + 1}/{fold_list[-1] + 1}] Valid Loss: {np.round(fold_valid_loss, 4)}')
                print(f'Fold[{fold + 1}/{fold_list[-1] + 1}] Pearson Score: {np.round(fold_score, 4)}')

                if cfg.swa:
                    update_bn(loader_train, swa_model)  # Stochastic Weight Averaging
                    fold_swa_loss, fold_swa_score = swa_valid(
                        cfg,
                        loader_valid,
                        swa_model,
                        criterion,
                        valid_labels,
                    )
                    fold_swa_loss = fold_swa_loss.detach().cpu().numpy()
                    fold_swa_loss = np.mean(fold_swa_loss)
                    fold_swa_score = np.mean(fold_swa_score)

                    wandb.log({
                        f'<Fold{fold + 1}> SWA Valid Loss': fold_swa_loss,
                        f'<Fold{fold + 1}> SWA Pearson_Score': fold_swa_score,
                    })

                    print(f'Fold[{fold + 1}/{fold_list[-1] + 1}] SWA Loss: {np.round(fold_swa_loss, 4)}')
                    print(f'Fold[{fold + 1}/{fold_list[-1] + 1}] SWA Score: {np.round(fold_swa_score, 4)}')

                if val_score_max <= fold_swa_score:
                    print(f'[Update] Valid Score : ({val_score_max:.4f} => {fold_swa_score:.4f}) Save Parameter')
                    print(f'Best Score: {fold_swa_score}')
                    torch.save(model.state_dict(),
                               f'SWA_Ver2-3_Token_Classification_Fold{fold}_DeBERTa_V3_Large.pth')
                    val_score_max = fold_score

                del fold_swa_loss
                gc.collect()
                torch.cuda.empty_cache()

            wandb.finish()

    def _save_checkpoint(self, epoch: int, save_best=False) -> None
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def save_parameter(self) -> None:
        """
        Save model's parameter which recorded best cross validation score
        """
        best_path = self.checkpoint_dir
        best_name = type(self.model).__name__

        torch.save(
            self.model.state_dict(),
            f'{best_path}/{best_name}.pth'
        )

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
