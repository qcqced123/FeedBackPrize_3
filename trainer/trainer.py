import numpy as np
import torch

from functools import reduce
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from tqdm.auto import tqdm

from base import BaseTrainer
from utils import inf_loop, MetricTracker
from trainer_utils import get_optimizer_grouped_parameters


class Input(object):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tokenizer = self.cfg.tokenizer
        self.df = df  # return dataset
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score) {self.cfg.model_name}_state_dict.pth'  # checkpoint

    # LLRD
    def get_optimizer_grouped_parameters(self, model, layerwise_lr, layerwise_weight_decay, layerwise_lr_decay):
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if "model" not in n],
                                         "weight_decay": 0.0,
                                         "lr": layerwise_lr,
                                         }, ]
        # initialize lrs for every layer
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = layerwise_lr
        for layer in layers:
            optimizer_grouped_parameters += [
                {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": layerwise_weight_decay,
                 "lr": lr,
                 },
                {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0,
                 "lr": lr,
                 }, ]
            lr *= layerwise_lr_decay
        return optimizer_grouped_parameters


    def make_batch(self, fold: int):
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Custom Dataset
        train_dataset = UPPPMDataset(self.cfg, train)
        valid_dataset = UPPPMDataset(self.cfg, valid, is_valid=True)
        valid_labels = valid['scores'].explode().to_numpy()

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return loader_train, loader_valid, train, valid, valid_labels

    def model_setting(self):
        """
        [model]
        1) Re-Initialze Weights of Encoder
           - DeBERTa => Last Two Layers == EMD
        2) SWA
           - original model => to.device
           - after calculate, update swa_model
        """
        model = UPPPMModel(self.cfg, n_vocabs=len(self.tokenizer))
        # model.load_state_dict(torch.load('Token_Classification_Fold0_DeBERTa_V3_Large.pth'))
        model.to(self.cfg.device)
        # SWA: Stochastic Weighted Averaging
        if self.cfg.swa:
            swa_model = AveragedModel(model)
        else:
            swa_model = 'none'

        # Setting Loss_Function
        # Because we don't need to calculate none-target token: Output will be same shape as input
        if self.cfg.loss_fn == 'BCE':
            criterion = nn.BCEWithLogitsLoss(reduction='none')
        if self.cfg.loss_fn == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        if self.cfg.loss_fn == 'pearson':
            criterion = PearsonLoss()
        if self.cfg.loss_fn == 'RMSE':
            criterion = RMSELoss()

        # optimizer
        grouped_optimizer_params = self.get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )

        optimizer = AdamW(
            grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam)

        return model, swa_model, criterion, optimizer, self.save_parameter


class MPLInput(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class FBPTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

    # Step 3.1 Train & Validation Function
    def train_fn(cfg,
                 loader_train,
                 loader_valid,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 valid,
                 valid_labels,
                 epoch,
                 swa_model=None,
                 swa_start=None,
                 swa_scheduler=None, ):
        # Train Stages
        # torch.amp.gradscaler
        awp = AWP(model, criterion, optimizer, cfg.awp, adv_lr=cfg.awp_lr, adv_eps=cfg.awp_eps)
        if cfg.amp_scaler:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        global_step, score_list = 0, []  # All Fold's average of mean F2-Score
        losses = AverageMeter()
        model.train()
        for step, (inputs, _, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(cfg.device)  # train to gpu
            labels = labels.to(cfg.device)  # label to gpu
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=cfg.amp_scaler):
                preds = model(inputs)
                loss = criterion(preds.view(-1, 1), labels.view(-1, 1))
                mask = (labels.view(-1, 1) != -1)
                loss = torch.masked_select(loss, mask).mean()  # reduction = mean
                losses.update(loss, batch_size)
                """
                [gradient_accumlation]
                - GPU VRAM OVER 문제해결을 위해 사용
                - epoch이 사용자 지정 에폭 횟수를 넘을 때까지 Backward 하지 않고 그라디언트 축적
                - 지정 epoch 넘어가면 한 번에 Backward
                """
                if cfg.n_gradient_accumulation_steps > 1:
                    loss = loss / cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            """
            [Adversarial Weight Training]
            """
            if cfg.awp and epoch >= cfg.nth_awp_start_epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()
            """
            1) Clipping Gradient && Gradient Accumlation
            2) Stochastic Weight Averaging
            """
            if cfg.clipping_grad and (
                    (step + 1) % cfg.n_gradient_accumulation_steps == 0 or cfg.n_gradient_accumulation_steps == 1):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                global_step += 1
                scheduler.step()

        # Validation Stage
        preds_list, label_list = [], []
        valid_losses = AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, target_masks, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(cfg.device)
                labels = labels.to(cfg.device)
                batch_size = labels.size(0)
                preds = model(inputs)
                valid_loss = criterion(preds.view(-1, 1), labels.view(-1, 1))
                mask = (labels.view(-1, 1) != -1)
                valid_loss = torch.masked_select(valid_loss, mask).mean()
                valid_losses.update(valid_loss, batch_size)

                y_preds = preds.sigmoid().to('cpu').numpy()

                anchorwise_preds = []
                for pred, target_mask, in zip(y_preds, target_masks):
                    prev_i = -1
                    targetwise_pred_scores = []
                    for i, (p, tm) in enumerate(zip(pred, target_mask)):
                        if tm != 0:
                            if i - 1 == prev_i:
                                targetwise_pred_scores[-1].append(p)
                            else:
                                targetwise_pred_scores.append([p])
                            prev_i = i
                    for targetwise_pred_score in targetwise_pred_scores:
                        anchorwise_preds.append(np.mean(targetwise_pred_score))
                preds_list.append(anchorwise_preds)
        # error_list = [[i, preds_list.index(i)] for i in preds_list if i == 'nan' or i == float('inf')]
        # print(error_list)
        epoch_score = pearson_score(valid_labels, np.array(reduce(lambda a, b: a + b, preds_list)))
        return losses.avg, valid_losses.avg, epoch_score, grad_norm, scheduler.get_lr()[0]

    def swa_fn(self, cfg, loader_valid, swa_model, criterion, valid_labels):
        swa_preds_list, swa_label_list = [], []
        swa_model.eval()
        swa_valid_losses = AverageMeter()

        with torch.no_grad():
            for step, (swa_inputs, target_masks, swa_labels) in enumerate(tqdm(loader_valid)):
                swa_inputs = collate(swa_inputs)

                for k, v in swa_inputs.items():
                    swa_inputs[k] = v.to(cfg.device)

                swa_labels = swa_labels.to(cfg.device)
                batch_size = swa_labels.size(0)

                swa_preds = swa_model(swa_inputs)

                swa_valid_loss = criterion(swa_preds.view(-1, 1), swa_labels.view(-1, 1))
                mask = (swa_labels.view(-1, 1) != -1)
                swa_valid_loss = torch.masked_select(swa_valid_loss, mask)
                swa_valid_loss = swa_valid_loss.mean()
                swa_valid_losses.update(swa_valid_loss, batch_size)

                swa_y_preds = swa_preds.sigmoid().to('cpu').numpy()

                anchorwise_preds = []
                for pred, target_mask, in zip(swa_y_preds, target_masks):
                    prev_i = -1
                    targetwise_pred_scores = []
                    for i, (p, tm) in enumerate(zip(pred, target_mask)):
                        if tm != 0:
                            if i - 1 == prev_i:
                                targetwise_pred_scores[-1].append(p)
                            else:
                                targetwise_pred_scores.append([p])
                            prev_i = i
                    for targetwise_pred_score in targetwise_pred_scores:
                        anchorwise_preds.append(np.mean(targetwise_pred_score))

                swa_preds_list.append(anchorwise_preds)
        swa_valid_score = pearson_score(valid_labels, np.array(reduce(lambda a, b: a + b, swa_preds_list)))
        del swa_preds_list, swa_y_preds, swa_labels, anchorwise_preds
        gc.collect()
        torch.cuda.empty_cache()
        return swa_valid_losses.avg, swa_valid_score


class MPLTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

    def train_fn(self):
        return

    def swa_fn(self):
        return
