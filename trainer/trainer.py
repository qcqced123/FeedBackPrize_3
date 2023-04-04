import numpy as np
import dataset_class.dataclass as dataset_class
import model.loss as model_loss
import model.model as model_arch
from functools import reduce
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from transformers import AdamW

from dataset_class.text_preprocessing import *
from utils.helper import *
from trainer_utils import *
from model.metric import *

class FBPTrainer:
    def __init__(self, cfg, generator):
        self.cfg = cfg
        self.model_name = self.cfg.backbone.split('/')[1]
        self.generator = generator
        self.df = text_preprocess(load_data('../data/FB3_Dataset/train.csv'), self.cfg)
        self.tokenizer = self.cfg.tokenizer
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int):
        """ df를 주어진 파이프라인에 맞게 전처리하는 구조로 바꾸자 """
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)
        valid_labels = valid.iloc[:, 2:8].values

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.tokenizer, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.tokenizer, valid)

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
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
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))

        model.to(self.cfg.device)
        swa_model = AveragedModel(model)
        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        grouped_optimizer_params = get_optimizer_grouped_parameters(
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

    # Step 3.1 Train & Validation Function
    def train_fn(self, loader_train, loader_valid, model, criterion, optimizer, scheduler, valid,
                 valid_labels, epoch, swa_model=None, swa_start=None, swa_scheduler=None,):
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        global_step, score_list = 0, []  # All Fold's average of mean F2-Score
        losses = AverageMeter()
        model.train()
        for step, (inputs, _, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # label to gpu
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
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
                if self.cfg.n_gradient_accumulation_steps > 1:
                    loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            """
            [Adversarial Weight Training]
            """
            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()

            """
            1) Clipping Gradient && Gradient Accumlation
            2) Stochastic Weight Averaging
            """
            if self.cfg.clipping_grad and ((step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm
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
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
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


class MPLTrainer():
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

    def train_fn(self):
        return

    def swa_fn(self):
        return
