import gc, transformers
import dataset_class.dataclass as dataset_class
import model.loss as model_loss
import model.model as model_arch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from dataset_class.text_preprocessing import *
from utils.helper import *
from trainer.trainer_utils import *
from model.metric import *


class FBPTrainer:
    def __init__(self, cfg, generator):
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.df = load_data('./dataset_class/data_folder/Base_Train/train_df.csv')
        self.tokenizer = self.cfg.tokenizer
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int):
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

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
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
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
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )

        swa_scheduler = get_swa_scheduler(self.cfg, optimizer)
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        awp = None
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )

        return model, swa_model, criterion, optimizer, lr_scheduler, swa_scheduler, awp, self.save_parameter

    # Step 3.1 Train & Validation Function
    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None,
                 swa_model=None, swa_start=None, swa_scheduler=None,):
        """ Training Function """
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        global_step, score_list = 0, []  # All Fold's average of mean F2-Score
        losses = AverageMeter()
        model.train()

        for step, (inputs, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # label to gpu
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                preds = model(inputs)
                loss = criterion(preds, labels)
                losses.update(loss, batch_size)

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()

                if epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                global_step += 1
                scheduler.step()
        train_loss = losses.avg.detach().cpu().numpy()
        grad_norm = grad_norm.detach().cpu().numpy()
        return train_loss, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, criterion):
        """ Validation Function """
        valid_losses = AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                batch_size = labels.size(0)
                preds = model(inputs)
                valid_loss = criterion(preds, labels)
                valid_losses.update(valid_loss, batch_size)
        valid_loss = valid_losses.avg.detach().cpu().numpy()
        return valid_loss

    @torch.no_grad()
    def swa_fn(self, loader_valid, swa_model, criterion):
        """ Validation Function by Stochastic Weight Averaging """
        swa_model.eval()
        swa_valid_losses = AverageMeter()

        with torch.no_grad():
            for step, (swa_inputs, swa_labels) in enumerate(tqdm(loader_valid)):
                swa_inputs = collate(swa_inputs)

                for k, v in swa_inputs.items():
                    swa_inputs[k] = v.to(self.cfg.device)

                swa_labels = swa_labels.to(self.cfg.device)
                batch_size = swa_labels.size(0)
                swa_preds = swa_model(swa_inputs)
                swa_valid_loss = criterion(swa_preds, swa_labels)
                swa_valid_losses.update(swa_valid_loss, batch_size)
        swa_loss = swa_valid_losses.avg.detach().cpu().numpy()
        return swa_loss


class MPLTrainer():
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

    def train_fn(self):
        return

    def swa_fn(self):
        return
