import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel
from model.model_utils import freeze, reinit_topk


class FBPModel(nn.Module):
    """ Model class for Baseline Pipeline """
    def __init__(self, cfg):
        super().__init__()
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 6)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        self._init_weights(self.fc)

        if cfg.reinit:
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model)

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        embedding = self.pooling(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        logit = self.fc(embedding)
        return logit


class MPLModel(nn.Module):
    """ Teacher model for Meta Pseudo Label Pipeline """
    def __init__(self, cfg):
        super().__init__()
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 6)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        self._init_weights(self.fc)

        if cfg.reinit:
            # reinit_topk(self.model, cfg.num_reinit)
            self.reinit_topk_layers()

        if cfg.freeze:
            freeze(self.model)

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module):
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def reinit_topk_layers(self):
        """
        Re-initialize the last-k transformer layers.
        Args:
            model: The target transformer model.
            num_layers: The number of layers to be re-initialized.
        """
        self.model.encoder.layer[-self.cfg.num_reinit:].apply(self.model._init_weights)  # model class에 있는거

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        embedding = self.pooling(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        logit = self.fc(embedding)
        return logit
