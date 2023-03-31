import torch
import torch.nn as nn


def get_optimizer_grouped_parameters(self, model, layerwise_lr, layerwise_weight_decay, layerwise_lr_decay):
    """
    Grouped Version: Layer-wise learning rate decay
    """
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


def collate(inputs):
    """
    Descending sort inputs by length of sequence
    """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs

