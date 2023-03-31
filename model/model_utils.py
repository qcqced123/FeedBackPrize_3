from torch import Tensor


def freeze(module) -> None:
    """
    Freezes module's parameters.
    freezing embeddings and first 2 layers of encoder
    [Example]
    1) freeze(model.embeddings)
    2) freeze(model.encoder.layer[:2])
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freezed_parameters(module) -> list[Tensor]:
    """
    Returns names of freezed parameters of the given module.

    [Example]
    freezed_parameters = get_freezed_parameters(model)
    """
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)

    return freezed_parameters


def _init_weight(module) -> None:
    """
    Initializes weights of the given module.
    """


def reinit_topk(module, topk: int = 2) -> None:
    """
    Reinitializes topk parameters of the given module.
    """
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            if topk > 0:
                _init_weight(parameter)
                topk -= 1
            else:
                break


