import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
# Step 1.1 Configuration Setting
"""
[Configuration]
    - Pooling: mean, attention, max, weightedlayer, concat (This Pipeline doesn't need to pooling)
    - Optimizer: AdamW, SWA
    - Scheduler: cosine, linear, cosine_annealing, linear_annealing
    - Clip_grad_norm, Gradient Checking: T/F
    - LLRD
    - Re-Init
    - AWP
"""
class CFG:
    """--------[Common]--------"""
    wandb, train, competition, seed, cfg_name = True, True, 'UPPPM', 42, 'CFG'
    device, gpu_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0
    num_workers = 0
    """ Mixed Precision, Gradient Check Point """
    amp_scaler = True
    gradient_checkpoint = True # save parameter
    output_dir = './output/'
    """ Clipping Grad Norm, Gradient Accumulation """
    clipping_grad = True # clip_grad_norm
    n_gradient_accumulation_steps = 1 # Gradient Accumulation
    max_grad_norm = n_gradient_accumulation_steps * 1000
    """ Model """
    model_name = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    pooling = 'attention'
    max_len = 512
    """ CV, Epoch, Batch Size """
    n_folds = 4
    epochs = 180
    batch_size = 64
    """ SWA, Loss, Optimizer, Scheduler """
    swa = True
    swa_start = int(epochs*0.75)
    swa_lr = 1e-4
    anneal_epochs = 4
    anneal_strategy = 'cos' # default = cos, available option: linear
    loss_fn = 'BCE'
    optimizer = 'AdamW' # options: SWA, AdamW
    weight_decay = 1e-2
    scheduler = 'cosine_annealing' # options: cosine, linear, cosine_annealing, linearannealing
    num_cycles = 0.5
#    num_warmup_steps = 0
    warmup_ratio = 0.1 # options: 0.05, 0.1
    batch_scheduler = True
    # encoder_lr = 5e-5
    # decoder_lr = 1e-5
    min_lr = 1e-7
    # eps = 1e-6
    betas = (0.9, 0.999)
    """ LLRD """
    llrd = True
    layerwise_lr = 5e-5
    layerwise_lr_decay = 0.9
    layerwise_weight_decay = 1e-2
    layerwise_adam_epsilon = 1e-6
    layerwise_use_bertadam = False
    """ Re-Init, AWP """
    reinit = True
    num_reinit = 5
    awp = False
    nth_awp_start_epoch = 10
    awp_eps = 1e-2
    awp_lr = 1e-4