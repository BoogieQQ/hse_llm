base = {
    'batch_size': 16, 'accumulation': 1, 'lr': 3e-4,
    'scheduler': 'constant', 'optim': 'adamw_torch_fused',
    'compile': False, 'dtype': 'bf16', 'tf32': True,
}

variants = {
    'baseline': {},
    'batch4_acc4': {'batch_size': 4, 'accumulation': 4},
    'batch8_acc2': {'batch_size': 8, 'accumulation': 2},
    'lr5e5': {'lr': 5e-5},
    'lr1e3': {'lr': 1e-3},
    'cosine': {'scheduler': 'cosine'},
    'linear': {'scheduler': 'linear'},
    'compiled': {'compile': True},
    'adamw': {'optim': 'adamw_torch'},
    'tf32_off': {'tf32': False},
    'fp16': {'dtype': 'fp16'},
}
