base = {'seconds': 300, 'attention': 'sdpa', 'changes': {}}

probes = {
    'batch16_acc4': {'seconds': 90, 'changes': {'per_device_train_batch_size': 16, 'gradient_accumulation_steps': 4}},
    'batch8_acc8': {'seconds': 90, 'changes': {'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 8}},
    'checkpoint': {'seconds': 90, 'changes': {'gradient_checkpointing': True}},
    'flash': {'seconds': 90, 'attention': 'flash_attention_2'},
    'compile': {'seconds': 90, 'changes': {'torch_compile': True}},
    'packing': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'packing': True}},
    'padding_free': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'padding_free': True}},
    'liger': {'seconds': 90, 'changes': {'use_liger_kernel': True}},
    'offload': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'activation_offloading': True}},
    'flash_compile': {'seconds': 180, 'attention': 'flash_attention_2', 'changes': {'torch_compile': True}},
    'flash_liger': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'use_liger_kernel': True}},
    'flash_batch8': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 8}},
    'compile_extended': {'seconds': 180, 'changes': {'torch_compile': True}},
    'packing_batch8': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'packing': True, 'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 8}},
    'padding_free_liger': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'padding_free': True, 'use_liger_kernel': True}},
    'flash_packing_liger': {'seconds': 90, 'attention': 'flash_attention_2', 'changes': {'packing': True, 'use_liger_kernel': True}},
}
