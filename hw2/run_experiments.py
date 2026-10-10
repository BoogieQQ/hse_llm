import json
import os
import subprocess
import sys
from pathlib import Path

from experiments import base, probes


root = Path(__file__).resolve().parent
os.chdir(root)
variants = {'baseline': base, **probes}

if len(sys.argv) > 1:
    import train

    name = sys.argv[1]
    config = variants[name]
    train.TRAIN_SECONDS = config['seconds']
    train.ATTENTION_IMPLEMENTATION = config.get('attention', 'sdpa')
    train.TRAINING_CONFIG['packing'] = False
    train.TRAINING_CONFIG['use_liger_kernel'] = False
    train.TRAINING_CONFIG.update(config.get('changes', {}))
    train.TRAINING_CONFIG['output_dir'] = f'results/{name}'
    train.main()

    output = Path(train.TRAINING_CONFIG['output_dir'])
    (output / 'run_meta.json').write_text(json.dumps({
        'name': name,
        'duration_budget_seconds': config['seconds'],
        'attention': train.ATTENTION_IMPLEMENTATION,
        'changes': config.get('changes', {}),
        'training_config': train.TRAINING_CONFIG,
    }, indent=2) + '\n')
else:
    failed = []
    for name in variants:
        output = Path('results') / name

        if (output / 'summary.json').exists() and (output / 'run_meta.json').exists():
            continue

        if output.exists():
            print(f'Пропуск незавершённого {name}: сохранена папка {output}')
            continue

        print(f'Запуск: {name}')
        with (Path('logs') / f'{name}.log').open('w') as stream:
            subprocess.run([sys.executable, '-u', __file__, name],
                            stdout=stream, stderr=subprocess.STDOUT, check=True)
