import os
from pathlib import Path
import subprocess

from experiments import base, variants

root = Path(__file__).resolve().parent
os.chdir(root)
python = str(root / '.venv/bin/python')
Path('logs').mkdir(exist_ok=True)

def run(name, args):
    print(f'Запуск: {name}', flush=True)
    with open(f'logs/{name}.log', 'w') as log:
        subprocess.run([python, '-u', 'solution.py'] + args,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


run('prepare', ['prepare', '--num-proc', '8'])
subprocess.run([python, 'data_stats.py'], check=True)

for name, changes in variants.items():
    output = Path('results') / name
    if (output / 'final_metrics.json').exists() and (output / 'generations.json').exists():
        continue
    config = base | changes
    args = ['train', '--name', name]
    for key, value in config.items():
        if key == 'compile':
            if value:
                args.append('--compile')
        elif key == 'tf32':
            if not value:
                args.append('--no-tf32')
        else:
            args += ['--' + key.replace('_', '-'), str(value)]
    if name == 'baseline':
        args.append('--initial')
    run(name, args)

print('Эксперименты завершены', flush=True)
