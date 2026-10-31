import subprocess
import sys
from pathlib import Path

from train import EXPERIMENTS, ROOT


(ROOT / "logs").mkdir(exist_ok=True)

for name in EXPERIMENTS:
    if (ROOT / "results" / name / "summary.json").exists():
        continue
    print(f"Запуск {name}", flush=True)
    with (ROOT / "logs" / f"{name}.log").open("w") as log:
        subprocess.run([sys.executable, "-u", str(ROOT / "train.py"), name], stdout=log, stderr=subprocess.STDOUT, check=True)
