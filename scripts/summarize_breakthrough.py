#!/usr/bin/env python3
"""Tabulate every finished worker result under given roots, grouped by arm directory name."""
import json, re, sys
from collections import defaultdict
from pathlib import Path

rows = defaultdict(dict)
for root in sys.argv[1:]:
    for path in sorted(Path(root).glob("seed*/result.json")):
        seed, name = re.match(r"seed(\d+)_(.*)", path.parent.name).groups()
        rows[name][int(seed)] = json.loads(path.read_text())
print(f"{'arm':34}{'seeds':>10}{'acc%':>26}{'mean':>8}{'sparsity':>9}{'MAC-':>8}  q0 / zero layers")
for name, seeds in rows.items():
    acc = [seeds[s]["final_validation"]*100 for s in sorted(seeds)]
    pick = lambda key: sum(seeds[s].get(key, 0) for s in seeds)/len(seeds)
    print(f"{name:34}{','.join(map(str, sorted(seeds))):>10}{' '.join(f'{a:.2f}' for a in acc):>26}{sum(acc)/len(acc):8.3f}"
          f"{pick('sparsity')*100:9.2f}{pick('mac_reduction')*100:8.2f}  {[seeds[s].get('q_zero') for s in sorted(seeds)]} {sum(len(seeds[s].get('zero_layers', [])) for s in seeds)}")
