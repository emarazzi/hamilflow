"""Write the k-fold ``dataset_split_{1..k}.json`` files for training a DeepH ensemble.

Fold i validates on group i and trains on the other k - 1 groups; point each model's
``process.train.dataloader.dataset_split_json`` in train.toml to one of these files.

Structures are grouped into families, and the members of a family are spread over different
validate sets, so every ensemble member sees every family. The family of a structure is given
by GROUP_KEY, a function ``name -> family label`` (see the options below). Names that share no
family with any other name end up in a balanced random split.

Active learning: for cycle 2 onwards, set PREVIOUS_CYCLE_DIR to the folder holding the previous
cycle's splits. Every structure already in them keeps its assignment; only the structures that
are new in INPUTS_DIR are dealt into the k groups (smallest folds first).

Input: INPUTS_DIR, the DeepH ``dft/`` folder with one subfolder per structure; the subfolder
names are the structure names. Outputs: ``dataset_split_{1..k}.json`` in CYCLE_DIR.

Next step: write_train_tomls.py.

Run with:  python make_kfold_splits.py
"""

import json
from collections import Counter
from pathlib import Path

from hamilflow.training.dataset_split import parent_of, write_kfold_splits

# --- User settings -----------------------------------------------------------
INPUTS_DIR = Path("/path/to/your/training/inputs/dft")
CYCLE_DIR = Path("/path/to/your/models/cycle2")
PREVIOUS_CYCLE_DIR = None  # e.g. Path("/path/to/your/models/cycle1"); None = first AL cycle
K = 5
SEED = 137

# Family of each structure; pick one option:
#   parent_of   cuts at the last "-": "tMoS2Bilayer_n1m5Ang6-003" and "tMoS2Bilayer_n1m5Ang6-init"
#               -> "tMoS2Bilayer_n1m5Ang6" (names without a "-" are their own family)
#   None        no families: plain random split
#   your own function of the name, e.g. for "Si64_rattle07" -> "Si64":
#       GROUP_KEY = lambda name: name.rsplit("_rattle", 1)[0]
#   or a lookup table {name: family} from a file (unlisted names are their own family):
#       families = json.load(open("/path/to/your/families.json"))
#       GROUP_KEY = lambda name: families.get(name, name)
GROUP_KEY = parent_of
# -----------------------------------------------------------------------------

names = sorted(p.name for p in INPUTS_DIR.iterdir() if p.is_dir())
paths = write_kfold_splits(
    names, K, CYCLE_DIR, previous_dir=PREVIOUS_CYCLE_DIR, group_key=GROUP_KEY, seed=SEED
)

key = GROUP_KEY or (lambda name: name)
print(f"{len(names)} structures in {len({key(n) for n in names})} families -> {CYCLE_DIR}")
for path in paths:
    split = json.loads(path.read_text())
    worst = max(Counter(key(n) for n in split["validate"]).values(), default=0)
    print(f"  {path.name}: {len(split['train'])} train, {len(split['validate'])} validate, "
          f"at most {worst} of one family in validate")
