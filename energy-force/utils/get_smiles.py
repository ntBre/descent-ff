# recover from not saving the smiles to json in step 1

import json

from datasets import Dataset

ds = Dataset.load_from_disk("test.out")
smiles = ds.unique("smiles")
with open("smiles.json", "w") as out:
    json.dump(smiles, out)
