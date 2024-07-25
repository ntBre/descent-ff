"""Filter any molecules that are not parameterizable by the OpenFF 2.1.0 force
field.

"""

from dataclasses import dataclass

import datasets
import descent.targets.energy
import torch


@dataclass
class Config:
    dataset: str  # path to dataset directory


def main(table_path, torch_path, filtered_path):
    dataset = datasets.Dataset.load_from_disk(table_path)
    unique_smiles = descent.targets.energy.extract_smiles(dataset)

    force_field, topologies = torch.load(torch_path)
    topologies = {k: v for k, v in topologies.items() if k in unique_smiles}

    dataset_size = len(dataset)
    dataset = dataset.filter(lambda d: d["smiles"] in topologies)
    print(f"removed non-parameterizable: {dataset_size} -> {len(dataset)}")

    dataset.save_to_disk(filtered_path)


if __name__ == "__main__":
    conf = Config(dataset="test.out")
    main(conf.dataset, "outputs/openff-2.1.0.pt", "filtered.out")
