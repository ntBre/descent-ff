"""Filter any molecules that are not parameterizable by the OpenFF 2.1.0 force
field.

"""

from dataclasses import dataclass

import click
import datasets
import descent.targets.energy
import torch


@dataclass
class Config:
    dataset: str  # path to dataset directory


@click.command()
def main(nworkers):
    conf = Config(dataset="test.out")

    dataset = datasets.Dataset.load_from_disk(conf.dataset)
    unique_smiles = descent.targets.energy.extract_smiles(dataset)

    force_field, topologies = torch.load("outputs/openff-2.1.0.pt")
    topologies = {k: v for k, v in topologies.items() if k in unique_smiles}

    dataset_size = len(dataset)
    dataset = dataset.filter(lambda d: d["smiles"] in topologies)
    print(f"removed non-parameterizable: {dataset_size} -> {len(dataset)}")

    dataset.save_to_disk("filtered.out")


if __name__ == "__main__":
    main()
