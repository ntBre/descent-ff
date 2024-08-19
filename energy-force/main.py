import argparse
import functools
import itertools
import json
import logging
import os
import resource
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import datasets
import datasets.table
import openmm.unit
import pyarrow
from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.utils import _CachedPortalClient, portal_client_manager
from openff.toolkit import Molecule
from qcportal.optimization import OptimizationRecord
from tqdm import tqdm
from yaml import Loader, load

from _filter import main as step3
from convert import main as step5
from parameterize import main as step2
from train import main as step4

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s")
logger.setLevel(logging.INFO)

HARTEE_TO_KCAL = (
    1.0 * openmm.unit.hartree * openmm.unit.AVOGADRO_CONSTANT_NA
).value_in_unit(openmm.unit.kilocalorie_per_mole)

BOHR_TO_ANGSTROM = (1.0 * openmm.unit.bohr).value_in_unit(openmm.unit.angstrom)

# copied from descent.targets.energy
DATA_SCHEMA = pyarrow.schema(
    [
        ("smiles", pyarrow.string()),
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("forces", pyarrow.list_(pyarrow.float64())),
    ]
)


@dataclass
class Dataset:
    kind: str
    path: str


@dataclass
class Config:
    datasets: list[Dataset]
    table_path: str
    smiles_path: str
    force_fields: list[str]
    torch_path: str
    filtered_path: str
    world_size: int
    output_path: str

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            d = load(f, Loader=Loader)
            tmp = cls(**d)
            # tidy up the types before returning
            tmp.datasets = [Dataset(**d) for d in tmp.datasets]
            tmp.world_size = int(tmp.world_size)
            return tmp


def convert_torsion_data(
    td: TorsionDriveResultCollection,
) -> Iterable[tuple[OptimizationRecord, Molecule]]:
    """Convert a ``TorsionDriveResultCollection`` into a sequence of
    (``OptimizationRecord``, ``Molecule``) Pairs just like
    ``OptimizationResultCollection.to_records``.

    """
    for rec, mol in td.to_records():
        opts: dict[tuple[int], OptimizationRecord] = rec.minimum_optimizations

        grid_to_conf = {
            grid: conf
            for grid, conf in zip(mol.properties["grid_ids"], mol._conformers)
        }

        # TorsionDriveResultCollection.to_records already packs all of the
        # qcportal geometries into mol.conformers. all I need to do is copy the
        # molecule and give it a single conformer

        for k, opt in opts.items():
            mol = Molecule(mol)  # this should use the copy initializer
            mol._conformers = []
            mol.add_conformer(grid_to_conf[k])
            yield opt, mol


COUNT = 0


def process_entry(rec, mol) -> dict[str, Any] | None:
    """Turn a single rec, mol pair from
    `OptimizationResultCollection.to_records` into the dict expected by
    descent"""
    smiles = mol.to_smiles(mapped=True, isomeric=True)
    assert len(mol.conformers) == 1
    coords = mol.conformers[0]
    energy = rec.energies[-1]
    grad = rec.trajectory[-1].properties.get("current gradient", None)
    if grad is None:
        return None
    global COUNT
    COUNT += 1
    if COUNT % 100 == 0:
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tqdm.write(f"Current mem: {mem}")
    return dict(
        smiles=smiles,
        coords=coords.flatten().tolist().magnitude,  # already in ang
        energy=[energy * HARTEE_TO_KCAL],
        forces=[g * HARTEE_TO_KCAL / BOHR_TO_ANGSTROM for g in grad],
    )


def batched(iterable, n):
    "rough equivalent to itertools.batched in 3.12, from the docs"
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def create_batched_dataset(
    entries: Iterable[dict[str, Any]], chunksize=128
) -> Iterable[pyarrow.RecordBatch]:
    """Batched version of descent.targets.energy.create_dataset"""
    for batch in batched(entries, chunksize):
        yield pyarrow.RecordBatch.from_pylist(batch, schema=DATA_SCHEMA)


def load_batches(d, n):
    "Load ``n`` sequential batch files from directory ``d``."
    for i in range(n):
        logger.info(f"loading batch {i}")
        path = os.path.join(d, f"batch{i}.json")
        with open(path) as inp:
            entries = json.load(inp)
            for entry in entries:
                yield entry


def step1(datasets_: list[Dataset], output_path: str, smiles_path: str):
    """Load data from qcsubmit result collections and store the result to disk.

    The records themselves are written to a ``datasets.Dataset`` and saved to
    ``output_path``, while the unique SMILES in the ``datasets.Dataset`` are
    serialized to JSON and written to ``smiles_path``.
    """
    client = _CachedPortalClient(
        "https://api.qcarchive.molssi.org:443", ".cache"
    )
    total_results = 0
    iters = list()
    with portal_client_manager(lambda _: client):
        for d in datasets_:
            match d.kind:
                case "opt":
                    ds = OptimizationResultCollection.parse_file(d.path)
                    iters.append(ds.to_records())
                case "td":
                    ds = TorsionDriveResultCollection.parse_file(d.path)
                    iters.append(convert_torsion_data(ds))
                case k:
                    raise ValueError(f"unrecognized dataset kind: {k}")
            total_results += ds.n_results
    records_and_molecules = itertools.chain(*iters)

    # this part does use all the ram
    entries = (
        process_entry(rec, mol)
        for rec, mol in tqdm(
            records_and_molecules,
            desc="Processing records",
            total=total_results,
        )
    )
    entries = (e for e in entries if e is not None)

    with TemporaryDirectory() as d:
        for i, batch in enumerate(batched(entries, 1024)):
            path = os.path.join(d, f"batch{i}.json")
            logger.info(f"writing batch {i}")
            with open(path, "w") as out:
                json.dump(batch, out)

        f = functools.partial(load_batches, d, i)
        dataset = datasets.Dataset.from_generator(f)

    dataset.set_format("torch")

    logger.info("writing to disk")
    path = Path(output_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)

    unique_smiles = dataset.unique("smiles")
    with open(smiles_path, "w") as out:
        json.dump(unique_smiles, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    logger.info(f"loading config from {args.config_file}")
    config = Config.from_file(args.config_file)

    logger.info("starting step 1")
    step1(config.datasets, config.table_path, config.smiles_path)

    logger.info("starting step 2")
    step2(config.force_fields, config.smiles_path, config.torch_path)

    logger.info("starting step 3")
    step3(config.table_path, config.torch_path, config.filtered_path)

    logger.info("starting step 4")
    out_ff = step4(config.world_size, config.torch_path, config.filtered_path)

    logger.info("starting step 5")
    step5(config.force_fields, out_ff, config.output_path)

    logger.info("finished")


if __name__ == "__main__":
    main()
