import itertools
import logging
from typing import Any, Iterable

import datasets
import datasets.table
import openmm.unit
import pyarrow
from descent.targets.energy import DATA_SCHEMA
from openff.qcsubmit.results import OptimizationResultCollection
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s")
logger.setLevel(logging.INFO)

# based on convert-espaloma-data, it looks like we need to return a dict of
# smiles, coords, energy, and forces


HARTEE_TO_KCAL = (
    1.0 * openmm.unit.hartree * openmm.unit.AVOGADRO_CONSTANT_NA
).value_in_unit(openmm.unit.kilocalorie_per_mole)

BOHR_TO_ANGSTROM = (1.0 * openmm.unit.bohr).value_in_unit(openmm.unit.angstrom)

# according to descent.targets.energy.Entry, we want coords in Å, energy in
# kcal/mol, and forces in kcal/mol/Å

logger.info("loading result collection")

ds = OptimizationResultCollection.parse_file("combined-opt.json")


def process_entry(rec, mol):
    """Turn a single rec, mol pair from
    `OptimizationResultCollection.to_records` into the dict expected by
    descent"""
    smiles = mol.to_smiles(mapped=True, isomeric=True)
    assert len(mol.conformers) == 1
    coords = mol.conformers[0]
    energy = rec.energies[-1]
    grad = rec.trajectory[-1].properties["current gradient"]
    return dict(
        smiles=smiles,
        coords=coords.flatten().magnitude,  # already in ang
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


table = pyarrow.Table.from_batches(
    create_batched_dataset(
        (
            process_entry(rec, mol)
            for rec, mol in tqdm(ds.to_records(), desc="Processing records")
        )
    ),
    schema=DATA_SCHEMA,
)
dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
dataset.set_format("torch")

logger.info("writing to disk")
dataset.save_to_disk("test.out")
