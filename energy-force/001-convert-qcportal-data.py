import descent
import openmm.unit
from openff.qcsubmit.results import OptimizationResultCollection

# based on convert-espaloma-data, it looks like we need to return a dict of
# smiles, coords, energy, and forces


HARTEE_TO_KCAL = (
    1.0 * openmm.unit.hartree * openmm.unit.AVOGADRO_CONSTANT_NA
).value_in_unit(openmm.unit.kilocalorie_per_mole)

BOHR_TO_ANGSTROM = (1.0 * openmm.unit.bohr).value_in_unit(openmm.unit.angstrom)

# according to descent.targets.energy.Entry, we want coords in Å, energy in
# kcal/mol, and forces in kcal/mol/Å


ds = OptimizationResultCollection.parse_file("combined-opt.json")

entries = list()
records = ds.to_records()
for rec, mol in records:
    smiles = mol.to_smiles(mapped=True, isomeric=True)
    assert len(mol.conformers) == 1
    coords = mol.conformers[0]
    energy = rec.energies[-1]
    grad = rec.trajectory[-1].properties["current gradient"]
    entries.append(
        dict(
            smiles=smiles,
            coords=coords.magnitude,  # already in ang
            energy=[energy * HARTEE_TO_KCAL],
            forces=[g * HARTEE_TO_KCAL / BOHR_TO_ANGSTROM for g in grad],
        )
    )

dataset = descent.targets.energy.create_dataset(entries)
dataset.save_to_disk("test.out")
