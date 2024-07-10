# testing how to extract energies and gradients from TorsiondriveRecords. it
# appears that all I have to do is get the minimum_optimizations field from
# each TorsiondriveRecord and then pair that up with Molecules from somewhere.
# I can call final_molecule on each of the optimization records I get back.
# I'll then just need to peek at OptimizationResultCollection.to_records to see
# how to turn that into an OpenFF Molecule. the good news is that I wrote the
# code in qcsubmit to do just that.

from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.toolkit import Molecule
from qcportal.optimization.record_models import OptimizationRecord

td = "/home/brent/omsf/projects/valence-fitting/02_curate-data/datasets/combined-td.json"

td = TorsionDriveResultCollection.parse_file(td)
print(td.n_results)

k, v = next(iter(td.entries.items()))

td.entries[k] = v[:1]
print(td.n_results)

for rec, mol in td.to_records():
    energies: dict[tuple[int], float] = rec.final_energies
    opts: dict[tuple[int], OptimizationRecord] = rec.minimum_optimizations

    grid_to_conf = {
        grid: conf
        for grid, conf in zip(mol.properties["grid_ids"], mol._conformers)
    }

    # TorsionDriveResultCollection.to_records already packs all of the qcportal
    # geometries into mol.conformers. all I need to do is copy the molecule and
    # give it a single conformer

    records_and_molecules = []
    for k, e in energies.items():
        mol = Molecule(mol)  # this should use the copy initializer
        mol._conformers = []
        mol.add_conformer(grid_to_conf[k])
        records_and_molecules.append((opts[k], mol))
        assert mol.n_conformers == 1

# the end goal is a sequence of (rec, mol) pairs that can be used like (and
# chained onto) an OptimizationResultCollection.to_records output
