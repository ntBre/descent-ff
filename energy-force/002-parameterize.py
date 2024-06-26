"""Apply OpenFF 2.1.0 parameters to each unique molecule in the data set."""

import functools
import json
import multiprocessing
import pathlib

import openff.interchange
import openff.toolkit
import smee
import smee.converters
import torch
import tqdm


def build_interchange(
    smiles: str, force_field_paths: tuple[str, ...]
) -> openff.interchange.Interchange | None:
    try:
        return openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField(*force_field_paths),
            openff.toolkit.Molecule.from_mapped_smiles(
                smiles, allow_undefined_stereo=True
            ).to_topology(),
        )
    except BaseException as e:
        print(f"failed to parameterize {smiles}: {e}")
        return None


def apply_parameters(
    unique_smiles: list[str], *force_field_paths: str
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    build_interchange_fn = functools.partial(
        build_interchange, force_field_paths=force_field_paths
    )

    with multiprocessing.get_context("spawn").Pool() as pool:
        interchanges = list(
            pool.imap(
                build_interchange_fn,
                tqdm.tqdm(
                    unique_smiles,
                    total=len(unique_smiles),
                    desc="building interchanges",
                ),
            )
        )

    unique_smiles, interchanges = zip(
        *[(s, i) for s, i in zip(unique_smiles, interchanges) if i is not None]
    )

    force_field, topologies = smee.converters.convert_interchange(interchanges)

    return force_field, {
        smiles: topology for smiles, topology in zip(unique_smiles, topologies)
    }


def main():
    force_field_paths = ["openff-2.1.0.offxml"]

    smiles: list[str] = json.loads((pathlib.Path("smiles.json")).read_text())

    unique_smiles: set[str] = set(smiles)

    print(f"N smiles={len(unique_smiles)}", flush=True)

    unique_smiles = sorted(unique_smiles)

    force_field, topologies = apply_parameters(
        unique_smiles, *force_field_paths
    )
    torch.save(
        (force_field, topologies), pathlib.Path("outputs", "openff-2.1.0.pt")
    )


if __name__ == "__main__":
    main()
