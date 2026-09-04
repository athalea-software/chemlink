# Free-energy perturbation (FEP)

ChemLink includes a standalone preparation workflow for relative (RBFE) and
absolute (ABFE) binding free-energy calculations. It is separate from
conventional `pligand` molecular dynamics because it requires multiple
thermodynamic legs and alchemical lambda states.

## Initial command

```bash
chemlink dynamic fep \
  --protein protein.pdb \
  --ligand-a ligand_A.pdb \
  --ligand-b ligand_B.pdb \
  --charge-a 0 \
  --charge-b 0 \
  --fep-type auto \
  --replicas 3 \
  --lambda-windows 24 \
  --prepare-only \
  --run-name rbfe_A_B
```

PDB, PDBQT-formatted PDB, SDF, and MOL ligand inputs are accepted. FEP uses the
same charge/protonation resolver as `pligand`: PDB/PDBQT inputs are protonated
with Open Babel at `--ligand-ph` (7.4 by default), and `--charge-mode` controls
automatic, strict, or interactive resolution. `--charge-a` and `--charge-b`
remain explicit overrides. When PDB bond orders are ambiguous, provide
`--smiles-a` and `--smiles-b`, or use SDF input.

Before ACPYPE, ChemLink rejects odd-electron structures because AM1-BCC uses a
singlet calculation by default. This commonly identifies PDBQT conversions
whose implicit hydrogens were not restored.

## What preparation produces

- validated SDF representations of both ligands;
- an atom-mapping and maximum-common-substructure report;
- `complex` and `solvent` thermodynamic legs;
- one directory and `production.mdp` per replica and lambda state;
- dependency and transformation reports.

For ABFE, preparation also creates:

- parameterization manifests for ACPYPE/GAFF2;
- definitions for the four systems (ligands A and B, each in complex and
  solvent);
- automatically selected protein/ligand Boresch anchors and their initial
  geometry;
- staged restraint, electrostatic, and van der Waals lambda schedules;
- `pilot_plan.json`, which defines the reduced validation run required before
  production.

Before `pdb2gmx`, the executable setup uses PDBFixer to rebuild missing heavy
atoms in existing protein residues. It does not invent missing sequence
segments and does not add hydrogens (GROMACS adds them according to the chosen
force field). All rebuilt atoms are listed in `common/protein_repair.json`.

Inspect `common/abfe_restraints.json` before running dynamics. An
`anchors_selected_pending_short_md` status means that the initial geometry is
acceptable, but anchor stability must still be verified with the short pilot.
`manual_anchor_review_required` means the anchors must be corrected first.

In `auto` mode, ChemLink selects RBFE when the ligands have equal net charge and
at least 50% common-heavy-atom coverage. Otherwise it prepares two independent
ABFE calculations, one per ligand. Use `--fep-type rbfe` or `--fep-type abfe` to
make the selection explicit.

## Execution status

`--prepare-only` generates validated inputs and execution manifests. Omitting
that option for ABFE additionally runs ACPYPE and GROMACS to build, solvate, and
neutralize the four systems. Every command and its output are recorded in
`setup_commands.log`; completion is summarized in `abfe_setup.json`.

Full production execution will require pmx for RBFE hybrid topologies and alchemlyb/pymbar for
BAR, MBAR, and TI analysis. ABFE additionally requires producing the GROMACS
topologies, validating orientational restraints with the pilot, and applying
the standard-state correction. Until those execution stages are connected,
The ABFE pilot remains locked after system construction until the six Boresch
restraints have been mapped to the final solvated atom indices and their
standard-state correction has been calculated. A `prepared` report does not
mean that molecular dynamics has already run.

`install.sh` and the Docker image install `alchemlyb` and `pymbar` in the main
`bio` environment. Re-running `install.sh` upgrades an existing environment as
well as configuring new installations.
