# Free-energy perturbation (FEP)

ChemLink includes a standalone preparation workflow for relative (RBFE) and
absolute (ABFE) binding free-energy calculations. It is separate from
conventional `pligand` molecular dynamics because it requires multiple
thermodynamic legs and alchemical lambda states.

## Initial command

```bash
chemlink dynamic fep \
  -p protein.pdb \
  -a ligand_A.pdb \
  -b ligand_B.pdb \
  --charge-a 0 \
  --charge-b 0 \
  --fep-type auto \
  --replicas 3 \
  --lambda-windows 24 \
  --prepare-only \
  --run-name rbfe_A_B
```

`-p`/`-a`/`-b` are short aliases for `--protein`/`--ligand-a`/`--ligand-b`;
the long forms work the same.

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

Without `--prepare-only`, restraint mapping and `grompp` validation run
automatically as part of the original `chemlink dynamic fep` command, which
then continues straight into the pilot — topology validation, replica
equilibration, short MD at the pilot lambda subset, and the standard-state
correction (see "Running the whole workflow in one command" below; there is
no separate command for this step).

The pilot maps protein anchors by residue/atom identity, maps ligand anchors onto
their global complex indices, writes one distance, two angle, and three
dihedral Boresch interactions, and validates the coupled restraint state with
`grompp` (recorded in `boresch_mapping.json`). These bonded A/B interactions
are controlled with `bonded-lambdas`; GROMACS reserves `restraint-lambdas`
for dihedral restraints and pull-code restraints.

It then equilibrates every replica that isn't already equilibrated (see
below), and runs the pilot itself: short MD at the `pilot_plan.json` lambda
subset for **both** ligands' complex leg (not only ligand A, since the
standard-state correction is specific to each ligand's own anchor geometry),
seeded from each replica's NPT endpoint. The resulting trajectories are
re-read with `gmx traj` to recompute the six restraint coordinates per frame
and check they stay within the same bounds `select_boresch_anchors` already
validated at the start (15°-165° for both angles, 1.5-15 Å for the
distance), plus a per-frame dihedral-jump check. The standard-state
correction is computed analytically (Boresch et al., *J. Phys. Chem. B*
2003, 107, 9535, eq. 32; closed-form harmonic approximation) from each
ligand's restraint geometry and force constants — no additional simulation
is needed for that part. Results are written to `pilot_report.json`;
`fep_preflight.json["production_locked"]` flips to `false` only once both
ligands' anchors come back `stable`. A `manual_review_required` status means
the anchors (or the system's setup) need to be revisited before production.

In `auto` mode, ChemLink selects RBFE when the ligands have equal net charge and
at least 50% common-heavy-atom coverage. Otherwise it prepares two independent
ABFE calculations, one per ligand. Use `--fep-type rbfe` or `--fep-type abfe` to
make the selection explicit.

### How many replicas and lambda windows

There's no single number the field agrees on, but Q1-cited conventions give
a workable range:

- **Replicas**: at least 3 independent replicas (different initial
  velocities) is the most commonly cited minimum for checking reproducibility
  (Klimovich, Shirts & Mobley, *J. Comput. Aided Mol. Des.* 2015; Mobley &
  Gilson, *Annu. Rev. Biophys.* 2017) — a single estimator's statistical
  error is not a substitute for this, so `--replicas` defaults to **3** and
  does not vary by method.
- **Lambda windows**: RBFE (one A->B transformation) typically needs fewer
  windows than ABFE (a full restrain+discharge+vanish decoupling per
  ligand) for comparable neighbor overlap — commonly 12-24 for RBFE
  (e.g. Wang et al., *JACS* 2015, the FEP+ accuracy paper) and 20-40 for
  ABFE (e.g. Aldeghi et al., *Chem. Sci.* 2016/2018, the same Boresch
  double-decoupling scheme used here). `--lambda-windows` defaults to the
  low end of each range — **12 for RBFE, 20 for ABFE**.
- `--production-time` defaults to **5 ns/window** (2-10 ns/window is the
  typical range — 5 ns is the middle of it, not the floor).
- Adding more replicas, windows, or per-window time costs total sampling
  time roughly linearly; these defaults are meant as a defensible starting
  point, not a ceiling — the actual test of "enough" is whether
  `fep-analyze`'s automatic diagnostics (see "Interpreting the result"
  below) come back `"reliable"`, not matching a number from a paper. If you
  raise `--production-time` well above the default, consider lowering
  `--lambda-windows` to the low end of its range to keep total sampling
  cost in check rather than compounding both increases.

## Execution status

`--prepare-only` generates validated inputs and execution manifests only.
`--build-only` additionally runs ACPYPE and GROMACS to build, solvate, and
neutralize the four systems, and validates the Boresch restraints with
`grompp`, then stops. Every command and its output are recorded in
`setup_commands.log`; completion is summarized in `abfe_setup.json`.

For ABFE, omitting *both* flags is the normal way to run this command: it
chains system construction straight into the pilot, production, and analysis
stages in this same invocation — see "Running the whole workflow in one
command" below. RBFE only supports `--prepare-only`; production and analysis
are not implemented yet for it (that path needs pmx to build the hybrid A/B
topologies), so `chemlink dynamic fep` without `--prepare-only` refuses to
run for RBFE.

### Replica equilibration

The first real molecular dynamics of the ABFE workflow, run automatically as
part of the pilot step above (there is no separate command for it):
independent EM + NVT + NPT for every replica of every system (3 replicas x
4 systems = 12 equilibrations), each with its own randomized initial
velocities (`gen_vel = yes`, `gen_seed = -1` in NVT) so replicas stay
statistically independent. All three stages run at `init-lambda-state = 0`
— restraints off, ligand fully coupled — the physical reference state every
lambda window is later seeded from. NVT and NPT each run for
`--equilibration-time` ns. Output lands in
`<system>/replica_NN/equilibration/{em,nvt,npt}.*`, alongside where
`replica_NN/lambda_NN/` production windows already exist;
`equilibration_report.json` at the run root summarizes per-replica status,
and re-running (e.g. via `--resume`, below) is safe: completed stages,
detected via `{phase}.gro` + `"Finished mdrun"` in `{phase}.log`, are
skipped.

### Production

Runs automatically once the pilot passes (or continue an existing run with
`--resume`, below). MD across every prepared lambda window — 2 ligands x 2 legs x
`--replicas` x `--lambda-windows` states (240 with this project's defaults
for ABFE: 3 replicas, 20 windows). Refuses to start while
`fep_preflight.json["production_locked"]` is `true` (the pilot has not
passed yet). Every window in a replica is seeded independently from that
replica's own `npt.gro`/`npt.cpt` (equilibration, not chained
window-to-window), using the `production.mdp` already written during
preparation — `grompp` + `mdrun` land inside that window's own
`replica_NN/lambda_NN/` directory as `production.{tpr,gro,log,...}` plus
`production.xvg`/`dhdl.xvg` (`separate-dhdl-file = yes` is already set),
which is all Fase 4 analysis (BAR/MBAR/TI) needs — production does not
write compressed trajectories, unlike the pilot, since 240 windows of `.xtc`
would be a large and unnecessary amount of disk.

Production is fully resumable: it walks all the states in a deterministic
order, skips any already `"Finished mdrun"`, and a single window failing
does not abort the rest — it is recorded in
`production_progress.json["failed"]` and the loop continues. Re-running
(via `--resume`, below) only touches what is still pending or failed. Each
window's `state.json` tracks its own `pending/running/completed/failed`
status independently of the run-wide `production_progress.json` summary.

Because production is sized by `--production-time` per window across 240
windows, wall-clock time for a full run is realistically days to weeks on a
single machine, depending on system size and hardware — this is exactly why
resumability is not optional here.

### Analysis

Runs automatically once production finishes (or continue an existing run
with `--resume`, below). Turns the completed `production.xvg` (GROMACS dhdl) files into a binding free
energy per ligand. Refuses to run until `pilot_report.json["status"] ==
"passed"`, since that is what validates both the restraint geometry and the
standard-state correction used below.

For each ligand, each leg (`complex`, `solvent`) is estimated independently
with `alchemlyb`/`pymbar`:

- **TI** integrates the mean dH/dlambda across the lambda chain; always
  available, since every window records its own derivative regardless of
  `calc-lambda-neighbors`.
- **BAR** uses each window's energy evaluated at its immediate neighbors;
  always available, since `calc-lambda-neighbors = 1` (the setting
  `fep_pipeline._fep_lambda_schedule` writes into every `production.mdp`)
  produces exactly the neighbor pair each BAR step needs. Its end-to-end
  error is the standard chain propagation (root-sum-square) over the
  per-step adjacent errors, since BAR itself does not report a direct
  first-to-last uncertainty.
- **MBAR** needs every lambda pair's foreign energy for every window, which
  `calc-lambda-neighbors = 1` does not produce; it is reported as
  `"status": "unavailable"` whenever the pooled energy matrix has missing
  pairs, which is expected with this project's current production settings.

Every replica that finished all `lambda_windows` for a given (ligand, leg) is
pooled into one combined estimate; replicas that didn't (missing or failed
windows) are excluded and listed under `incomplete_replicas`, and each
completed replica's own TI estimate is also reported (`replica_estimates`,
`replica_spread`) as a reproducibility check — a replica whose value sits far
from the others usually means it needs a longer production run rather than a
real difference in ΔG.

The two legs are combined into a per-ligand binding free energy with the
standard Boresch double-decoupling formula:

```
delta_g_bind = delta_g_solvent - delta_g_complex + delta_g_restraint
```

`delta_g_complex` is turning the ligand's nonbonded interactions off in the
complex while the Boresch restraint turns on (lambda 0: coupled,
unrestrained -> lambda 1: decoupled, restrained); `delta_g_solvent` is the
same decoupling in bulk solvent, unrestrained throughout; `delta_g_restraint`
is the analytical standard-state correction the pilot step already computed
(Boresch et al. 2003, eq. 32). Results (per estimator, in kJ/mol and
kcal/mol, with propagated errors) are written to `fep_analysis.json`.

### Interpreting the result

Every ligand's `binding_free_energy` in `fep_analysis.json` carries a
`diagnostics` block that automates the checklist you'd otherwise apply by
hand — none of its thresholds are hard physical limits, just the same rules
of thumb documented here, so a `"needs_review"` verdict means "look at this
by hand," not "this is wrong":

- **`binding_magnitude`** — |delta_g_bind| outside roughly 2-20 kcal/mol is
  atypical for a drug-like binder and usually means unconverged sampling
  rather than a genuinely very weak or very tight binder.
- **`ti_bar_agreement`** — TI and BAR are two different estimators over the
  same data; if they disagree by more than ~2 kcal/mol, lambda windows
  likely don't overlap enough or production is too short.
- **`statistical_error`** — flags a NaN or a TI error large relative to the
  estimate itself (more sampling needed); a NaN in BAR's error usually means
  a specific lambda pair had poor overlap.
- **`<leg>_replica_count`** / **`<leg>_replica_agreement`** — independent
  replicas are the most direct convergence check available: a single
  completed replica means reproducibility was never actually tested, and a
  large spread between replicas' TI estimates means they disagree with each
  other, regardless of how small any individual estimator's error looks.
- **`<leg>_replica_completeness`** — replicas dropped from the combined
  estimate due to missing or failed production windows (see "Production").
- **`standard_state_correction_range`** — outside roughly -12 to -3 kcal/mol
  suggests the Boresch anchor geometry itself (`common/*_boresch_anchors.json`)
  is questionable, independent of anything from production.

`fep_analysis.json["review_status"]` is `"reliable"` only when every check
for both ligands passed; the analysis step prints only the warnings (if any)
rather than the full checklist, to keep the terminal readable — the full
list is always in the JSON file.

Since ABFE gives each ligand's binding free energy independently (not a
relative A-vs-B number the way RBFE would), `fep_analysis.json["ranking"]`
compares them using TI (always available, unlike BAR/MBAR) and reports
which one binds tighter and by how much
(`delta_delta_g_kcal_mol = delta_g_bind(B) - delta_g_bind(A)`).

`install.sh` and the Docker image install `alchemlyb` and `pymbar` in the main
`bio` environment. Re-running `install.sh` upgrades an existing environment as
well as configuring new installations.

### Running the whole workflow in one command

There is only one command, `chemlink dynamic fep` — prepare, build,
Boresch validation, equilibration, pilot, production, and analysis are all
functions of this single command, not separate subcommands. For ABFE, it
runs that full sequence by default:

```bash
chemlink dynamic fep \
  -p protein.pdb \
  -a ligand_A.pdb \
  -b ligand_B.pdb \
  --replicas 3 \
  --lambda-windows 24 \
  --run-name abfe_A_B
```

This runs the same functions in the same order every time — prepare, build
the four systems, validate Boresch restraints, equilibrate, pilot,
production, analysis — it is not a different, faster path, just the default
shape of the command. Use `--prepare-only` or `--build-only` to stop
earlier instead (see "Execution status" above).

It stops and reports which stage failed rather than continuing past a
problem: in particular, if `pilot_report.json["status"] != "passed"`, it
stops before production (which would refuse to run anyway while
`production_locked` stays `true`) so a bad restraint geometry is never
silently carried forward into hours or days of production MD. On a real
production-sized run (the CLI's replica/window/time defaults), this single
command blocks for as long as the full workflow takes — realistically days —
so run it under `nohup`/`tmux`/a job scheduler rather than in a terminal you
need back.

### Continuing an interrupted or failed run

```bash
chemlink dynamic fep --resume data/output/dynamics/abfe_A_B
```

`--resume` skips prepare/build entirely and continues an *existing* run
directory straight from the pilot (which itself skips whatever equilibration
and Boresch validation is already done) through production and analysis.
Use it any time the workflow above stopped partway — the pilot didn't pass
the first time and you fixed something, `mdrun` was interrupted mid-production,
some windows failed and you want to retry them, or you just want to re-run
the analysis after inspecting the results. `--protein`, `--ligand-a`,
`--ligand-b`, and `--run-name` are not needed (and `--resume` is mutually
exclusive with `--prepare-only`/`--build-only`, since there is nothing left
to prepare or build). Every other flag (`--replicas`, `--lambda-windows`,
`--equilibration-time`, `--production-time`, ...) is read back from the run
directory's own `run_config.json`, not from the command line, so it doesn't
need to be repeated either.
