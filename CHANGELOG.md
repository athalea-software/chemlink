# Changelog

All notable changes to ChemLink are listed here. Versions follow
[semantic versioning](https://semver.org/); each one is also published as a
[GitHub release](https://github.com/athalea-software/chemlink/releases).

## [Unreleased]

### Added

- Code of conduct, security policy, support page, citation file, issue and
  pull request templates, and continuous integration (tests on Python
  3.10–3.12 and a CLI smoke test).

### Changed

- Pull requests now target `main`; the `develop` branch is retired.

### Fixed

- Tests that depended on a removed CLI helper or on a local Open Babel install.

## [2.0.0] - 2026-10-26

### Changed (breaking)

- Relicensed as AGPL-3.0-only, with a commercial license available from
  Athalea (see `LICENSING.md`).
- Standardised CLI inputs across pipelines (`-ir`, `-il`, `-ic`, `-ip`, `-o`).
  The v1.0.0 flags `--ligands`, `--ligand-dir`, `--receptor-dir`, `--protein`,
  `--workers`, `--max-workers`, `--mode` and the tool-path flags were removed.
  Worker counts are set per stage (`-r`, `-l`, `-a`, `-d`).
- `chemlink dynamics` is now `chemlink dynamic`. Docking is split into
  `prepare`, `full`, `flow`, `receptor`, `ligand`, `active-site`, `run` and
  `analyze`.
- The repository moved to `athalea-software/chemlink`, and the website to
  [athalea.co/chemlink](https://athalea.co/chemlink).

### Added

- Peptide–protein docking with LightDock, PRODIGY and a robustness pipeline
  (`chemlink docking peptide`).
- CPU docking with AutoDock Vina (`--engine both` runs GPU and CPU together),
  flexible-receptor docking, co-crystal centre detection, redocking and
  receptors with cofactors.
- Robust z-score as the main combined docking ranking.
- Molecular dynamics of protein–ligand + cofactor systems, force-field and
  water-model selection, 2 replicas per protein–ligand run by default, and
  `dynamic resume` / `dynamic analyze`.
- MM-PBSA / MM-GBSA with an entropy term.
- Terminal design system, interactive TUI, `chemlink demo`, `chemlink update`,
  `chemlink completion`, and `--no-color` / `--plain` output.
- English output, README and documentation.

### Fixed

- Ligand validation before docking and MD (Open Babel 3.2.1 bond orders,
  hydrogen stripping for over-valent nitrogens), automatic charge detection,
  non-canonical residues, PDBQT glue atoms, pose element export, retries of
  transient preparation crashes, and installer requirements.

### Removed

- The FEP prototype and committed run outputs.

## [1.0.0] - 2026-05-22

First stable release: docking pipeline (fpocket, MGLTools + AutoGrid4,
AutoDock-GPU), GROMACS molecular dynamics for six system types, single-node
and SLURM multi-node modes, `chemlink doctor`, Prometheus + Grafana
monitoring and Docker support. Distributed under the MIT license.

[Unreleased]: https://github.com/athalea-software/chemlink/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/athalea-software/chemlink/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/athalea-software/chemlink/releases/tag/v1.0.0
