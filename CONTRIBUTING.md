# Contributing to ChemLink

Thanks for helping improve ChemLink. Bug reports, documentation fixes, new
adapters and pipeline improvements are all welcome. By taking part you agree to
follow the [code of conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and asking for features

Open an [issue](https://github.com/athalea-software/chemlink/issues/new/choose)
and fill in the form, including the output of `chemlink doctor`. For usage
questions see [SUPPORT.md](SUPPORT.md); for security problems see
[SECURITY.md](SECURITY.md) and do not open a public issue.

## Making a change

1. Fork the repository and create a `feature/<name>` or `fix/<description>`
   branch from `main`.
2. Set up a development environment. The interface and most tests only need
   Python ≥ 3.10:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ./chemlink demo                 # check the CLI runs
   python -m pytest tests          # run the test suite
   ```

   The scientific programs (GROMACS, AutoDock-GPU, fpocket…) are only needed to
   run real pipelines; see [docs/Installation.md](docs/Installation.md).
3. Make focused commits with a descriptive prefix (`Add:`, `Fix:`,
   `Refactor:`, `Update:`, `Remove:`), in English.
4. Add or update tests for what you changed, and update the README or `docs/`
   if commands or options changed. Add user-facing changes under
   `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
5. Open a pull request against `main` and fill in the template. CI runs the
   tests on Python 3.10–3.12 and a CLI smoke test; it must pass before merging.

Code conventions, the repository layout and the release process are described
in [docs/Development.md](docs/Development.md).

## Licensing of contributions

ChemLink is offered under the AGPL-3.0 and under a commercial license (see
[LICENSING.md](LICENSING.md)). Before we can accept your pull request, you
need to sign the [contributor license agreement](CLA.md) once, by commenting
on the pull request with the text given at the end of that document.

## Third-party code and dependencies

Dual licensing only works if ChemLink does not include code under incompatible
licenses. Before submitting a change:

- **Do not copy code** from GPL/LGPL projects, from content licensed
  CC BY-SA (for example Stack Overflow or cp-algorithms) or from software
  licensed for non-commercial use only. If an algorithm is already available
  in a permissively licensed library (MIT, BSD, Apache-2.0), use it as a
  dependency.
- **Do not import GPL libraries** into the ChemLink process. A GPL program
  can still be run as a separate program (`subprocess`), as `obabel` and
  AutoDock-GPU are today.
- **Every new dependency** must have a permissive license (MIT, BSD,
  Apache-2.0, PSF, MPL-2.0), or LGPL if used unmodified. Add it to
  `requirements.txt`, to the `install.sh` and `Dockerfile` environments, and to
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- **Do not add data without a clear license** to the repository, such as force
  fields or structures from databases with restrictions. If it is needed, have
  `install.sh` download it from the official source, as is done for CHARMM36
  (see [forcefields/README.md](forcefields/README.md)).
