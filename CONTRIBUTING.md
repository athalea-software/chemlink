# Contributing to ChemLink

The branch workflow, code conventions and tests are described in
[docs/Development.md](docs/Development.md). In short: work on a `feature/*`
or `fix/*` branch, open the pull request against `develop` and run the test
suite (`python -m pytest tests`) before submitting it.

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
