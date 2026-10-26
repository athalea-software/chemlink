# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 2.x | ✅ |
| 1.x | ❌ (upgrade to 2.x) |

## Reporting a vulnerability

**Do not open a public issue for security problems.** Write to
**[dev@athalea.co](mailto:dev@athalea.co)** with:

- the ChemLink version (`chemlink --version`) and how it was installed
  (`install.sh`, Docker, or a git checkout);
- what the problem is and what an attacker could do with it;
- the steps to reproduce it, or a proof of concept.

We will confirm that we received the report within 5 working days, keep you
informed while we work on a fix, and credit you in the release notes unless you
prefer otherwise. Please give us a reasonable time to publish a fix before you
disclose the problem.

## Scope

ChemLink runs scientific programs (GROMACS, AutoDock, fpocket, Open Babel,
LightDock and others) as separate processes and is usually deployed on shared
HPC clusters. Problems that are in scope include, for example:

- command injection through file names, CLI options or input files;
- path traversal or writes outside the run directory;
- the installer (`install.sh`) or the Docker image fetching or running
  unverified code;
- secrets or credentials leaking into logs or result files.

Vulnerabilities in the third-party programs themselves should be reported to
their projects (see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)); we will
update the pinned versions once a fix is available.
