## What and why

<!-- What this change does and the problem it solves. Link the issue: "Closes #123". -->

## How it was tested

<!-- Commands you ran, and for pipeline changes the smallest-scale run
     (10 ligands / 0.1 ns) and its result. -->

## Checklist

- [ ] `python -m pytest tests` passes
- [ ] New external tools or libraries are permissively licensed and listed in `requirements.txt`, `install.sh`, the `Dockerfile` and `THIRD_PARTY_LICENSES.md`
- [ ] No hardcoded absolute paths; SLURM scripts are parametrized through environment variables
- [ ] README / `docs/` updated if commands or options changed
- [ ] I have signed the [CLA](https://github.com/athalea-software/chemlink/blob/main/CLA.md) (first contribution only: comment on this PR with the text at the end of CLA.md)
