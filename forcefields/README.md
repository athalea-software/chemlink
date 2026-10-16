# Force fields

ChemLink looks for GROMACS force fields (`<name>.ff/`) here, as well as in
`CHEMLINK_FORCEFIELD_DIR`, `GMXLIB` and the GROMACS data directories.

CHARMM36m (`charmm36-jul2022.ff`) is not included in the repository because its
files do not state any redistribution terms. `install.sh` downloads it from the
official source (MacKerell group, University of Maryland), verifies its
SHA-256 and places it in this directory; the `Dockerfile` does the same in
`/opt/forcefields`. To install it by hand:

```bash
curl -fL -o charmm36-jul2022.ff.tgz \
  "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files%2Fcharmm36-jul2022.ff.tgz"
echo "9f31db8502852bc8fcf36497c27db14f1b6509b33701e7cafc2187840b7d2f09  charmm36-jul2022.ff.tgz" | sha256sum -c
tar -xzf charmm36-jul2022.ff.tgz -C forcefields/
```
