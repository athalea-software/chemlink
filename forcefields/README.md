# Campos de fuerza

ChemLink busca aquí los campos de fuerza de GROMACS (`<nombre>.ff/`), además de
`CHEMLINK_FORCEFIELD_DIR`, `GMXLIB` y las carpetas de GROMACS.

CHARMM36m (`charmm36-jul2022.ff`) no se incluye en el repositorio porque sus
archivos no declaran condiciones de redistribución. `install.sh` lo descarga
de la fuente oficial (grupo MacKerell, Universidad de Maryland), verifica su
SHA-256 y lo deja en esta carpeta; el `Dockerfile` hace lo mismo en
`/opt/forcefields`. Para instalarlo a mano:

```bash
curl -fL -o charmm36-jul2022.ff.tgz \
  "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files%2Fcharmm36-jul2022.ff.tgz"
echo "9f31db8502852bc8fcf36497c27db14f1b6509b33701e7cafc2187840b7d2f09  charmm36-jul2022.ff.tgz" | sha256sum -c
tar -xzf charmm36-jul2022.ff.tgz -C forcefields/
```
