# Contribuir a ChemLink

El flujo de ramas, las convenciones de código y las pruebas están en
[docs/Desarrollo.md](docs/Desarrollo.md). En resumen: trabaje en una rama
`feature/*` o `fix/*`, abra el Pull Request hacia `develop` y corra la suite
(`python -m pytest tests`) antes de enviarlo.

## Licencia de las contribuciones

ChemLink se ofrece bajo la AGPL-3.0 y bajo una licencia comercial (ver
[LICENSING.md](LICENSING.md)). Para poder aceptar su Pull Request, necesitamos
que firme el [acuerdo de licencia de contribución](CLA.md) una sola vez,
comentando en el Pull Request el texto indicado al final de ese documento.

## Código y dependencias de terceros

La licencia dual depende de que ChemLink no incorpore código con licencias
incompatibles. Antes de enviar un cambio:

- **No copie código** de proyectos GPL/LGPL, de respuestas con licencia
  CC BY-SA (por ejemplo Stack Overflow o cp-algorithms) ni de software con
  licencias "solo uso no comercial". Si un algoritmo ya está en una biblioteca
  permisiva (MIT, BSD, Apache-2.0), úsela como dependencia.
- **No importe bibliotecas GPL** dentro del proceso de ChemLink. Un programa
  GPL sí puede usarse como programa aparte (`subprocess`), como hoy `obabel` o
  AutoDock-GPU.
- **Toda dependencia nueva** debe tener una licencia permisiva (MIT, BSD,
  Apache-2.0, PSF, MPL-2.0) o LGPL si se usa sin modificar. Agréguela a
  `requirements.txt`, a los entornos de `install.sh` y del `Dockerfile`, y a
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- **No agregue al repositorio datos sin licencia clara**, como campos de fuerza
  o estructuras de bases de datos con restricciones. Si hace falta, haga que
  `install.sh` los descargue de la fuente oficial, como con CHARMM36
  (ver [forcefields/README.md](forcefields/README.md)).
