# Energía de unión con entropía

El análisis de dinámica permite activar una estimación entrópica en gmx_MMPBSA:

- `--mmpbsa-entropy none`: desactiva explícitamente la corrección entrópica.
- `--mmpbsa-entropy ie`: entropía de interacción (IE); valor por defecto.
- `--mmpbsa-entropy c2`: aproximación C2.

La temperatura del cálculo entrópico es la de la dinámica (300 K) y el segmento IE es
el 25 % de la trayectoria; ambos son fijos porque la dinámica generada por ChemLink
siempre corre a 300 K.

`--mmpbsa-entropy` se acepta en los tipos con energía de unión (`pligand`, `ppeptide`,
`pprotein`, `ppligand`), en `dynamic analyze` y en `hpc dynamics`. El reanálisis activa IE por defecto incluso si una ejecución anterior guardó `none`. Para desactivar la entropía se debe indicar `--mmpbsa-entropy none`; para utilizar C2 se debe indicar `--mmpbsa-entropy c2`. Las demás opciones conservan la configuración guardada cuando no se especifican valores nuevos.

El flujo completo `pligand` ejecuta el análisis de unión con IE al terminar la dinámica, sin añadir una opción de entropía:

```bash
chemlink dynamic pligand          # proteína y ligando detectados en el directorio actual
```

Para ejecutar ese flujo sin corrección entrópica:

```bash
chemlink dynamic pligand --mmpbsa-entropy none
```

Para reanalizar una simulación existente sin repetir la dinámica:

```bash
chemlink dynamic analyze --only binding    # última corrida de results/dynamics
chemlink dynamic analyze /ruta/a/simulacion --only binding
```

Para usar C2, añadir `--mmpbsa-entropy c2`; para desactivar la entropía, añadir `--mmpbsa-entropy none`. `--mmpbsa-frames` y `--mmpbsa-discard` controlan el conjunto de frames de energía y entropía. Una estimación entrópica requiere estudiar su estabilidad al variar el muestreo; aumentar el número de frames no garantiza convergencia.

## Resultados

La estimación corregida es `ΔG_unión ≈ ΔE_MM + ΔG_solvación − TΔS`. Los métodos IE y C2 son aproximaciones, no cálculos de toda la entropía de unión.

En `binding_energy_analysis/gmx_MMPBSA/`:

- `SUMMARY_REPORT.txt` presenta energía base, corrección `−TΔS`, energía libre estimada, temperatura y diagnósticos disponibles.
- `binding_energy.png` compara energía base y corregida cuando existe la corrección.
- `analysis_metadata.json` conserva `total_mean`/`total_std` como energía base y añade `corrected_mean`, `corrected_uncertainty` y `entropy_correction`. Esta última se obtiene como diferencia entre los dos totales redondeados por gmx_MMPBSA; `entropy_reported_correction` conserva además la contribución de su tabla entrópica cuando está disponible.
- `FINAL_RESULTS_MMPBSA.dat` conserva los resultados originales y sus tablas de incertidumbre.
- La serie temporal y su gráfica muestran energía **sin** entropía: la corrección se estima sobre un conjunto de frames, no sobre cada frame individual.

La incertidumbre base corresponde a la desviación estándar de los frames. La incertidumbre corregida se copia de gmx_MMPBSA; su definición depende de la versión. No se debe interpretar automáticamente como la misma desviación estándar o como un intervalo de confianza.

Se registran la versión de gmx_MMPBSA, la desviación de la energía de interacción (`σ`) y los avisos/diagnósticos que produce el motor. En versiones 1.6.x, `ie_segment` determina el tramo final usado en el resumen IE. En versiones recientes de desarrollo, controla un diagnóstico de convergencia y el resultado principal usa el conjunto completo. ChemLink conserva el resultado del motor sin cambiar su definición. Consultar la [documentación de gmx_MMPBSA](https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/input_file/) para la versión utilizada.

Si el motor termina con error al activar entropía, ChemLink conserva los logs y la entrada de ese intento con el prefijo `entropy_attempt_` y realiza un segundo intento con entropía desactivada. Esto puede repetir el cálculo de energía base. Si logra recuperarla, marca el análisis con advertencias y la corrección como incompleta. Si también falla la recuperación, marca el análisis como fallido.
