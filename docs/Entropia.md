# Energía de unión con entropía

El análisis de dinámica permite activar una estimación entrópica en gmx_MMPBSA:

- `--mmpbsa-entropy none`: energía MM-GBSA/MM-PBSA sin corrección entrópica; valor por defecto.
- `--mmpbsa-entropy ie`: entropía de interacción (IE).
- `--mmpbsa-entropy c2`: aproximación C2.
- `--mmpbsa-temperature K`: temperatura del cálculo entrópico; 300 K por defecto, igual que la dinámica generada por ChemLink. Para trayectorias externas debe coincidir con su temperatura de simulación.
- `--mmpbsa-ie-segment PCT`: porcentaje entero entre 1 y 100; 25 por defecto.

Estas opciones se aceptan en los comandos de dinámica local, `dynamic analyze` y `hpc dynamics`. El reanálisis conserva la configuración guardada cuando no se especifican opciones nuevas. `--mmpbsa-entropy none` desactiva explícitamente una corrección guardada.

Para reanalizar una simulación existente sin repetir la dinámica:

```bash
chemlink dynamic analyze /ruta/a/simulacion --only binding \
  --mmpbsa-entropy ie --mmpbsa-temperature 300 --mmpbsa-ie-segment 25
```

Para usar C2, sustituir `ie` por `c2`. `--mmpbsa-frames` y `--mmpbsa-discard` controlan el conjunto de frames de energía y entropía. Una estimación entrópica requiere estudiar su estabilidad al variar el muestreo; aumentar el número de frames no garantiza convergencia.

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
