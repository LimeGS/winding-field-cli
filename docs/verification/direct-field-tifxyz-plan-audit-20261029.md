# Auditoría adversarial del plan field→TIFXYZ

Fecha:2026-10-29. Revisión de documentos/código existente; feature no implementado.
Base:`6b84457c51591a2f7a50ed49586dd73d44c65c05`.
Diseño:`../superpowers/specs/2026-10-29-direct-field-tifxyz-design.md`.
Plan:`../superpowers/plans/2026-10-29-direct-field-tifxyz.md`.
Revisor independiente: sesión `ses_e7718d69affetzyp1sUsqifP13`.

## Primera revisión

Veredicto inicial: REQUEST CHANGES. Encontró2Critical,8Important,1Minor.
Se verificaron sus referencias contra las APIs reales y se revisaron ambos
documentos. “Corregido” aquí significa requisito/algoritmo/prueba incorporados al
plan; no significa que una implementación haya pasado tests inexistentes.

| ID | Ataque / fallo | Cambio en el diseño y prueba propietaria |
|---|---|---|
| C1 | Nodos aislados pueden cubrir la fuente aunque VC3D no emita una sola cara |Normalizar a superficie visible, exigir celdas/triángulos, excluir huérfanos del índice y reauditar; T4/T5/T8 |
| C2 | FieldSource conserva todo el manifest y lo rehashea en cada read; cache no lo limita |Open/parse dentro del worker, envelope16MiB/65536chunks, cargo explícito y benchmark de metadata real; NO-GO si requiere rediseñar lector; T1/T6 |
| I1 | Empezar en stride mínimo admisible deja sin refinamientos válidos |Calendario[4*s_min,2*s_min,s_min] filtrado; test8193³y dimensiones insuficientes; T2 |
| I2 | Root ownership no distingue un cusp[1,0,1] de un cruce simple |Clasificar ambos segmentos, reglasROI/empates/DDA/error numérico y redondeo float32; T3 |
| I3 | Caps512rayos/20Mpares y cota de arista rechazan hasta planos fáciles |4096celdas/rayo preadmitidas, counters locales/no-locales separados, cotaL/√3 demostrada; positivos anisótropo y8000²; T3/T5/T6 |
| I4 | MC o CLI flatten no son baseline de un TIFXYZ global equivalente |Separar oráculo de baseline completo; misma fuente/ROI/frame/tau/auditor nuevo, tiempos de apertura a aceptación; T6 |
| I5 | “Cuotas” no especificadas no frenan journals/temporales/índices |SQLite DELETE+max_page_count, reserva journal, sin sorts globales, capsencoder/allocatedblocks y rollback; T4/T5 |
| I6 | FIELD.json sin RUN binding puede asociar mal fuente/frame; open_field importa torch |Cadena dehashes, digestRUN.source=frame y constructorFieldSource torch-free; T1/T7 |
| I7 | Config por pathname mutable, ROI indefinida y BLOCKED contado como éxito |Snapshot por contenido+settings efectivos, ROI en nodos, precedencelevels, tablaestados/exits y READY exclusivo; T1/T6/T7 |
| I8 | Negativos topológicos exigen garantías que el diseño rechaza |Islas>tau fallan; isla≤tau es sensibilidad topológica UNMEASURED, no un test de rechazo obligatorio; T5/T6 |
| M1 | Grafo ponía el GO antes del task que lo implementa |T6 contiene motor+benchmark+GO; T7–T9 sólo después |

## Decisiones propuestas tras la revisión

1. V1 permite sub-ROI explícita de un checkpoint existente, en nodosZYX, con
   origen absoluto conservado; no inventa selección automática de componentes.
2. Primer caso útil: cuatro ROIs prefijadas de512³celdas del W−4 histórico175A,
   derivadas de percentiles10/35/65/90 de sus tiles intersectados. PILOT se fija
   antes de resultados. Al menos una comparación válida≥250kcaras debe pasar;
   publicar también los otros tres resultados. No equivale a W−4completo.
3. No cambiar reader congelado para hacer bonito el benchmark: si su manifest
   domina, se informa NO-GO y se pide un plan separado para ese cuello de botella.
4. Gates propuestos:3GiB agregado,≤50%RAM y≤2×tiempo del baseline equivalente.
   Su utilidad se somete a aprobación; no son mediciones ni garantías previas.

## Autoauditoría adicional

Se corrigieron referencias a WAL frente a journalDELETE, la firma de tipos
auxiliares, el presupuesto del controlador, el frame del baseline, la fuente de
conformance histórica y los intervalos de la auditoría de subceldas. Todas las
escrituras de esta sesión son documentación; no se alteró el plan Qual V2.

## Límites de esta auditoría

No se ejecutó el algoritmo nuevo, benchmarks, tests de implementación, loader
VC3D o enforcementLinux. No demuestra equivalencia topológica/hoja física,
viabilidad de W−4completo ni tiempo de121niveles. Una auditoría del plan puede
cerrar contradicciones y fijar pruebas, pero no sustituir los gates experimentales.

## Revisión de cierre

La segunda pasada cerró C1/C2, I1–I4/I7/I8 y M1. Dejó dos correcciones concretas
en I5/I6, ambas incorporadas antes del cierre:

- **R1 — RUN mutable:** guardar snapshot histórico y comparar sólo identidad
  estable del productor/fuente/field/checkpoint. El padre puede registrar
  COMPLETE/PAUSED sin invalidar su export. T1/T7 incorporan la regresión.
- **R2 — journal no limitado por max_page_count:** imponer RLIMIT_FSIZE por fase
  para journal_cap, con pruebas de spill/cache pequeño, reap y recovery. La cifra
  reservada es una cuota aplicada, no un máximo natural inventado de SQLite.

También se corrigieron la fila-resumen de temporales SQLite y los estados de
retorno separados de export/verify. El revisor confirmó la cotaL/√3 con una
demostración baricéntrica independiente y contrastó el configSHA histórico175A.
La comprobación final dirigida del mismo revisor confirmó R1/R2 cerrados y
ninguna contradicción concreta en esos cambios. Veredicto final:
**READY FOR USER APPROVAL OF THE GATED IMPLEMENTATION PLAN**.

No quedan bloqueantes documentales de esa auditoría. Se autoriza entregar el
plan para aprobación; no se declara probado el runtime, enforcement de recursos,
rendimiento, cobertura de W−4completo ni viabilidad de121niveles.

## Entrega

Sólo3documentos nuevos: diseño, plan y esta auditoría. No se implementó ninguna
etapa y no se cambiaron fuentes, tests, dependencias o el plan Qual V2 existente.
La próxima acción requiere aprobación del alcance y de las decisiones propuestas.
