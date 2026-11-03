# Direct Field → TIFXYZ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Producir un TIFXYZ global directamente desde el campo fino, evitando la malla nativa completa y su conversión posterior, con RAM acotada y aceptación geométrica explícita.

**Architecture:** Construir y aplanar sólo una guía pequeña; localizar los nodos de una rejilla UV en el campo fino por raíces acotadas. Persistir bloques y auditar el nivel completo a tolerancia antes de publicar TIFXYZ. Es una rama field→tifxyz, no mesh→flatten→convert.

**Tech Stack:** Python3.11+, NumPy/SciPy/scikit-image/SQLite existentes; extra opt-in libigl2.6.3 y tifffile. Linux para el gate de recursos; pruebas geométricas portables.

**Spec:** `docs/superpowers/specs/2026-10-29-direct-field-tifxyz-design.md`.

**Approved amendment (2026-10-30):** User chose **quality first** after the exact
native reference failed in SLIM even at a separately bounded3.5GiB envelope.
T6→T7 now requires a fully verified direct result and<=3GiB, with experimental
labelling. Comparative performance remains UNMEASURED/NO_COMPARISON. Original
ratio targets below remain future measurements, not a gate we silently passed.
All source, geometric, publication, memory-reserve and disk-reserve gates remain.

## Global Constraints

- Repositorio objetivo `/Users/user/Software.nosync/Factory/winding-field-cli`, base revisada `6b84457`. El plan Qual V2 untracked es trabajo ajeno; preservarlo.
- No implementar hasta aprobación de este plan. No commits, instalaciones globales, cambios IAM/S3, alquileres ni borrados sin petición explícita.
- No modificar `_core/`, solver, checkpoints, predicción/trazado o productos existentes. Adaptar APIs read-only; registrar procedencia del código reutilizado.
- Ruta nueva no llama a mesh/flatten/conversor ni almacena triángulos nativos completos. MC sólo sobre guía con stride≥2; nunca fallback nativo.
- Niveles explícitos; guía de un componente o BLOCKED. Sin largest implícito, atlas de microparches, puentes, relleno de soporte o relajación silenciosa de tolerancia.
- XYZ internos en µm; TIFXYZ en frame de volumen explícito. W mantiene sus unidades. ArraysZYX y vectoresXYZ; origen=centro de voxel.
- Extra `tifxyz` opt-in: libigl==2.6.3 y tifffile>=2023.7; no instalar desde comandos de ejecución.
- Límites iniciales: exportador+supervisor3GiB+reserva2GiB; scratch16GiB+reserva12,25GiB en gpu-1; guía≤4Mnodos/250kcaras; máximo64Mpíxeles/nivel; cache128MiB y SQLite agregado64MiB; CPU1/BLAS1/sinCUDA.
- Tolerancia por defecto=max(spacing físico); pitch=min(spacing físico); raíz bracket≤min(spacing físico)/1000. No modificar fuente al reanudar.
- READY requiere todos los gates; `physical_acceptance=false`, `native_face_identity=false`, `native_topology_equivalence="UNMEASURED"`.

## Review Focus

1. Una guía pierde una isla o una capa fina: auditar todo el ROI soportado, no sólo sus rayos; prueba adversarial en T5.
2. Dos/tres raíces con extremos de igual signo o junto a un límite de chunk: contar todas, nunca saltar de hoja; T3.
3. Origen no cero, voxel8,64µm, spacing anisótropo del campo y CT L1: aplicar un solo transform y no adivinar L0; T1/T6.
4. SQLite, KD-trees, TIFF y vértices de auditoría reintroducen RAM O(N): prueba de techo real y prohibición de lecturas completas; T4/T5/T8.
5. Reanudación entra a mesh/flatten o publica READY incompleto tras fallo/fsync: rama alternativa, identidad y publicación final; T4/T6/T7.

## Mapa de archivos y dependencias

Nuevos módulos bajo `src/winding_field/`:

| Archivo | Responsabilidad |
|---|---|
| `tifxyz_contract.py` | settings normalizados, frame, tipos de resultados/estados |
| `tifxyz_field.py` | adaptador físico, muestreo agrupado, índice min/max y guía gruesa |
| `tifxyz_guide.py` | guía triangulada acotada, cortes, UV globales, normales/costuras |
| `tifxyz_project.py` | DDA, polinomio trilineal por rayo, todas las raíces y bloquesXYZ |
| `tifxyz_store.py` | ledger, bloques, publicación/lectura TIFF acotada y hashes |
| `tifxyz_audit.py` | índices espaciales en disco, cobertura y geometría de celdas |
| `tifxyz_runtime.py` | admisión host/cgroup, worker y supervisión STOP/recursos |
| `tifxyz.py` | export/verify, reintentos acotados e integración de fases |

Modificar `cli.py`, `config.py`, `workflow.py` para ruta alternativa;
`pyproject.toml`, `README.md`, `docs/FORMATS.md`. No importar/modificar `field.py`
en el worker: abrir FieldSource directamente por un adaptador torch-free.
No modificar `meshes.py`, `flatten.py` ni su semántica; tests existentes son regresión.
Tests nuevos por módulo; `tools/bench_direct_tifxyz.py` es benchmark, no fallback.
T1→T2/T3→T4→T5→T6 **[GO/NO-GO]**→T7→T8→T9. Ejecutar secuencialmente.

## Interfaces compartidas

Definir en T1 y no renombrar en tareas posteriores:

- `RequestedConfig`: copia normalizada por contenido del JSON§6.1, con SHA,
  defaults dependientes de geometría todavía sin resolver.
- `Settings`: levels, roi_lo_zyx/roi_hi_zyx, pitch_um, tolerance_um, memory_bytes,
  scratch_bytes, reserve_disk_bytes, reserve_ram_bytes, max_pixels, output_frame,
  normalized algorithm limits. Fuente/vista ROI mantienen identidades distintas.
- `PhysicalField`: context manager, `shape` ZYX, `origin_um`, `spacing_um`,
  `identity`; `read(lo_zyx, hi_zyx)->(float64 values,bool support)`; `validate(hashes=False)`.
- `Guide`: vertices_um float64[N,3], faces int32[M,3], uv_um float64[N,2],
  normals float64[N,3], seam ancestry, effective_stride y SHA; caras≤250000.
- `GridSpec`: shape_yx, uv_min_um, pitch_um, guíaSHA y nivel; resolución exacta.
- `ProjectedTile`: tile_id, owned_box_yx, xyz float32[H,W,3] en destino, bool mask,
  float64 root_error_um[H,W], reason counts; halo1 no se duplica en publicación.
  `DeliveredTile` tiene igual forma después de normalizar orphan nodes conforme
  al lector; sus hashes/mascara forman otra generación inmutable.
- `AuditResult`: status PASS/BLOCKED, structured reasons, maxima/counters,
  bounded-work evidence, source/grid hashes y alcance; no usa bool truthiness como PASS.
- `Blocked` con código estable; `Paused`; errores de integridad son FAILED.
- `RootResult`: status UNIQUE/ABSENT/AMBIGUOUS/UNSUPPORTED, lista de brackets
  t_um y cell_ids globales, xyz_um/gradient cuando UNIQUE, error_um, residual.
- `CoarseReceipt`: paths+SHA de values/support/minmax, shape/origin/spacing
  físicos, stride, source identity, counts de trabajo; no arrays finos globales.
- `BudgetReceipt`: límites, headroom observado, parent RSS, techo del hijo y
  reserva scratch estimada/real. `IndexReceipt`: paths/SHA, tipo, filas,
  bytes/cache/page quotas y binding a los tiles. `ArtifactManifest`: lista exacta
  path relativo, bytes, SHA y shape/dtype/scale declarados de productos publicados.

### Task 1: Contrato de fuente, unidades y presupuestos

**Files:** crear `tifxyz_contract.py`, `tifxyz_field.py`, `tifxyz_runtime.py`; tests
`test_tifxyz_contract.py`, `test_tifxyz_field.py`, `test_tifxyz_runtime.py`.

**Interfaces:** consume `FieldSource` y geometry/source identity guardadas. Produce
`validate_request(mapping)->RequestedConfig`,
`load_settings(request,field_geometry,output_frame)->Settings`,
`open_physical_field(field_run,cache_bytes)->PhysicalField`,
`to_output_xyz(xyz_um,frame)->float32`, `admit(settings,estimate,out)->BudgetReceipt`.

- [ ] Escribir tests RED: source.origin=[11,23,37], spacing=[2,3,4], unit8,64;
  índiceXYZ[1,2,3]→mundo[112,32;250,56;423,36]µm→destino[3,9,19] con origen
  destino[86,4;172,8;259,2]µm y voxel8,64. Pitch17,28→scale[0,5;0,5].
- [ ] Añadir tests que rechacen NaN/Inf/bool/overflow/pitch0, destino anisótropo
  implícito, voxel desconocido, frame diferente y sample fuera de soporte.
  Chunk-boundary interpolación requiere sus8nodos válidos, sin sustituir ausentes.
- [ ] Añadir admisión con cgroup hijo ausente y ancestro limitante, disco insuficiente,
  presupuesto gigante manipulado y source mutation en cache hit. Bloquear antes
  de asignar/decodificar; no introducir lectura host-only permisiva.
- [ ] RED: rechazar manifest>16MiB antes de parse, count>65536 bajo supervisor,
  accounting de metadata superior a cache, FIELD.json desvinculado de RUN,
  checkpoint/geometry diferentes y frame derivado cuyo digest no coincide.
  `sys.modules` en proceso aislado no contiene torch tras abrir/leer/cerrar.
  Fijar snapshot RUN histórico pero identidad estable sólo en subconjunto fuente/
  productor/field/checkpoint; timings/status/etapa TIFXYZ no entran al binding.
- [ ] RED: sub-ROI half-open en nodos fuente conserva XYZ absolutos, no puede
  salir del campo ni tomar stencils de fuera; todo el scope se registra antes de guía.
- [ ] Run: `python -m pytest tests/test_tifxyz_contract.py tests/test_tifxyz_field.py tests/test_tifxyz_runtime.py -q`.
  Expected: nuevos casos FAIL por interfaces ausentes; luego implementar y repetir hasta PASS.
- [ ] Adaptar el observador Linux probado del kit; parámetros inyectables sólo para
  observar fixtures de kernel. Child env antes de NumPy; supervisor limita sólo su
  worker y nunca procesos ajenos. Mac puede probar matemáticas; runtime no soportado
  debe rechazar claramente, sin afirmar un techo Linux.
  Restar RSS del controlador+64MiB antes del techo de direcciones del hijo;
  probar controlador grande y crecimiento de cache que excede la suma admitida.
- [ ] Ejecutar `python -m pytest tests/test_field.py tests/test_sampling.py -q`.
  Expected: PASS; los callers y valores de `field.open_field` permanecen intactos.

### Task 2: Guía global realmente pequeña

**Files:** completar `tifxyz_field.py`; crear `tifxyz_guide.py` y
`tests/test_tifxyz_guide.py`; agregar extra/provenance/licencia del código adaptado.

**Interfaces:** consume PhysicalField/Settings. Produce
`build_coarse(field,stride,out,stop)->CoarseReceipt` y
`build_guide(coarse,level,settings,stop)->Guide`.

- [ ] Tests RED: plane/cylinder/torus (un disco UV con costuras), cara invertida,
  guía no manifold y dos componentes. Conservar orientaciones y XYZ de guía;
  bloquear una estrella UV que pase sólo un test de giros locales.
- [ ] Test de hueco fino entre esquinas gruesas soportadas: coarse cell inválida.
  Añadir isla fina ausente en guía para que T5 demuestre omisión, no para aprobarla.
- [ ] Test `no_native_fallback`: incluso después de3intentos, ningún stride<2,
  ninguna llamada a `extract_mesh` nativa ni ampliación automática de250kcaras.
- [ ] RED: shape8193³ y cap4M → s_min64, intentos[256,128,64], todos admisibles;
  descartar intentos con<2nodos/eje. Sin ningún stride admisible→BLOCKED, no s=1.
- [ ] Run: `python -m pytest tests/test_tifxyz_guide.py -q`. Expected: RED.
- [ ] Implementar s0, agregación de soporte/minmax por tiles y MC grueso en proceso
  limitado. Adaptar sólo helpers de guía del kit, con libigl lazy-imported; valores
  normativos en diseño§5.1. En exceso de recursos registrar BLOCKED, no crash del host.
- [ ] Run mismo comando. Expected: PASS con guía única/global y máximos medidos;
  sin importación requerida de libigl para comandos legacy/doctor/inspect.

### Task 3: Raíces nativas sin saltos de capa

**Files:** `tifxyz_project.py`; tests `test_tifxyz_project.py`.

**Interfaces:** consume Guide/PhysicalField/GridSpec. Produce
`ray_roots(field,origin_um,direction,t_interval,xtol_um)->RootResult`,
`project_tile(field,guide,grid,tile_id,settings,stop)->ProjectedTile`.

- [ ] Tests RED de plano y hoja curva contra coordenadas analíticas. Un rayo
  con F=(x−0,25)(y−0,5)(z−0,75) sobre x=y=z debe devolver3raíces, no una.
  Raíz doble tangente, polinomio nulo y pendiente incierta→AMBIGUOUS.
- [ ] Tests de raíz en nodo/cara/chunk/último borde del ROI contada una sola vez;
  dos raíces cercanas no se fusionan. Normal nula, rayo>4096celdas o búsqueda con
  soporte desconocido→BLOCKED; no transformar esos errores en máscara transparente.
- [ ] RED: cusp samples[1,0,1] a nivel0→AMBIGUOUS aunque las pendientes laterales
  sean no nulas; plano alineado con nodo y endpointROI tiene política one-sided
  explícita. Reversed rays, eje0 y cruces simultáneos por arista/vértice no duplican.
- [ ] RED: plano anisótropo spacing(1,1,100)µm admite al menos un intento sin
  desbordar cola4096; preflight de celdas precede lecturas y cola total≤16384.
- [ ] Probar igualdad del resultado con tiles128/256 y distinto orden de bloques;
  mismo nivel/frame con float32 output debe conservar error de redondeo medido.
- [ ] Run: `python -m pytest tests/test_tifxyz_project.py -q`. Expected: RED.
- [ ] Implementar DDA half-open, coeficientes trilineales grado≤3, extremos de
  derivada y bisección en intervalos monótonos. Residuales se recalculan con los
 8valores fuente, no con un polinomio distinto. Casos de clasificación incierta
  bloquean; aplicar escalado, intervalos/outward-rounding y límites incidentes
  del diseño§5.2. Agrupar tiles en lotes≤16384, sin crop envolvente gigante.
- [ ] Run mismo comando. Expected: PASS; counters demuestran cache/lotes acotados.

### Task 4: Trabajo en disco y reanudación transaccional

**Files:** `tifxyz_store.py`; tests `test_tifxyz_store.py`, `test_tifxyz_runtime.py`.

**Interfaces:** `TileStore(out,identity,budget,phase,resume=False)`, phase
projected/delivered, con
`commit(tile)->receipt`, `read(tile_id)->ProjectedTile`, `pending_ids(grid)`;
`write_segment(store,grid,frame,pending_dir)->ArtifactManifest`.

- [ ] RED: interrumpir antes/después de escribir/fsync/commit de cada bloque;
  reanudar produce los mismos XYZ/mascaras. Detectar payload corrupto, cursor
  adelantado, symlink, output ajeno, grid/settings/frame/código cambiados.
- [ ] RED: fallo al cerrar TIFF o escribir meta/STATUS jamás deja READY aceptado;
  falta de disco antes de la siguiente asignación pausa/bloquea conservando lo válido.
- [ ] RED: shape enorme o TIFF/NPY/NPZ header malicioso se rechaza antes de decode;
  fallar si writer/verifier intenta cargar imagen, grafo o lista global completos.
  Probar también que el tamaño real de un índice supere la estimación inicial:
  cuotas de DB/journal/temporales deben bloquear antes de consumir scratch/reserva.
- [ ] Run: `python -m pytest tests/test_tifxyz_store.py tests/test_tifxyz_runtime.py -q`.
  Expected: RED; implementar ledger SQLite FULL/DELETE + max_page_count y reserva
  journal completa, fsync de payload/directorio antes de commit, archivos atomizados
  propios y lector por tiles; repetir hasta PASS. Seguir mecanismo disco§6: cuotas
  de bytes/bloques físicos, no WAL ni sort/tmp global, cap del encoder y recursos
  de todos los intentos/outputs retenidos. EXPLAIN no introduce temp B-tree.
- [ ] RED: cache SQLite mínimo + spill bajo RLIMIT_FSIZE=journal_cap. La cuota
  falla sin consumir reserva, commits previos sobreviven a reap/reopen y no hay
  READY. max_page_count por sí solo no cuenta como enforcement del journal.
- [ ] Test subprocess Linux de asignación sobre presupuesto→fallo propio, no READY.
  STOP/SIGINT reap del child; ningún proceso huérfano escribiendo tras PAUSED.

### Task 5: Auditoría completa de cobertura, soporte e intersecciones

**Files:** `tifxyz_audit.py`; tests `test_tifxyz_audit.py`.

**Interfaces:** consume PhysicalField/TileStore/Guide/Settings. Produce
`normalize_delivered(projected_store,grid,reader_contract)->delivered_store`,
`build_spatial_index(store,budget)->IndexReceipt`,
`audit_level(field,store,guide,settings,stop)->AuditResult`.

- [ ] RED: guía omite isla/ramificación soportada **a distancia>tau**, truncamiento
  de ROI y cambio de componente lejano; source→grid no puede PASS por revisar
  sólo nodos proyectados. Isla omitida dentro de tau es control de sensibilidad:
  puede PASS espacial, pero nunca cambia topology UNMEASURED por preservada.
- [ ] RED: cap pequeño y tira aislada con nodos dentro de tau pero sin celdas
  visibles→BLOCKED; en salida no vacía, nodos huérfanos tampoco cubren otra zona.
  Normalización across-tile/halo quita huérfanos y reconstruye índices antes de auditar.
- [ ] RED: cuatro esquinas soportadas con gap interior, doble capa próxima, puente
  entre hojas, triángulos adyacentes que solapan más allá de su arista, cruce3D,
  contacto coplanar no explicado y costura auténtica con ascendencia verificada.
- [ ] RED: una celda activa falla la cota `d+halfdiag≤tau` hasta subdividir; leaf
  incierta al depth8, max256Msubceldas o20Mpares→BLOCKED, nunca PASS/UNMEASURED convertido a true.
  Near-zero en esquinas interpoladas exige intervalos hacia fuera: el centro
  redondeado no puede excluir una subcelda cuyo intervalo aún contiene W.
- [ ] RED: certificado usa bytes float32 entregados, no puntos float64 descartados;
  probar ambas diagonales y bound L/sqrt(3)+root_error+numeric_margin≤tau.
  Triángulos agudo/obtuso/recto y plano pitch=h,tau=h deben dar cotas conservadoras
  sin refinar4×por una arista diagonal. Error excesivo sí bloquea/refina.
- [ ] RED: pares de vecindad de rejilla se revisan en barrido local y se cuentan
  aparte de los20M no-locales únicos. La prueba de escala plano8000² no debe
  consumir el límite no-local sólo por contactos topológicos esperados.
- [ ] Run: `python -m pytest tests/test_tifxyz_audit.py -q`. Expected: RED.
- [ ] Implementar índices R-tree/SQLite paginados y auditores del diseño§5.3.
  Probar capacidad R-tree antes del trabajo. No usar cKDTree/AABB de N nodos
  finales en memoria. Bounding boxes se expanden conservadoramente por redondeo;
  predicados inciertos bloquean. Cache global≤64MiB, cursor fetchmany≤4096filas,
  sin ORDER BY/DISTINCT/global temp. No confundir LIMIT con cota de trabajo.
- [ ] Run mismo comando. Expected: PASS; añadir control positivo plano completo
  y control negativo de hueco muy pequeño entre muestras para evitar un test vacuo.

### Task 6: Gate de viabilidad y publicación independiente

**Files:** `tifxyz.py`; `tools/bench_direct_tifxyz.py`; tests `test_tifxyz.py`;
`docs/verification/direct-field-tifxyz-feasibility.json` durante ejecución.

**Interfaces:** `export_field(field_run,out,settings,resume=False,stop=None)->dict`,
`verify_export(out,stop=None)->dict`. Export consume T1–T5 y devuelve
READY/PAUSED/BLOCKED o error de integridad. Verify sólo devuelve
TIFXYZ_VERIFIED al validar READY; de otro modo resultado no exitoso/error,
sin publicar ni modificar la salida.
El motor privado `_export_source(field,out,settings,resume,stop)` consume
PhysicalField ya validado; benchmark puede usar su adaptador histórico explícito.
El CLI sólo abre su cadena de FIELD_RUN, nunca fabrica un RUN para otra fuente.
`verify_export` devuelve TIFXYZ_VERIFIED y no publica; la verificación del adaptador
histórico se invoca desde benchmark con su fuente y binding exactos.

- [ ] RED: aprobar sólo si fuente, todos los bloques, auditoría, metadata y TIFF
  concuerdan; modificar cualquier hash, selección de nivel, scale, bbox, frame,
  máscara, flags físicos o mapeo de rejilla debe fallar. READY es último archivo.
- [ ] RED: verificador en proceso nuevo vuelve a admitir recursos y lee fuente
  fijada; no confía en presupuestos/conteos de manifest; una permutación de rejilla
  que conserva distancias no puede pasar. Reproyectar tiles y comparar usando
  brackets/tolerancias, validando que cada nodo pertenece a su UV/rayo correcto.
- [ ] Run: `python -m pytest tests/test_tifxyz.py -q`. Expected: RED; implementar
  reintentos guía/pitch acotados y publicación; repetir hasta PASS.
- [ ] Fijar `PILOT.json` antes de ejecutar: las cuatro ROIs512³celdas de W−4 de
  diseño§7.1, fuente compuesta histórica verificada y adaptador de lectura congelado.
  Contrastar configSHA con el original; comprobar conformance12tiles/127casos sin
  cambiar fuentes. No fabricar FIELD.json ni mover ROIs tras ver resultados.
- [ ] Ejecutar positivos sintéticos plano/cilindro/hoja deformada/anisótropo y
  negativos definidos por mecanismo, además de ROIs reales prefijadas. Separar
  oráculo MC de baseline de optimización: extracción→UVglobales/runtime03→TIFXYZ,
  más auditor común nuevo en ambos resultados. Nunca CLI flatten ni MC solo.
  El baseline recertifica distancia de sus nodos al campo; no hereda el root_error
  del productor directo. No debe excluir componentes ni caras para poder comparar.
- [ ] Medir mediante supervisor RSS real+temporales+I/O+tiempo de todas las fases.
  **GO sólo** si positivos producen READY, cada negativo cumple su contrato y
  al menos una ROI real prefijada con baseline≥250kcaras/componenteúnico produce
  READY en ambos: pico nuevo≤3GiB y≤50%baseline, tiempo≤2×baseline. Guardar todas
  las ROIs/resultados/tamaños/comandos, no sólo la favorable. Los controles de
  topología dentro de tau no se cuentan como negativos de cobertura.
- [ ] Medir apertura/parsing/metadatos: manifest bytes, chunk count, read_calls,
  hashes repetidos y rchar/read_bytes. Source con manifiesto real/grande cercano
  al límite debe estar en pruebas de recursos, no sólo fixtures de un chunk.
  Fijar max_pixels4.194.304 para ambas rutas ROI, sin cambiar tolerancias, y
  usar KERNEL-BENCH.json como fuente exacta del conjunto de conformance histórico.
- [ ] Si NO-GO: detener tareas7–9, documentar causa y volver al usuario; no instalar
  una ruta “directa” que internamente vuelva a mesh/flatten ni bajar resolución fina.

### Task 7: Rama CLI y compatibilidad de los runs existentes

**Files:** `cli.py`, `config.py`, `workflow.py`, tests `test_cli.py`,
`test_tifxyz_cli.py`, documentación de opciones. Sólo tras GO T6.

**Interfaces:** `stage_path(config)->tuple[(stage,receipt)]`; rama legacy fija
prepare/fibres/field/mesh/flatten, nueva prepare/fibres/field/tifxyz.
`surface_output` default atlas para configs antiguas sin modificar su JSON.
Las opciones standalone y `verify-tifxyz` llaman T6 sin `workflow.run` del origen.

- [ ] RED: run directo y resume nunca llaman mesh/flatten; monkeypatch esas
  funciones para lanzar AssertionError y afirmar ausencia de meshes/flat/OBJ.
  No basta inspeccionar el texto del código. Pipeline legacy conserva sus outputs.
- [ ] RED: `--until field` pausa recordando tifxyz; resume completa esa rama;
  `--until mesh/flatten` incompatible se rechaza antes de preparar datos.
  `plan` estima sólo productos de la rama elegida, no requiere espacio de malla nativa.
- [ ] RED: cambiar el archivo JSON original tras pausar no cambia el request
  congelado; JSON/--levels contradictorios se rechazan antes de I/O caro. Probar
  defaults de --until por rama, separate reserves y JSON desconocido/duplicado.
- [ ] RED: export standalone desde FIELD_READY previo no cambia un byte/mtime
  del origen ni abre Volume/CT/red. No relajar implementation SHA de resume viejo.
  Tests con origen sólo lectura, CT inaccesible y output distinto bajo nuevo lock.
- [ ] RED: `snapshot_exports`, `_validate_products`, estado COMPLETE y gallery no
  asumen FLAT.json; falta/corrupción de un TIFF tras READY se detecta al reanudar.
- [ ] RED: BLOCKED jamás se registra como etapa completa, ni llama gallery de
  atlas. Aplicar tabla§6.1: PAUSED0, BLOCKEDresource75, BLOCKEDgeometry2, FAILED1;
  sólo READY registra TIFXYZ.json y permite COMPLETE. Mantener exits legacy.
- [ ] RED: export integrado→padre escribe COMPLETE→verify en proceso nuevo pasa;
  export PAUSED→padre actualiza RUN→resume conserva identidad. Cambiar FIELD,
  checkpoint, source/frame o producer implementation sí invalida el binding.
- [ ] Run: `python -m pytest tests/test_tifxyz_cli.py tests/test_cli.py -q`.
  Expected: RED; implementar rama mínima y defaults compatibles; repetir hasta PASS.
- [ ] Documentar ejemplos integrados/standalone y dependencia opt-in; ninguna
  descarga/instalación de libigl/tifffile ocurre durante ejecución.

### Task 8: Lectura real VC3D, campo real y resource regression

**Files:** tests `test_tifxyz_vc3d.py`, benchmark T6, `docs/FORMATS.md`, README,
`docs/verification/direct-field-tifxyz-acceptance.json` durante ejecución.

**Interfaces:** consume la salida final de CLI instalado y export T6, no helpers
internos como oráculo. No cambiar implementación para satisfacer una imagen visual.

- [ ] Lector real `vc_tifxyz2obj --keep=100 --no-inpaint`: emparejar por UV/índice
  de rejilla, verificar XYZ, máscara, scale no unitario, origen/ROI, todas las
  caras y diagonales, incluidos tiles finales parciales. Fuente intacta.
- [ ] Fuente real: checkpoint del CLI fijado y ROI explícita de una hoja;
  baseline desde el mismo campo. Registrar GUIDES/ROOTS/COVERAGE/BUDGET, no sólo foto.
- [ ] Stop/resume real durante proyección, auditoría y ensamblado TIFF; comparar
  payloads/hashes finales y nueva verificación completa en proceso independiente.
- [ ] Linux: geometrías de tamaño creciente con límite real3GiB y espacio acotado;
  comprobar consumo agregado worker+supervisor+cache, no sólo un array NumPy.
  Si no cabe, BLOCKED sin tocar otros jobs. Tests de queries muy densas y R-tree
  degenerado prueban que no se hace fetchall sobre toda la superficie.
- [ ] Run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q`.
  Expected: suite completa PASS; separa skips de dependencia/plataforma/VC3D.
- [ ] Build wheel en entorno autorizado, instalar en venv existente de verificación,
  ejecutar tests/CLI fuera del checkout; comprobar assets/licencias y lazy imports.

### Task 9: Gate de escala y entrega

**Files:** benchmark T6 y recibo de escala; documentación. No nueva abstracción.

- [ ] Antes de W−4 completo, revalidar el campo/adaptador histórico ya fijados en
  T6, incluyendo unidades y soporte. No atribuir resultados de un campo CLI nuevo
  al W−4 antiguo ni extrapolar el éxito de una ROI a todo Q1.
- [ ] Preflight real de RAM, scratch, source I/O y export frame. GPU ajena/Work3,
  alquileres, limpieza o ampliaciones no son autorizaciones de este plan.
- [ ] Ejecutar un nivel grande sólo con recursos admitidos, medir todos los intentos
  y validación. Si no pasa, conservar evidencia y declarar ese nivel no resuelto.
  No extrapolar121niveles desde una ROI o desde el principal de W−57.
- [ ] Confirmar ahorro estructural: cero native mesh/OBJ/atlas en call graph y
  artefactos, guía bajo límite, techoRAM observado y costes completos publicados.
- [ ] Revisión independiente de implementación, correcciones RED→GREEN y suite;
  `git diff --check`; entregar rutas, comandos, métricas y límites. Commits/push
  siguen requiriendo petición explícita; no tocar el plan Qual V2 ni su ejecución.

## Matriz de cobertura y handoff

| Contrato | Propietario / prueba |
|---|---|
| ahorro real, no cadena oculta |T2 no_native_fallback; T7 call graph; T9 artefactos |
| root correcto / sin salto de rama |T3 cúbica3raíces, tangencia y chunk boundary |
| fuente completa a tolerancia |T5 isla omitida y subdivisión exhaustiva acotada |
| no certificar topología/hoja física |T1/T6 metadata exacta; README y recibos |
| marcos y unidades |T1 transform literal; T8 lector real independiente |
| RAM/disco acotados también en verify |T1/T4/T5/T6/T8 |
| resume/publicación segura |T4 fault injection; T6 READY; T7 ruta |
| interoperabilidad y regresión legacy |T7/T8 suite+wheel |
| utilidad práctica medida |GO/NO-GO T6 y gate de escala T9 |

El plan termina en entrega para aprobación, no en implementación automática.
Recomendación de ejecución: inline secuencial con revisión independiente al cerrar
T6 (decisión de viabilidad) y al final; no paralelizar los contratos geométricos
interdependientes. Los límites numéricos son decisiones propuestas del autor,
no cifras medidas ni exigencias previamente dadas por el usuario.

## Autoauditoría del autor

Revisadas las secciones del diseño contra T1–T9 y las interfaces compartidas.
Corregidos antes de revisión externa: tipos auxiliares faltantes; presupuesto
agregado del controlador; cuota real de índices frente a estimación; política
de reintentos guía/pitch; I/O repetido en lotes; fuente histórica175A distinta
del formato del CLI. No se ejecutaron tests o benchmarks del feature: aún no existe.
