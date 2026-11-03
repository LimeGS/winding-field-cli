# Extracción directa del campo a TIFXYZ — diseño propuesto

Fecha: 2026-10-29. Estado: propuesta para aprobación; no implementación.
Repositorio: `/Users/user/Software.nosync/Factory/winding-field-cli`, base `6b84457`.

Approved amendment2026-10-30: the user selected quality-first experimental
integration after the reference global UV solver exhausted the available bounded
envelope. Require full direct-output geometry/integrity validation and<=3GiB;
defer comparative RAM/time ratios explicitly as UNMEASURED. No quality, source,
resolution, or reserve gate is waived. The historical comparison specification
below remains documented as the uncompleted performance experiment.

## 1. Intención y diferencia real

El usuario quiere ahorrar una etapa geométrica, no envolver OBJ→UV→TIFXYZ en un
comando. La ruta nueva será **campo fijado por SHA → guía acotada → rejilla sobre
el campo fino → TIFXYZ**, sin extraer/ensamblar la malla triangular nativa completa.
Los nodos finales se localizarán en el campo fino; no se interpolará simplemente
la geometría gruesa como resultado final.

Hoy `meshes.mesh` llama a `_core.surface_mesh.extract_mesh`, que materializa todos
los triángulos en SQLite y exporta OBJ. `flatten.flatten` ya lee SQLite sin pasar
por OBJ, y divide en charts de1024caras por defecto. `preview.preview` rasteriza
XYZ por chart, pero su límite de256píxeles es visual, no un producto TIFXYZ global.
Guardar esos previews o ejecutar el conversor existente no satisface este diseño.

## 2. Contrato de producto y alcance

- Un mapa UV global y un directorio TIFXYZ por nivel solicitado y ROI declarada.
- La ruta directa no llama a `meshes.mesh`, `flatten.flatten`, ni al conversor
  OBJ→TIFXYZ; no crea `meshes/`, `flat/`, OBJ nativos ni tablas de caras nativas.
- Una guía pequeña puede tener triángulos y costuras topológicas. Sus límites
  son duros y no se elevan automáticamente hasta convertirla en la malla nativa.
- V1 exige niveles explícitos. La selección automática de niveles del flujo
  histórico permanece intacta y no es un prerrequisito de la ruta directa.
- No se unen componentes desconectados, no se empaquetan charts y no se elige
  silenciosamente el mayor componente. Una guía con varios componentes bloquea
  ese nivel. Se puede trabajar sobre una ROI explícita; nunca llamarla todo el W.
- Una guía única no demuestra que el campo fino tenga un solo componente. La
  auditoría recorre el dominio soportado completo y exige cobertura espacial a
  tolerancia; no certifica igualdad de topología ni cuenta componentes nativos.
- `READY` significa resultado geométrico verificado con el alcance y tolerancias
  declarados. Siempre `physical_acceptance=false`, `native_face_identity=false`,
  `native_topology_equivalence="UNMEASURED"`. No hay conteos de caras nativas inventados.
- Si se exige conservar exactamente la triangulación/topología de los OBJ, esta
  ruta no es un sustituto contractual. Esa exigencia requiere otro trabajo.
- No se modifican el solver W, entrenamiento, SD2/Qual, trazadores, CT, backups,
  buckets o IAM. El plan Qual V2 existente se preserva como trabajo independiente.

## 3. Entradas, unidades y salida

Entrada primaria: `FIELD.json` + checkpoint referenciado, por `FieldSource`, con
política `observed` idéntica a `open_field`. NaN, huecos o chunks ausentes son
inadmisibles; no se rellenan ni se sustituyen por el prior/coarse field.

El adaptador expone lectura ZYX y geometría XYZ **en micrómetros**. Para fuentes
generales: `origin_um=source.origin*source.voxel_um` y análogamente spacing; para
el CLI actual `voxel_um=1`. W y los isovalores permanecen en unidades del campo,
independientes de unidades XYZ. Lectura y hashes siguen siendo responsabilidad
del `FieldSource` existente, que no se reimplementa ni relaja.

El origen debe pasar esta cadena read-only: snapshot del `RUN.json` leído, hash de
`FIELD.json` igual a `RUN.stages.field.sha256`, status FIELD_READY, SHA del
checkpoint igual al declarado y geometría embebida consistente. Para frame
derivado exigir `digest(RUN.source)==FIELD.geometry.frame`. Guardar el SHA de
implementación del productor antiguo como procedencia y el del exportador nuevo
como otra identidad; no exigir igualdad entre ambos. Usar `FieldSource`
directamente: importar `field.open_field` arrastra `block_solver` y torch.

`RUN.json` es mutable: su hash completo identifica el snapshot histórico
guardado como `producer-run.snapshot.json`, no exige que el archivo vivo siga
idéntico. La identidad estable del origen es el digest de: schema/implementation
del productor, `RUN.source`, binding `{receipt,sha256}` de la etapa field (sin
seconds), geometría FIELD y descriptor/hash checkpoint. Excluir status, tiempos,
otras etapas y el recibo de TIFXYZ para evitar una dependencia circular. En resume
y verify, contrastar ese subconjunto estable con el snapshot y revalidar FIELD,
checkpoint/payloads; permitir que el padre registre PAUSED/COMPLETE o nuevas etapas.
Esto no cambia la regla de implementation SHA del resume legacy del productor.

**Límite explícito de metadatos:** todo open/parse/validate ocurre dentro del
worker supervisado. V1 admite manifiestos de≤16MiB y≤65.536chunks referenciados;
reservar en RAM `64*manifest_bytes + 2048*chunk_count` además del cache/crops.
Comprobar bytes antes de parsear y count dentro del límite del worker. El lector
congelado vuelve a leer/hashear el manifiesto antes/después de cada read: medir
read_calls, bytes de manifiesto procesados y `/proc/PID/io` (rchar y read_bytes)
además de payloads/cache. No asumir que cache_bytes limita esos metadatos ni
suprimir esos hashes. Si este coste hace fallar el gate real, NO-GO y propuesta
separada de lector snapshot/indexado; no integrarlo a escondidas en este plan.

Standalone V1 permite `roi={lo_zyx,hi_zyx}` en índices de **nodos del campo**,
half-open, al menos2nodos por eje y dentro del checkpoint. No son voxels CT ni
coordenadas locales del OBJ. Omitir ROI usa todo el dominio persistido. La vista
desplaza origen físico por `lo_xyz*spacing_um`, conserva los valores/soporte y
no lee stencils fuera de esa ROI. Ésta se fija en identidad y alcance; nunca se
recorta automáticamente al bounding box de la guía.

El JSON de exportación incluye `output_frame={id, origin_xyz_um, voxel_um,
shape_zyx, source_identity_sha256}`. V1 admite un frame destino de vóxeles
isotrópicos, con origen y unidad positivos/finitos según corresponda. No se
deducen L0, ROI offsets o nivel de pirámide de nombres/rutas. Puede derivarse el
frame de `RUN.source` sólo si es isotrópico, está SHA-vinculado y se declara como
el **nivel seleccionado**, no automáticamente L0. Un destino distinto exige
especificación explícita y identidad contrastable del volumen.

Transformación única: `xyz_tif=(xyz_um-output_origin_um)/output_voxel_um`.
Rechazar nodos válidos fuera de `[0, shape_xyz-1]`. Inválidos: los tres canales=-1.
Con pitch UV físico `p`, `meta.scale=[output_voxel_um/p, output_voxel_um/p]`.
La rejilla usa `uv_min+(column,row)*p`; se conserva el pitch exacto, sin estirar
la última fila/columna para ajustar el bounding box. Filas/columnas adicionales
fuera del dominio son inválidas. La resolución UV no implica isometría física.

Salida: `level-NNNN/{x.tif,y.tif,z.tif,meta.json,VERIFIED.json}`; índices estables
según la lista de niveles, valor W explícito en meta. TIFF float32 tiled256²,
BigTIFF cuando corresponda, deflate, sin cargar una imagen completa en RAM.
`TIFXYZ.json` agrega resultados; `READY.json` se publica al final, tras verificar
todo. Las guías, bloques y recibos permanecen fuera de los directorios de segmentos.

## 4. Parámetros y límites iniciales (decisiones propuestas)

Un único `--tifxyz-config` JSON evita decenas de flags. Valores normalizados y
efectivos quedan fijados en la identidad de exportación, con validación antes de I/O caro.

| Campo | Valor inicial / significado |
|---|---|
| `levels` | lista explícita, finita, única, máximo4096; orden conservado |
| `pitch_um` | por defecto mínimo spacing físico del campo |
| `tolerance_um` | por defecto máximo spacing físico del campo; nunca se relaja automáticamente |
| `memory_gib` |3 de presupuesto RSS agregado exportador+supervisor, además de2GiB de reserva host/cgroup |
| `scratch_gib` |16 de máximo de trabajo; reserva libre12,25GiB en ejecución gpu-1 |
| `max_pixels` |64.000.000 por nivel; overflow y dimensiones se validan antes de reservar |
| límites guía |4.000.000nodos gruesos y250.000caras finales; stride≥2 |
| intentos guía |miembros admisibles de[4*s_min,2*s_min,s_min], máximo3; s_min≥2 |
| UV |armónica positiva +50iteraciones SLIM, borde convexo exacto |
| campo/cache |tiles de64³celdas+halo1; cache128MiB; lotes de≤16.384consultas |
| SQLite |cache agregado≤64MiB; sin sorts/tmp globales; cursor paginado y journalDELETE (§6) |
| raíces |≤4096celdas por rayo, tras clipping al ROI; ancho final≤min(spacing_um)/1000 |
| auditoría |profundidad de subdivisión≤8;≤256Msubceldas;≤20Mpares geométricos |

Los límites de algoritmos son parte de `algorithm_version`, no knobs ocultos.
Agotar presupuesto produce `BLOCKED`, no PASS parcial ni cambio de tolerancia.
3GiB es objetivo a probar, no memoria medida ni promesa de que W−4 funcione.
El disco temporal puede superar lo libre hoy en gpu-1; se rechaza antes de
reservar, sin borrar originales ni usar S3 como disco aleatorio/mmap.

## 5. Algoritmo

### 5.1 Preparación acotada de la guía

1. Hallar el menor stride s_min potencia de2, mínimo2, cuya rejilla tenga≤4M
   nodos y≥2nodos por eje. Intentos: miembros admisibles de
   `[4*s_min,2*s_min,s_min]`, en ese orden, sin duplicados. Si no existe rejilla
   admisible, COARSE_GRID_UNREPRESENTABLE. La rejilla gruesa conserva centros/origen;
   último nodo nunca fuera del ROI. No convertir s=1 en fallback.
2. Leer el campo por tiles y reunir los valores gruesos. Una celda gruesa sólo
   es soportada si lo son **todos** los nodos finos que abarca, no sólo sus8esquinas.
   Persistir también mínimos/máximos por tile fino+halo, vinculados al SHA fuente,
   para descartar después tiles que no pueden contener el nivel.
3. Extraer únicamente la guía gruesa con Marching Cubes. Ejecutar esta operación
   en worker limitado incluso antes de conocer cuántas caras genera. Rechazar
   exceso de caras, no manifold, orientación incoherente o más de un componente.
4. Reutilizar/adaptar los cortes conservativos, UV armónicas, SLIM y validadores
   del kit ya probado (`vesubius-challenge/tools/mesh_to_tifxyz/geometry.py`). No
   copiar su lector OBJ, selección largest, raster 3D nativo ni estimación60GiB.
   Registrar procedencia; añadir libigl/tifffile sólo como extra opt-in `tifxyz`.
5. Guardar guía en XYZ µm, caras, UV µm, normales por vértice antes de duplicar
   costuras y correspondencias de costura. Verificar orientación consistente,
   disco, Jacobianos UV positivos, borde simple convexo y geometría3D de la guía.
   Un mapa armónico aislado no basta: W−57 ya demostró compresión problemática.

### 5.2 Proyección directa sobre el campo fino

Para cada nodo UV, localizar su triángulo guía, interpolar posición `g` y normal
unitaria `n` (normales continuas interpoladas; mantener equivalencias en costuras).
Buscar sobre `g+t*n` dentro del ROI y `|t|≤R`, con
`R=2*norm(s*spacing_um)` para el stride efectivo. No aumentar R en silencio.

- Recorrer celdas finas por DDA, no por muestreo de pasos que pueda saltar raíces.
- En cada celda, W trilineal restringido a una recta es un polinomio grado≤3.
  Aislar **todas** las raíces usando extremos de la derivada, partición en
  intervalos monótonos y bisección. Newton no bracketed y mero cambio de signo
  en extremos del rayo están prohibidos. Raíces dobles/tangentes o clasificación
  numérica incierta son `AMBIGUOUS`, nunca una raíz elegida arbitrariamente.
- Deduplicar raíces compartidas por celdas mediante propiedad half-open global,
  con tratamiento explícito del extremo máximo del ROI. No usar tolerancias de
  deduplicación mayores que el bracket para fusionar dos raíces reales.
  La propiedad decide conteo, no simplicidad: en un límite interno examinar
  ambos segmentos incidentes; el cusp con samples[1,0,1], nivel0, es AMBIGUOUS
  aunque cada polinomio lateral sea lineal. Endpoints del ROI se admiten sólo con
  igualdad certificada y pendiente lateral no nula; declarar clasificación one-sided.
- Un nodo válido exige exactamente una raíz simple, soporte completo de su
  interpolación, bracket geométrico acotado y residual recomputado desde el campo.
- Falta de raíz, múltiples raíces, normal degenerada o soporte desconocido en la
  búsqueda no se convierten en píxeles transparentes para aprobar: bloquean el
  intento. Sólo el exterior del dominio UV y huecos de soporte declarados pueden
  invalidarse; la auditoría fina debe demostrar que no se perdió nivel soportado.
- Agrupar consultas por celda/tile y respetar cache total. Evitar `FieldSource.read`
  por píxel y BVH sobre toda la malla nativa, que no existe en esta ruta.

Contrato numérico del aislador: t local por celda normalizado a[0,1]; coeficientes
escalados por su máximo absoluto, conservando factor. Acotar operaciones/signos
con redondeo hacia fuera; coeficientes compatibles con cero no se eliminan por
epsilon arbitrario. Si grado, discriminante, signo o separación de raíces no
pueden clasificarse, AMBIGUOUS. DDA avanza todos los ejes empatados, maneja
direcciones0, rayos invertidos y caras exactamente alineadas. En límites se
identifican raíces por el plano/nodo global y se comprueban ambos lados; no se
deduplican sólo por distancia. El error geométrico almacenado incluye bracket
en t físico, interpolación/evaluación y desplazamiento al codificar float32.
El residual de W es diagnóstico adicional, no conversión arbitraria de W a µm.

Antes de procesar un rayo, acotar su número de celdas por cruces de planos del
segmento clippeado. Si excede4096, RAY_WORK_LIMIT; las colas contienen≤16384
segmentos en total, nunca 16384×4096. El plano anisótropo(1,1,100)µm forma parte
del positivo de viabilidad, para detectar límites que rechacen geometría sencilla.
El índice min/max SHA-vinculado puede descartar intervalos de tiles completamente
soportados que excluyan W, antes de evaluar sus polinomios finos. Registrar celdas
cruzadas, descartadas y evaluadas por separado; no saltar intervalos desconocidos
ni usar este descarte para esconder un exceso de presupuesto.

Cada bloque256²+halo1 escribe XYZ float32 de destino, máscara y error de raíz
en archivos acotados. No se crea un array gigante ni un mmap global de imágenes;
el cursor y hashes permiten reanudar y verificar con la misma cota de RAM.

Política de reintentos: fallos de guía, raíz ambigua/sin raíz y cobertura fina
omitida habilitan el siguiente stride predeclarado; no cambian de componente.
Distancias/aristas demasiado largas habilitan pitch/2 sobre la misma guía hasta
que la siguiente dimensión exceda max_pixels. Cada combinación guía/pitch tiene
namespace/identidad propios. Se conservan diagnóstico y coste acumulado de intentos;
su scratch computa en el límite y se detiene si no cabe, sin limpieza implícita.

### 5.3 Auditoría independiente del resultado

La auditoría consume **los floats32 que se van a entregar**, reconvertidos a µm,
y la fuente original; no sólo los resultados internos del proyector.

Primero fijar la superficie realmente entregada: celdas con4nodos válidos y
triángulos que el lector VC3D fijado emita sin inpainting. Exigir≥1celda y
≥1triángulo no degenerado. Quitar nodos que no participen en esa superficie y
reconstruir máscara/índice antes de auditar. Un nodo huérfano nunca es testigo de
cobertura, incluso cuando haya otras celdas válidas. Fijar la regla exacta y sus
diagonales con código/lector real antes de aceptar; si el lector usa otra regla,
adaptar publicación y prueba conjuntamente, sin relajar esta condición.

**Resultado→campo:** revalidar todos los nodos y sus brackets/soporte. Para cada
triángulo de una celda TIFXYZ válida, exigir área>0, orientación coherente con el
gradiente, soporte de los stencils en todas las celdas atravesadas y
`vertex_cover_um + max_vertex_root_error_um + numerical_margin_um ≤ tolerance_um`,
con `vertex_cover_um=L/2` para triángulos rectángulos/obtusos (clasificación exacta
de las longitudes al cuadrado) y `L/sqrt(3)` para los demás.
La cota de cobertura por vértices de cualquier triángulo es≤L/sqrt(3): si el
circuncentro está dentro, el ángulo mayor está entre60°y90° y su radio cumple la
cota; si está fuera, el máximo se alcanza en el borde y es≤L/2. Evaluar la cota
con redondeo hacia fuera. Así un plano isométrico con pitch=h, tau=h no requiere
cuadruplicar píxeles sólo por una cota excesiva. Refinar pitch si hace falta;
nunca aceptar sólo un promedio.
Fijar y probar las dos diagonales de celda posibles del lector antes de certificar.

**Campo→resultado:** recorrer todas las celdas finas soportadas del ROI. Si el
nivel no está entre mínimo/máximo de sus8esquinas, no hay cruce trilineal. Para
las demás, buscar un nodo final `p` que satisfaga
`distance(cell_center,p)+half_cell_diagonal ≤ tolerance_um`. Si existe, toda la
celda —y por tanto su porción de nivel— queda cubierta. Si no, subdividir la
celda, evaluar sus esquinas por la misma interpolación trilineal y repetir.
En subceldas usar intervalos redondeados hacia fuera para interpolación, distancia
y semidiagonal. Si un intervalo de valor contiene W no puede descartarse por el
valor central redondeado; sólo excluir con una cota estricta segura. Reconocer
ceros estructurales exactos sin cambiar el nivel; casos no clasificables bloquean.
Profundidad/trabajo agotados, campo identicamente igual al nivel en una región o
cualquier celda activa sin cota demostrada impiden READY. No usar sólo vértices,
centroides, p95 o porcentaje de puntos; esta prueba detecta zonas omitidas por la guía.

Esta cota espacial no prueba equivalencia topológica: dos componentes muy
próximos pueden estar dentro de una misma tolerancia. El recibo lo declara y no
lo transforma en “todos los componentes preservados”. Casos de doble capa,
puentes, pliegues y componentes pequeños son controles adversariales obligatorios.
Una isla a distancia>tau del resultado debe fallar; múltiples raíces en un rayo
buscado deben fallar. Una isla omitida que quede íntegramente a≤tau puede pasar
el contrato espacial: su fixture debe registrar esa sensibilidad y los flags
topológicos UNMEASURED, no exigir una garantía que este método no proporciona.

**Soporte e intersecciones:** no basta que las4esquinas de una celda sean válidas.
Comprobar interiores/edges contra stencils soportados. Usar un índice espacial
en disco para candidatos de nodos y triángulos, con consultas paginadas. Rechazar
intersecciones de interiores, pliegues, aristas colapsadas y contactos no explicados;
la adyacencia de rejilla permite sólo su vértice/arista común, no solapes adicionales.
Las costuras se identifican por ascendencia de guía; no se fusionan por proximidad.
Predicados numéricamente inciertos bloquean; un límite de pares no significa cero cruces.
El contador20M sólo limita pares **no locales** únicos sometidos al narrow phase.
La vecindad de rejilla se comprueba en un barrido O(píxeles) aparte, con counters
propios y sin eximir sus solapes indebidos. Consultas R-tree por cursor
fetchmany≤4096, sin ORDER BY/DISTINCT/GROUP BY global ni fetchall. Registrar
candidatos y tests efectivos; detectar trabajo patológico por el presupuesto de
fase antes de prometer una duración. El plano8000×8000 es un test de escala del
barrido/índice, no necesariamente una prueba rutinaria de CI.

## 6. Persistencia, recursos y API

- Núcleo nuevo aislado de `_core/`; sus archivos/provenance no se modifican.
- Estado de exportación: `PREPARING_GUIDE`, `PROJECTING`, `AUDITING`, `PUBLISHING`,
  `PAUSED`, `BLOCKED`, `FAILED`, `READY`. Motivo estructurado para BLOCKED.
- Un único writer por destino, lock propio, raíz canónica, rechazo de symlinks y
  destinos ajenos. Bind a source hashes, soporte, frame, niveles, parámetros,
  implementación y versiones numéricas. No reanudar con otra identidad.
- Hash y fsync de bloque antes de commit en ledger SQLite; crash antes/después
  del commit tiene recuperación definida. No confiar en un cursor sin archivos.
- Armar TIFF a un directorio `.pending` en el mismo filesystem, verificar, fsync,
  publicar segmentos y después READY como último acto. STATUS no es aceptación.
- Mismas admisiones en `verify-tifxyz`: no confiar en tamaños/estimaciones de un
  recibo externo. Parsear headers y límites antes de decodificar; no full-image reads.
- Linux es el entorno de aceptación de memoria. Worker CPU1/nice19/sinCUDA, BLAS1,
  supervisor con STOP/SIGINT, techo de direcciones y RSS observado; reservar2GiB
  intersectando host y cgroup real/ancestros. Falta de observación bloquea.
  El presupuesto incluye al controlador en la ruta integrada: medir su RSS base,
  restarlo junto con64MiB de margen antes de fijar RLIMIT_AS del hijo, y observar
  la suma durante la fase. No cargar torch en el worker ni descontar cache real
  por llamar a una asignación “mmap”. Si el controlador consume el presupuesto, bloquear.
- Estimar scratch incluyendo bloques, TIFF final y temporal simultáneos, índices
  espaciales, SQLite/journal y margen. Comprobar máximo y espacio libre por fase.
  Inicialmente reservar por nivel activo `coarse_bytes + 256*grid_pixels + 256MiB`;
  es una estimación de admisión, no una cota de tamaño de R-tree. Imponer cuotas
  reales a DB/temporales y verificar el uso agregado antes de cada lote. Índices
  patológicos pueden bloquear aun después de admitir; nunca comer la reserva.

**Mecanismo de disco:** todas las escrituras son bajo el destino canónico propio
y en su filesystem. SQLite usa páginas4096, journal DELETE, synchronous FULL,
automatic_index OFF; no WAL. No consultas que precisen sort/materialización
global; comprobar EXPLAIN QUERY PLAN. `temp_store=MEMORY` evita escapes a /tmp;
cualquier temporario inesperado queda bajo el techo de RAM y no se cuenta como
una optimización válida. No crear tablas temporales masivas.

Particionar el presupuesto por archivo: DB con cuota de un octavo del scratch
aprobado (2GiB con scratch16GiB), `max_page_count` duro, más reserva
para rollback de hasta `1.01*DB_quota+2MiB`; ledger y R-tree comparten esa DB.
Esa cifra es una **cuota impuesta**, no una supuesta cota natural de SQLite:
durante la fase de escritura SQLite, fijar RLIMIT_FSIZE al journal_cap reservado;
max_page_count aplica la cota más pequeña al DB. Cerrar payloads grandes antes
de esa fase y cambiar el límite soft únicamente en transiciones de fase admitidas.
Un spill que exceda la cuota produce FILE_SIZE_LIMIT/BLOCKED, termina/reabre
la conexión y verifica recuperación del journal; jamás se continúa con un
cursor dudoso ni se publica READY. Ensayo con cache pequeño y spill debe probar
que los commits anteriores sobreviven. Supervisar también terminación SIGXFSZ.
Antes de cada transacción reservar crecimiento restante del DB hasta su cuota y
ese journal máximo, además de payloads/outputs simultáneos. INSERTs por lotes≤128;
SQLITE_FULL/ENOSPC conserva bloques committed y no publica READY. Esta reserva
conservadora no supone que LIMIT acote escrituras o trabajo interno.

Bloques numéricos con tamaños derivados de headers conocidos; escribir en
chunks≤8MiB verificando límites. Por canal TIFF, reservar techo
`5*pixel_count + 1MiB + 4096*tile_count`, fijar RLIMIT_FSIZE durante ensamblado y
revalidar tamaños; sólo tags estáticos/offsets, nada de metadata proporcional sin
presupuesto. Mantener espacio para final+pending simultáneos. Si el encoder no
respeta ese techo, bloquear. Medir tanto st_size como `st_blocks*512`; admitir
con f_bavail y usar el mayor cargo más margen por filesystem, no contabilizar
archivos sparse como espacio libre garantizado. Todos los intentos retenidos y
los productos terminados consumen el mismo presupuesto del destino; si el lote
no cabe, se bloquea, sin borrar ni migrar outputs automáticamente. La reserva
libre se comprueba antes de operaciones acotadas; no se promete aislamiento de
escritores ajenos que puedan consumir el mismo filesystem concurrentemente.

API pública propuesta:

```text
winding-field run ... --surface-output tifxyz --tifxyz-config export.json
winding-field tifxyz FIELD_RUN --output NEW_OUTPUT --config export.json
winding-field tifxyz FIELD_RUN --output NEW_OUTPUT --resume
winding-field verify-tifxyz NEW_OUTPUT
```

Ruta integrada: prepare→fibres→field→tifxyz; el default histórico continúa como
prepare→fibres→field→mesh→flatten. `resume` recuerda la rama; `--until field`
puede pausar ambas. `--until mesh/flatten` en rama tifxyz se rechaza temprano.
El comando standalone lee un FIELD_READY antiguo sin modificar RUN, STOP, locks,
exports ni source.sqlite del origen y sin abrir/descargar CT. Es la forma de
reusar campos tras un cambio de implementation SHA; no se debilita resume viejo.

### 6.1 Configuración y transición de estados

Ejemplo normativo (roi=null significa todo el campo persistido):

```json
{
  "schema": "winding-field-tifxyz-v1",
  "levels": [-4.0],
  "roi": null,
  "pitch_um": 17.28,
  "tolerance_um": 17.28,
  "memory_gib": 3,
  "scratch_gib": 16,
  "reserve_disk_gib": 12.25,
  "max_pixels": 64000000,
  "output_frame": "selected-source"
}
```

En standalone, roi también admite `{lo_zyx:[...],hi_zyx:[...]}` en nodos del
checkpoint. `output_frame` alternativo es el objeto explícito de§3. Rechazar
claves desconocidas/duplicadas y JSON no finito. `reserve_disk_gib` es distinto
de la reservaRAM2GiB; normalizarlos a nombres separados en Settings.

La configuración JSON debe leerse, validarse y copiarse por contenido a
`RUN.config.tifxyz_request` al crear el run integrado, antes de `--until field`.
Guardar su digest; no reabrir el pathname original durante resume. Defaults
dependientes del campo se resuelven una sola vez al iniciar exportación y se
guardan además como `effective_settings`. En rama tifxyz, JSON levels es
autoritativo: `--levels`, si también se da, debe coincidir exactamente en orden
y valores. No modificar configs antiguas para añadir defaults.

`--until` sin especificar resuelve flatten para atlas y tifxyz para tifxyz.
`resume` utiliza esa elección guardada. Standalone `--resume` usa su snapshot;
si se proporciona nuevamente config, exigir igualdad por contenido. No reusar
un directorio creado por otra ejecución/identidad.

| Resultado hijo | Estado padre | ¿Registrar etapa completa? | Exit CLI nueva |
|---|---|---|---|
| READY, todos los niveles | COMPLETE |sí, receipt TIFXYZ.json+READY vinculados |0 |
| PAUSED por STOP |PAUSED |no |0 |
| BLOCKED por recursos/límite trabajo |BLOCKED |no |75 |
| BLOCKED por geometría/ambigüedad |BLOCKED |no |2 |
| error de integridad/IO no recuperable |FAILED |no |1 |

`verify-tifxyz` es read-only y sólo devuelve `TIFXYZ_VERIFIED` con exit0 tras
validar una publicación READY; nunca crea READY ni promueve candidatos.
Las salidas de comandos legacy mantienen sus códigos/semántica existentes.

## 7. Pruebas y puertas de decisión

1. **Viabilidad primero:** plane/cylinder/warped-sheet, raíz triple sobre una recta,
   gap, componente omitido y doble capa. Prototipo acotado de guía/proyección/auditor
   antes de integrar workflow o declarar compatibilidad de producción.
2. Baseline geométrico de malla nativa sólo en ensayos acotados, en directorio
   separado. No es fallback ni dependencia del comando directo. Comparar contra
   el mismo checkpoint/soporte/frame/tolerancia, no contra un field de otra resolución.
3. Aceptación de optimización en tamaño intermedio donde el baseline complete:
   picoRSS nuevo≤3GiB y≤50% del baseline; tiempo total nuevo≤2×baseline (incluye
   hashes, índices, todos los intentos y auditorías). Si no cumple, no publicitar
   ahorro práctico: registrar NO-GO y replantear antes del despliegue grande.
4. Campo real con ROI explícita, lector VC3D real, STOP/reanudación y verificador
   en proceso nuevo. Informar por separado cobertura, errores, RAM, scratch,
   bytes leídos/hashes, cache misses y tiempo por fase.
5. W−4/escala60Mcaras es un gate posterior, no una promesa. Sólo usar su campo
   original con adaptador verificado; el checkpoint corregido175A histórico no
   es automáticamente un FIELD.json del CLI. No descargar todos los OBJ ni
   regenerar todo el campo para poder iniciar la implementación.

### 7.1 Comparación equivalente y primer caso útil propuesto

Separar dos controles:

- **Oráculo geométrico:** MC fino sobre una ROI acotada; compara geometría, pero
  su tiempo por sí solo no es baseline de conversión completa.
- **Baseline de optimización:** extracción nativa desde ese mismo campo/ROI →
  UV globales del kit congelado `mesh-to-tifxyz-kit-20261029/runtime-03` → un
  TIFXYZ, más el mismo auditor espacial/soporte/intersecciones nuevo. No utilizar
  `winding-field flatten`, que produce charts. Sin elegir largest: el caso del
  comparador debe tener un componente y no excluir caras. Adaptar explícitamente
  sus coordenadas al mismo frame destino antes del conversor, con unit_um correcta.

El auditor común exige cotas de distancia al campo para los nodos de ambos
resultados. Para el baseline MC, recalcularlas desde el campo, con raíces
bracketed en dirección del gradiente y radio tau; no atribuirle el error de raíz
pequeño de la ruta directa. Si no puede certificarse, el baseline es inadmisible.
La verificación de procedencia UV es específica de cada productor; las exigencias
de geometría entregada, soporte y tolerancia son iguales. Pitches finales pueden
diferir si ambos pasan; informar tamaños para que no se oculte un coste trasladado.

Medir desde apertura/hash fuente hasta publicación verificada, incluyendo
intentos fallidos, normalización de frame, índices y auditor común. Misma CPU1,
BLAS1, entorno/versiones y almacenamiento; registrar orden/caches, repetir en
orden inverso si la diferencia depende de cache. Comparar en secuencia, no
simultáneamente. Si el baseline no termina, no calcular un ratio ficticio.

**Primer caso real útil, propuesto para aprobación:** W−4 histórico175A,
no todo Q1 sino cuatro ROIs prefijadas de512³celdas finas (513³nodos),8,847mm
por lado. Elegir centros de los tiles intersectados por W−4 en percentiles10,
35,65,90 de sus IDs ordenados en los recibos nativos verificados; clamp de lo a
`[0,source.shape-513]`, hi=lo+513, deduplicar y guardar PILOT.json antes de ejecutar.
No mover ROIs después de observar resultados. Fuente exacta:
`CoarseDetailSource(native,coarse_delta,alpha=1.5843640137880175)` del runtime
congelado, con configSHA `4fdac93398fa748a51d4b43d0e27c89b041ef41ee880dff44df661b2d6594d81`.
El binding autorizado se contrasta con el recibo original antes de leer; un SHA
que no coincida es fallo, nunca se actualiza a lo que haya en disco.

El adaptador histórico es sólo benchmark, no un falso FIELD_READY del CLI ni una
reimplementación de su corrección. Comprobar conformance de valores/soporte/frame
contra el lector congelado antes de usarlo como PhysicalField. Es adelanto acotado
del adaptador que T9 necesita; no requiere regenerar CT/campo o descargar OBJ completos.
Referencias: `vesubius-challenge/experiments/winding-field-exp/175a-native-meshes-20261026/`
`{FULL-CONFIG.json,KERNEL-BENCH.json,benchmark_kernel.py}`; el conjunto12tiles/127casos
se toma de KERNEL-BENCH, no de una selección nueva favorable. Congelar SHA de estos
recibos y del runtime original al preparar PILOT. Para comparación ROI usar
max_pixels4.194.304 en ambos métodos, sin cambiar tau/pitch; el cap64M es del gate
de escala. Evita que el presupuesto máximo de pixels infle artificialmente la
admisión de RAM del conversor baseline cuando la salida real es mucho menor.

GO exige al menos una ROI prefijada, con baseline de≥250kcaras, un componente,
READY en ambos métodos y los ratios de§7. Si ninguna cumple, NO-GO/NO-COMPARISON,
informando las cuatro; no reemplazarlo por un éxito sintético o una ROI minúscula.
W−57 principal puede ser un control adicional de corrección, nunca el baseline
de memoria relativa por su overhead fijo dominante. W−4 completo sigue siendo
un gate posterior. Las restricciones del método/ROI son parte de esta propuesta
que el usuario debe aprobar, no preferencias ya atribuidas al usuario.

Un lote de niveles se ejecuta secuencialmente compartiendo la guía gruesa/minmax
cuando su identidad coincida. Las auditorías pueden releer tiles por nivel; ese
I/O cuenta en el benchmark. No se promete una pasada única por121niveles ni un
tiempo de lote extrapolado de un solo nivel. La primera entrega se decide por
los gates de un nivel; el benchmark de lote se informa separadamente.

Un resultado BLOCKED en un caso fuera del dominio no es un bug, pero tampoco
demuestra que la necesidad de exportar ese W esté resuelta. No se llama “perfecto”
a un algoritmo no medido: la auditoría del plan verifica decisiones y pruebas;
la ejecución futura debe demostrar los gates empíricos.
