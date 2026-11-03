# Mallas, aplanado global y render CT rápidos

Fecha: 2026-10-31. Estado: **aprobada por el usuario; implementación en curso**.

Ampliación aprobada: cancelación segura cuando tarde demasiado, conservando el
trabajo avanzado para reanudar. Ctrl+C/STOP se comprobarán entre unidades acotadas;
si una operación numérica o de I/O no responde, el supervisor puede terminar sólo
su worker y se recupera el último checkpoint confirmado. Los archivos incompletos
no cuentan como productos publicados. Tests deben cubrir parada cooperativa y
terminación del worker, no sólo el camino exitoso.

Decisión de arquitectura tras el perfil real: el usuario autorizó acelerar el
almacenamiento/ensamblado3D, y preguntó por su ubicación. Se implementará en
`native_mesh.py`, fuera del código vendorizado: reutilizar el kernel geométrico
de `_core/surface_mesh.py` y conservarlo como referencia. `cli.py` sólo despacha.
Vértices, caras, componentes y conservación se contrastan contra la extracción
original; el solver W queda intacto.

## Objetivo acordado

Dejar `winding-field-cli` produciendo mallas, imágenes CT aplanadas grandes y,
cuando sea viable, TIFXYZ de la misma superficie. El usuario fija **30 minutos
como máximo objetivo por W completa, idealmente muchísimo menos**, desde un
campo ya calculado hasta disponer de la malla y el render global.

Ese tiempo es un criterio de aceptación por comprobar, no una promesa ni una
extrapolación del recorte. El cálculo del campo queda fuera de ese reloj. Reusar
una malla o UV ya calculados tendrá un tiempo separado y no contará como prueba
del recorrido completo desde el campo.

Se mantiene geometría nativa. Los bloques son unidades internas de trabajo;
la entrega buscada son lienzos coherentes por superficie conectada. Todos los
componentes y caras fuente se contabilizan. La cobertura parcial se declara,
como en el trabajo anterior; no se inventan uniones ni se escoge silenciosamente
el componente mayor.

El usuario pide **cobertura objetivo configurable en cada run**. No hay una
proporcionalidad garantizada entre cobertura y tiempo: la malla completa tiene
coste fijo, los charts tienen distinta dificultad y la unión puede invalidar
parte de un mapeo provisional. El límite temporal no promete alcanzar cualquier
porcentaje solicitado.

Se proponen `coverage_target` en `(0,1]` y `surface_time_limit_s` positivo,
congelados en la configuración. Aplican a cada W solicitada, no a la suma de un
lote de niveles. Defaults propuestos: intentar100% con hasta1800s; terminar antes
si se logra el objetivo o se agotan las operaciones útiles previstas. Se permiten
objetivos menores, sin simplificar ni recortar la malla fuente.

Cobertura significa aquí **área fuente con parametrización UV global válida /
área total de la malla del nivel**. Se informan aparte soporte CT y conflictos
de píxel; no es una certificación del área física visible ni de la reconstrucción
TIFXYZ. El objetivo se evalúa después del ensamblado/validación final, no sumando
éxitos provisionales de charts. `target_met`, `deadline_met` y cobertura completa
son hechos distintos: alcanzar90% solicitado no permite declarar100%.

El planificador debe reservar tiempo para ensamblado, render y publicación; no
gastar todo el presupuesto en recuperar UV. Si no alcanza el objetivo, entrega
sólo productos válidos que haya podido finalizar, con estado parcial y motivo.
Nunca publica archivos incompletos ni afirma haber cumplido el objetivo cuando
sólo ha agotado el tiempo. Los nombres finales de opciones CLI quedan para el plan.

## Qué causó el sobrecoste anterior

La rama directa construye una guía gruesa, proyecta cada nodo sobre W mediante
raíces certificadas, comprueba geometría/intersecciones y recorre la ROI para
certificar cobertura espacial. Su verificación independiente vuelve a proyectar
y auditar. También hay costes de índices, consultas y reintentos; no se atribuye
todo el tiempo exclusivamente a cobertura.

Esas garantías son distintas del objetivo de visualizar la malla fina. Escribir
TIFXYZ no exige recalcular sus posiciones desde W: pueden proceder de la misma
interpolación de triángulos que usa el render CT. Esa salida se identificará como
derivada de malla, sin atribuirle certificados del exportador directo.

## Base existente comprobada

- `src/winding_field/meshes.py` y `_core/surface_mesh.py`: extracción indexada,
  vértices compartidos, componentes y OBJ por nivel.
- `src/winding_field/flatten.py`: LSCM, conservación de IDs fuente y recuperación
  de trabajo. Sus lotes de 1024 caras y previews por fragmento no constituyen un
  ensamblado UV global.
- `src/winding_field/preview.py` y `io.py`: coordenadas físicas, interpolación
  baricéntrica, muestreo trilineal y acceso/cache CT. El preview de unos 256 px y
  contraste independiente por fragmento se sustituye, en la ruta nueva, por un
  render global a pitch físico.
- `src/winding_field/tifxyz_store.py`: escritura TIFF por tiles, escala,
  comprobación de bytes y publicación de canales. Se reutiliza la escritura
  sin acoplar el nuevo productor a los certificados de raíces/cobertura.
- Referencia Paris4 en
  `../vesubius-challenge/experiments/winding-field-exp/six-surfaces-opt-20261016/`:
  `global_sheet.py` registra bordes compartidos y suelda UV; `global_sheet_render.py`
  rasteriza XYZ y muestrea CT por lotes. Sus constantes de checkpoint, frame,
  unidades, nivel y cortes específicos no se trasladan como supuestos generales.

## Recorrido propuesto

```text
campo fijado
  -> malla nativa indexada + OBJ
  -> aplanados locales ligados a IDs fuente
  -> ensamblado UV global por superficie conectada
  -> un mapa XYZ por tiles
       -> CT -> PNG global + overview + máscara
       -> TIFXYZ opcional
```

### Malla y reutilización

La entrada normal sigue siendo el campo de una ejecución del CLI. La malla se
extrae una vez y se consume desde su representación indexada, sin escribir y
volver a parsear OBJ para cada etapa. El OBJ sigue siendo un producto portable.

Para el caso real de 175A se admite también reutilizar sus mallas históricas:
preferir los fragmentos nativos y recibos ligados al OBJ cuando estén disponibles.
El adaptador debe conservar IDs, caras, niveles, hashes y la transformación
explícita L0 -> micrómetros. No fabricar un `RUN.json` o `FIELD.json` para hacer
pasar esa entrada por una ejecución del CLI. No se solicita un importador general
de materiales/texturas ni de formatos de terceros.

### Aplanado y ensamblado

Procesar grupos conectados de caras con tamaño acotado. Reutilizar el aplanado
existente y medir su coste antes de escoger tamaños o cambiar de solver. Aumentar
`chart_faces` sin más no es una optimización demostrada: la validación de solapes
actual tiene trabajo cuadrático y puede convertirse en otro cuello de botella.

Construir la vecindad con IDs/aristas fuente compartidos; registrar los UV locales
y ajustar sus costuras. Mantener todas las componentes por separado y los cortes
necesarios para abrir superficies. Una discontinuidad, fallo de parametrización
o pérdida de cobertura queda en el informe; no dispara refinamientos sin límite.

Conservar la correspondencia de cada cara al componente, chart, UV y estado.
Las caras sin ubicación válida continúan en el denominador de cobertura. No pegar
PNGs ya normalizados ni cargar todos los vértices/triángulos de una W en RAM para
aplicar un único SLIM global. La implementación debe mantener acotados también
el grafo de costuras y los datos residentes del ensamblador.

### Render y TIFXYZ desde el mismo XYZ

Rasterizar el lienzo por tiles. Para cada muestra UV válida, interpolar XYZ en
su triángulo nativo; después muestrear CT agrupando puntos por chunks. Persistir
el trabajo geométrico y aprovechar la caché CT existente en regeneraciones.

El mapa geométrico tiene una máscara propia. El PNG añade la condición de soporte
CT; falta de CT no invalida por sí sola una posición geométrica válida para
TIFXYZ. Los conflictos XYZ quedan explícitos y sin relleno. Usar una ventana de
intensidades común al lienzo y documentarla; no normalización por chart.

Si se pide TIFXYZ, escribir `x.tif`, `y.tif`, `z.tif` y metadatos desde ese mapa,
con las unidades/frame y escala del lector. Comprobar las celdas locales que
reconstruye el lector para evitar puentes sobre huecos o saltos geométricos. Si
ese producto no es válido, conservar OBJ/PNG válidos e informar que TIFXYZ no se
ha publicado, con el motivo. No sustituirlo automáticamente por la ruta directa
de raíces. La escritura de TIFXYZ y su tiempo adicional quedan identificados.

## Integración y compatibilidad propuestas

**Decisiones técnicas propuestas por el agente, pendientes de aprobación:**

- Una ruta `surface_output=native` para ejecuciones nuevas, con etapas
  `prepare -> fibres -> field -> mesh -> flatten -> assemble -> render`.
  Proponerla como default para nuevas ejecuciones; configuraciones antiguas sin
  selector mantienen `atlas`. `atlas` y `tifxyz` explícitos mantienen su contrato.
- `--export-tifxyz` solicita además la salida derivada de malla. La dependencia
  `tifffile` ya está contemplada; si falta, comunicarlo antes de trabajo caro.
  No requerir libigl o CUDA sólo para escribir los canales TIFF.
- Pitch inicial del render igual al menor spacing físico del campo fino; para
  el caso histórico175A, 17,28 µm/píxel. Es una elección de muestreo de salida,
  no una simplificación de la malla. Registrar siempre el pitch efectivo.
- Salidas globales en `surfaces/level-*/component-*/`, con imagen, overview,
  máscara, procedencia y TIFXYZ opcional. Los bloques de aplanado son intermedios
  reanudables, no miles de PNG/OBJ de entrega en la ruta nueva.
- Reutilizar NumPy/SciPy/Pillow y los lectores/escritores disponibles. Cualquier
  dependencia adicional requiere justificarla frente al trabajo reutilizable.

Los nombres y esquema exacto de los recibos se resuelven en el plan siguiendo
las convenciones existentes. Deben distinguir finalización de archivos,
cobertura parcial, derivados de malla y certificación del campo. Los recibos del
exportador directo y `verify-tifxyz` no se reutilizan para fingir su certificación.

## Controles necesarios y alcance de calidad

Mantener validación de entrada/frame/unidades, identidad de fuente, conservación
de caras, UV finitos y no invertidos, tratamiento de solapes, lectura CT válida,
compatibilidad de formato, reanudación e integridad de archivos publicados.
Conservar `physical_acceptance=false`.

Informar área fuente total, área con UV válido, caras/componentes sin mapear,
distorsión, píxeles CT válidos y ambiguos. Área mapeada de la malla no significa
cobertura certificada del campo ni de la superficie reconstruida por TIFXYZ.

No imponer nuevas pruebas exhaustivas de cobertura volumétrica, unicidad de
raíces o intersecciones 3D globales como condición del render. Esto no elimina las
comprobaciones locales del producto ni permite ocultar mapeos ambiguos.

## Recursos y prueba de rendimiento

### Caso de aceptación fijado

- W=-4 del campo histórico175A, dominio completo de nodos ZYX
  `[0,0,0]..[1594,4682,4682]` half-open, origenXYZ0 y spacing17,28µm.
- Configuración fuente en
  `/mnt/bulk/vesuvius-175A-20261024/native-meshes-20261026/job-01/CONFIG.json`,
  SHA256 `4fdac93398fa748a51d4b43d0e27c89b041ef41ee880dff44df661b2d6594d81`.
  Su inventario `pinned` fija los payloads y lectores históricos; comprobarlo,
  sin reconstruir un campo distinto a partir de nombres o parámetros similares.
- Campo corregido `1.5843640137880175 * native + trilinear(coarse_delta)`.
  Reutilizar la adaptación probada en `tools/bench_direct_tifxyz.py:open_historical`
  con la ROI completa. Ese helper es referencia de la adaptación, no una licencia
  para importar toda la rama de proyección/auditoría al nuevo productor.
- Frame `PHerc0175A/20250521115057`; CT L0,8,64µm, shapeZYX
  `[12748,9363,9363]`, en
  `https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr`.
  Fijar también la identidad de metadatos CT al abrirlo.
- El recibo local `docs/verification/direct-field-tifxyz-pilot-20261029.json`
  documenta esta geometría. Sus cuatro ROIs pequeñas sirven para pruebas
  acotadas; ninguna sustituye el dominio completo en la medición final.

**Decisión de medición propuesta por el agente:** el caso principal utiliza una
caché de payloads CT nueva/vacía y contabiliza las descargas necesarias dentro
del reloj. No se purga la caché del SO ni se borran caches ajenas. Un ensayo con
CT ya disponible se registra como caso caliente separado; una red lenta puede
hacer fallar el objetivo end-to-end y se verá en sus tiempos de I/O.

### Límites y secuencia de medición

Piloto en `gpu-1`, con sus 3 CPU lógicas y los límites de acceso existentes.
El objetivo inicial es <=3 GiB de RSS agregada, scratch <=16 GiB y reserva de
disco12,25 GiB. Estimar también productos persistentes y caché: no esconder
consumo reclasificando temporales. Si el caso no cabe, registrar el bloqueo y
replantear el acceso/formato antes de elevar límites. No tocar `_core/` ni sus
recibos congelados en esta integración; un cuello de botella allí requiere una
propuesta explícita antes de cambiar ese compromiso.

CUDA no es requisito de la arquitectura. Si un benchmark justifica usarla,
únicamente GPU0 ya autorizada. No detener trabajos activos ni medir una ejecución
en contención como si fuera una prueba dedicada. Esperar disponibilidad para la
medición final; preservar procesos, runtimes y datos anteriores.

1. Medir una selección acotada y representativa de bloques del caso W-4:
   extracción/carga, aplanado, ensamblado, rasterización, CT y escritura.
   Obtener geometría e imagen inspeccionables, no sólo tiempos de microkernels.
2. Usar los resultados para resolver el cuello de botella real. Reutilizar los
   artefactos de prueba; evitar conversiones completas repetidas sin un cambio
   justificado. El test pequeño no demuestra el objetivo de W completa.
3. Ejecutar un ensayo W-4 sobre todo el dominio del campo histórico declarado,
   con límite de tiempo cooperativo de30min. El dominio no es todo el rollo CT.
   Fijar y registrar antes de ejecutarlo el objetivo de cobertura elegido para
   ese run. Para cumplirlo se exige malla completa, cobertura final al menos igual
   a la solicitada y productos finalizados en tiempo. Una ROI reducida, una malla
   truncada o detenerse por presupuesto por debajo del objetivo no cuentan como
   éxito. Una cobertura solicitada menor de100% se sigue declarando parcial.
4. Separar tiempos desde campo, desde malla y desde UV persistidos. Registrar
   estado inicial de la caché CT y descarga/I/O dentro del reloj, con las
   condiciones fría/caliente fijadas arriba. Comparar tiempos entre runs sólo
   junto a sus objetivos y cobertura alcanzada; no inferir proporcionalidad.

El reloj principal incluye extracción, aplanado, ensamblado, muestreo CT,
validaciones necesarias y publicación de OBJ/PNG/recibos. Medir aparte el extra
de TIFXYZ, además del total con él habilitado. Se busca terminar mucho antes de
30min; si no se consigue, informar la fase limitante, sin declarar resuelto el
objetivo ni usar ratios del recorte como resultado medido.

Los runs con varios niveles necesitan progreso, presupuesto y publicación por
nivel: no esperar a completar el último W para entregar el primero. Compartir
fuente y caché donde sea útil, contabilizando el trabajo común sin ocultarlo.

## Verificación y decisiones del implementador

Tests de costuras entre bloques, todas las caras/componentes contabilizadas,
transformaciones físicas, conflictos UV/XYZ, muestreo CT conocido, celdas TIFXYZ
sobre huecos, reanudación y fuente inmutable. Roundtrip con el lector real para
la salida TIFXYZ que se publique. Probar CLI/configuración nueva y compatibilidad
con las dos ramas actuales. Benchmark real de extremo a extremo antes de afirmar
rapidez; los tests funcionales no sustituyen esa medición.

El implementador decide la división de módulos, tamaños de lotes, índice espacial,
almacenamiento acotado y el detalle del rasterizador después del perfilado. Puede
reutilizar componentes de Paris4 sin importar sus constantes o añadir un segundo
framework. Cambios de tolerancias, precisión geométrica, recursos o garantías
deben quedar explícitos; no son optimizaciones silenciosas.

El usuario aprobó implementar y probar este diseño, con cancelación y reanudación.
Ejecución nativa en el checkout actual; sin commits ni cambios en runtimes congelados.
