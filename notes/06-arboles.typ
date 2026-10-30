// OUTLINE (3 horas: 150 min de teoría + 30 min de laboratorio)
// 1. Árboles de decisión (25 min)
// 1.1. Un modelo no lineal: el ejemplo del esquí
// 1.2. Particionamiento recursivo
// 1.3. Qué predice cada hoja
// 2. ¿Qué pregunta hacer? (30 min)
// 2.1. Ganancia de una división
// 2.2. Primer intento: el error de clasificación
// 2.3. La impureza de Gini
// 2.4. Por qué Gini funciona: concavidad
// 2.5. Entropía cruzada
// 2.6. Cuántas preguntas hay que probar
// 3. Regresión, variables categóricas y regularización (25 min)
// 3.1. Árboles de regresión (y la relación entre Gini y la varianza)
// 3.2. Variables categóricas
// 3.3. Regularización y poda
// 3.4. Evaluar un clasificador: exactitud y matriz de confusión
// 3.5. Ventajas y desventajas
// 4. Ensamblaje: bagging y random forests (30 min)
// 4.1. Por qué promediar modelos
// 4.2. Bootstrap
// 4.3. Bagging
// 4.4. Random forests
// 5. Boosting (40 min)
// 5.1. Idea: combinar modelos débiles
// 5.2. AdaBoost
// 5.3. Boosting con pérdida cuadrática: ajustar los residuos
// 5.4. Gradient boosting
// 5.5. XGBoost
// 6. Laboratorio (30 min) -- PENDIENTE
//
// Los datos de las figuras se generan con `python generar_figuras.py`
// y se leen desde la carpeta figs/.

#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *

#let fvec(x) = math.bold(math.upright(x))
#let argmax = $op("argmax", limits: #true)$

#set text(lang: "es")
#set smartquote(quotes: "“”")

#let frame(stroke) = (x, y) => (
  left: 0pt,
  right: 0pt,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

#set table(
  stroke: frame(1pt + rgb("21222C")),
)

#show: minimal-note.with(
  title: [Árboles de decisión y métodos de ensamblaje],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

// Datos de las figuras
#let d-esqui = json("figs/06/esqui.json")
#let d-reg = json("figs/06/regresion.json")
#let d-esqui-reg = json("figs/06/esqui_regresion.json")
#let cf = json("figs/06/confusion.json")
#let d-xor = json("figs/06/xor.json")
#let d-diag = json("figs/06/diagonal.json")
#let d-boot = json("figs/06/bootstrap.json")
#let d-bag = json("figs/06/bagging.json")
#let d-err = json("figs/06/error_vs_B.json")
#let d-ada = json("figs/06/adaboost.json")
#let d-gb = json("figs/06/boosting.json")

#let num(x, digits: 2) = str(calc.round(x, digits: digits))
#let pct(x) = str(calc.round(x * 100, digits: 1)) + "%"

// Estilo de las clases (el mismo en todas las figuras)
#let c-neg = rgb("#1f77b4")
#let c-pos = rgb("#d62728")
#let c-3 = green.darken(25%)
#let c-4 = orange.darken(10%)

#let puntos(datos, clase, tam: 0.13) = plot.add(
  datos,
  mark: if clase < 1 { "o" } else { "triangle" },
  mark-size: tam,
  mark-style: (stroke: none, fill: if clase < 1 { c-neg } else { c-pos }),
  style: (stroke: none),
)

#let puntos-clases(d, tam: 0.13) = {
  puntos(d.neg, -1, tam: tam)
  puntos(d.pos, 1, tam: tam)
}

// Círculo vacío alrededor de un punto (para marcar errores)
#let marcar(pts, tam: 0.3) = plot.add(
  pts,
  mark: "o",
  mark-size: tam,
  mark-style: (stroke: 0.9pt + black, fill: none),
  style: (stroke: none),
)

// Puntos negros (datos de regresión)
#let puntos-reg(pts, tam: 0.08) = plot.add(
  pts,
  mark: "o",
  mark-size: tam,
  mark-style: (stroke: none, fill: gray.darken(30%)),
  style: (stroke: none),
)

// Rectángulo sombreado con el color de la clase predicha
#let region(x0, x1, y0, y1, clase, alfa: 82%) = plot.add(
  ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)),
  fill: true,
  fill-type: "shape",
  style: (
    stroke: (paint: gray.darken(20%), thickness: 0.5pt),
    fill: (if clase == 1 { c-pos } else { c-neg }).transparentize(alfa),
  ),
)

#let regiones(rects, alfa: 82%) = for r in rects {
  let (x0, x1, y0, y1, c) = r
  region(x0, x1, y0, y1, c, alfa: alfa)
}

#let segmento(a, b, estilo: (paint: black, thickness: 1.4pt)) = plot.add((a, b), style: (stroke: estilo))

#let curva(pts, color: black, grosor: 1.4pt, punteada: false) = plot.add(
  pts,
  style: (stroke: (paint: color, thickness: grosor, dash: if punteada { "dashed" } else { none })),
)

// Matriz de confusión: M = ((VN, FP), (FN, VP)), filas = clase real
#let matriz-confusion(M) = {
  let celda(n, etiqueta, bien) = table.cell(
    fill: if bien { c-neg.transparentize(80%) } else { c-pos.transparentize(85%) },
    align: center,
    [#text(size: 13pt)[*#n*] \ #text(size: 8pt, fill: gray.darken(30%))[#etiqueta]],
  )
  set text(size: 9pt)
  table(
    columns: (auto, 2.1cm, 2.1cm),
    rows: (auto, 1.2cm, 1.2cm),
    stroke: (x, y) => if x > 0 and y > 0 { 0.6pt + gray } else { none },
    align: horizon,
    [], [predice "no"], [predice "sí"],
    [real "no"], celda(M.at(0).at(0), "VN", true), celda(M.at(0).at(1), "FP", false),
    [real "sí"], celda(M.at(1).at(0), "FN", false), celda(M.at(1).at(1), "VP", true),
  )
}
#let exactitud-cf(M) = (M.at(0).at(0) + M.at(1).at(1)) / (M.at(0).sum() + M.at(1).sum())
#let pct-cf(M) = pct(exactitud-cf(M))
#let pct-cf-precision(M) = pct(M.at(1).at(1) / (M.at(1).at(1) + M.at(0).at(1)))
#let pct-cf-sensibilidad(M) = pct(M.at(1).at(1) / (M.at(1).at(1) + M.at(1).at(0)))

#let leyenda-clases = {
  set text(size: 9pt)
  grid(
    columns: 2,
    column-gutter: 16pt,
    box[#text(fill: c-neg)[●] negativo],
    box[#text(fill: c-pos)[▲] positivo],
  )
}

#let linea-leyenda(color, etiqueta, punteada: false) = box[
  #box(width: 0.6cm, height: 0pt, baseline: -0.3em, stroke: (
    top: (paint: color, thickness: 1.6pt, dash: if punteada { "dashed" } else { none }),
  ))
  #etiqueta
]

#let leyenda-lineas(..items) = {
  set text(size: 9pt)
  grid(columns: items.pos().len(), column-gutter: 14pt, ..items.pos())
}

// Diagramas de árboles
// nodos: diccionario nombre -> (posición, contenido)
// aristas: lista de (padre, hijo, etiqueta)
#let caja(cuerpo, color: black) = box(
  fill: white,
  stroke: 0.6pt + color,
  inset: (x: 4pt, y: 3pt),
  radius: 2pt,
  text(size: 8.5pt, fill: color, cuerpo),
)
#let hoja(clase) = if clase == 1 {
  caja(color: c-pos)[*+*]
} else if clase == 0 {
  caja(color: c-neg)[*−*]
} else {
  caja(color: gray)[?]
}

#let diagrama(nodos, aristas) = cetz.canvas({
  import cetz.draw: *
  for (a, b, etq) in aristas {
    let pa = nodos.at(a).at(0)
    let pb = nodos.at(b).at(0)
    line(pa, pb, stroke: 0.6pt + gray.darken(30%))
    let medio = ((pa.at(0) + pb.at(0)) / 2, (pa.at(1) + pb.at(1)) / 2)
    let lado = if pb.at(0) < pa.at(0) { "east" } else { "west" }
    content(
      (medio.at(0) + if lado == "east" { -0.08 } else { 0.08 }, medio.at(1) + 0.08),
      anchor: "south-" + lado,
      text(size: 7.5pt, fill: gray.darken(40%), etq),
    )
  }
  for (nombre, (pos, cuerpo)) in nodos {
    content(pos, cuerpo)
  }
})

#let panel-esqui(cuerpo, size: (6, 4.6)) = custom-plot(
  x-min: 1,
  x-max: 13,
  y-min: -90,
  y-max: 90,
  x-label: [mes],
  y-label: [latitud],
  x-tick-step: 2,
  y-tick-step: 30,
  size: size,
  cuerpo,
)

#let panel-2d(cuerpo, x-min: 0, x-max: 5, y-min: 0, y-max: 5, size: (5, 5), x-label: $x_1$, y-label: $x_2$) = custom-plot(
  x-min: x-min,
  x-max: x-max,
  y-min: y-min,
  y-max: y-max,
  x-label: x-label,
  y-label: y-label,
  x-tick-step: 1,
  y-tick-step: 1,
  size: size,
  cuerpo,
)

// Impurezas para clasificación binaria, en función de la proporción p de positivos
#let l-error(p) = calc.min(p, 1 - p)
#let l-gini(p) = 2 * p * (1 - p)
#let l-entropia(p) = if p <= 0 or p >= 1 { 0 } else { -p * calc.log(p, base: 2) - (1 - p) * calc.log(1 - p, base: 2) }

// Impureza promedio (ponderada) de los hijos de una división.
// Cada hijo es un par (positivos, negativos).
#let impureza-hijos(L, hijos) = {
  let total = hijos.map(h => h.at(0) + h.at(1)).sum()
  hijos.map(h => (h.at(0) + h.at(1)) / total * L(h.at(0) / (h.at(0) + h.at(1)))).sum()
}

= Parte 1: Árboles de decisión

== Un modelo no lineal: el ejemplo del esquí

Supongamos que queremos predecir si se puede esquiar en un lugar y un momento dados, a partir de dos features: el mes del año (de 1 a 12) y la latitud del lugar (de $-90 degree$ en el Polo Sur a $90 degree$ en el Polo Norte, con $0 degree$ en el Ecuador). En el hemisferio norte se puede esquiar al principio y al final del año. En el hemisferio sur, a mitad de año. Cerca del Ecuador, nunca.

#figure(
  caption: [#d-esqui.m ejemplos del problema del esquí. Los positivos (se puede esquiar) aparecen en tres zonas: arriba a la izquierda y arriba a la derecha (invierno del hemisferio norte), y abajo al centro (invierno del hemisferio sur). Ninguna recta separa las dos clases.],
  [
    #panel-esqui({
      puntos-clases(d-esqui, tam: 0.11)
    })
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-esqui>

La @fig-esqui muestra los datos. Un clasificador lineal no puede separarlos: los positivos están en tres zonas separadas. Podríamos inventar features nuevas, como en las clases anteriores, pero hay una forma mucho más directa de describir esta frontera: con preguntas. ¿Estamos al norte de los $30 degree$? Si es así, ¿estamos antes de marzo? Es el mismo procedimiento del juego de las 20 preguntas.

== Particionamiento recursivo

Un árbol de decisión divide el espacio de las features en regiones haciendo una pregunta a la vez. Cada pregunta mira una sola feature y la compara con un umbral.

Formalmente, empezamos con una región $R_p$ (la región "padre"), que al principio es todo el espacio. Escogemos una feature $j$ y un umbral $t$, y dividimos $R_p$ en dos regiones hijas:

$
  R_1 = {fvec(x) in R_p : x_j < t}, quad quad R_2 = {fvec(x) in R_p : x_j >= t}.
$

Luego repetimos el procedimiento por separado en $R_1$ y en $R_2$, y así sucesivamente. Por eso se dice que el algoritmo es recursivo y de arriba hacia abajo (top-down): empieza con todo el espacio y lo va partiendo en pedazos cada vez más pequeños. Además es codicioso (greedy): en cada paso escoge la mejor pregunta en ese momento, sin pensar en las preguntas que vendrán después.

#figure(
  caption: [Las dos primeras preguntas. La primera, latitud $> 30 degree$, divide el plano con una recta horizontal. La segunda, mes $< 3$, divide solo la región de arriba, con un segmento vertical. Las regiones que quedan corresponden a las hojas del árbol de la derecha. La región de arriba a la izquierda contiene #d-esqui.arriba_izq.at(0) positivos y #d-esqui.arriba_izq.at(1) negativos: ya es pura. En las otras dos regiones todavía hay ejemplos de las dos clases, marcadas con "?".],
  [
    #grid(
      columns: (1.2fr, 1fr),
      gutter: 0.6cm,
      align: horizon,
      panel-esqui(size: (5.5, 4.4), {
        segmento((1, 30), (13, 30))
        segmento((3, 30), (3, 90))
        puntos-clases(d-esqui, tam: 0.1)
        plot.annotate({
          import cetz.draw: *
          content((2, 82), text(size: 8pt)[$R_1$])
          content((12.2, 82), text(size: 8pt)[$R_2$])
          content((12.2, -80), text(size: 8pt)[$R_3$])
        })
      }),
      diagrama(
        (
          a: ((0, 0), caja[latitud $> 30 degree$]),
          b: ((-1.1, -1.2), caja[mes $< 3$]),
          c: ((1.1, -1.2), caja[$R_3$: ?]),
          d: ((-1.9, -2.4), caja(color: c-pos)[$R_1$: *+*]),
          e: ((-0.3, -2.4), caja[$R_2$: ?]),
        ),
        (("a", "b", [sí]), ("a", "c", [no]), ("b", "d", [sí]), ("b", "e", [no])),
      ),
    )
  ],
) <fig-dos-preguntas>

Si seguimos haciendo preguntas, eventualmente todas las regiones quedan puras. La @fig-arbol-completo muestra un árbol que clasifica correctamente todos los ejemplos.

#figure(
  caption: [Un árbol que clasifica correctamente todos los ejemplos. Tiene 7 hojas, y cada hoja corresponde a una de las 7 regiones rectangulares de la izquierda. Seguimos la rama "sí" si la respuesta a la pregunta es afirmativa.],
  [
    #grid(
      columns: (1fr, 1.15fr),
      gutter: 0.4cm,
      align: horizon,
      panel-esqui(size: (5, 4.2), {
        region(1, 3, 30, 90, 1)
        region(3, 12, 30, 90, 0)
        region(12, 13, 30, 90, 1)
        region(1, 13, -30, 30, 0)
        region(1, 6, -90, -30, 0)
        region(6, 9, -90, -30, 1)
        region(9, 13, -90, -30, 0)
        puntos-clases(d-esqui, tam: 0.09)
      }),
      diagrama(
        (
          a: ((0, 0), caja[latitud $> 30 degree$]),
          b: ((-1.9, -1.1), caja[mes $< 3$]),
          c: ((1.9, -1.1), caja[latitud $< -30 degree$]),
          b1: ((-2.7, -2.2), hoja(1)),
          b2: ((-1.1, -2.2), caja[mes $>= 12$]),
          b21: ((-1.7, -3.3), hoja(1)),
          b22: ((-0.5, -3.3), hoja(0)),
          c1: ((1.1, -2.2), caja[mes $< 6$]),
          c2: ((2.9, -2.2), hoja(0)),
          c11: ((0.5, -3.3), hoja(0)),
          c12: ((1.7, -3.3), caja[mes $< 9$]),
          c121: ((1.1, -4.4), hoja(1)),
          c122: ((2.3, -4.4), hoja(0)),
        ),
        (
          ("a", "b", [sí]),
          ("a", "c", [no]),
          ("b", "b1", [sí]),
          ("b", "b2", [no]),
          ("b2", "b21", [sí]),
          ("b2", "b22", [no]),
          ("c", "c1", [sí]),
          ("c", "c2", [no]),
          ("c1", "c11", [sí]),
          ("c1", "c12", [no]),
          ("c12", "c121", [sí]),
          ("c12", "c122", [no]),
        ),
      ),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-arbol-completo>

Algunas observaciones:

- Cada pregunta compara una sola feature con un umbral, así que las fronteras son siempre paralelas a los ejes y las regiones son rectángulos (o, con $d$ features, cajas en $RR^d$).
- Las regiones no se solapan y cubren todo el espacio: cada punto cae en exactamente una región.
- Cada región corresponde a una hoja del árbol (un nodo sin hijos). El número de regiones es el número de hojas.
- Una misma feature se puede usar varias veces, con distintos umbrales, en distintas ramas.

== Qué predice cada hoja

Supongamos que el árbol tiene $J$ hojas, con regiones $R_1, ..., R_J$. El árbol asigna a cada región una predicción $w_j$, y para un punto $fvec(x)$ predice el valor de la región donde cae:

$
  f(fvec(x)) = sum_(j=1)^J w_j bb(1){fvec(x) in R_j}.
$

Todos los términos de la suma son cero excepto el de la región que contiene a $fvec(x)$. En clasificación, $w_j$ es la clase más frecuente entre los ejemplos de entrenamiento que caen en $R_j$ (un voto de mayoría). En regresión, como veremos en la Parte 3, $w_j$ es el promedio de sus etiquetas.

Falta la pregunta más importante: ¿cómo escoger la pregunta (la feature $j$ y el umbral $t$) en cada paso?

= Parte 2: ¿Qué pregunta hacer?

== Ganancia de una división

Para comparar preguntas necesitamos medir qué tan "mezclada" está una región. Llamamos $L(R)$ a esa medida, la impureza de la región, y la definiremos en un momento. Una buena pregunta es la que deja hijos mucho menos mezclados que el padre.

Para una región $R$, sea $hat(p)_c$ la proporción de ejemplos de entrenamiento de la clase $c$ en $R$. La impureza $L(R)$ será una función de estas proporciones. Como las regiones hijas pueden tener tamaños muy distintos, promediamos sus impurezas ponderando por el número de ejemplos que contiene cada una. La ganancia de dividir $R_p$ en $R_1$ y $R_2$ es

$
  "ganancia" = L(R_p) - ((abs(R_1)) / (abs(R_p)) L(R_1) + (abs(R_2)) / (abs(R_p)) L(R_2)),
$

donde $abs(R)$ es el número de ejemplos de entrenamiento en $R$. En cada paso escogemos la feature $j$ y el umbral $t$ que maximizan la ganancia. Como $L(R_p)$ no depende de la pregunta, esto es lo mismo que minimizar la impureza promedio de los hijos.

== Primer intento: el error de clasificación

La medida más obvia es la fracción de ejemplos que clasificaríamos mal si predijéramos la clase mayoritaria de la región:

$
  L_"error" (R) = 1 - max_c hat(p)_c.
$

Parece razonable, pero tiene un problema. Consideremos una región padre con 900 positivos y 100 negativos, y dos posibles divisiones (@fig-dos-divisiones).

#let arbol-conteo(padre, h1, h2) = diagrama(
  (
    p: ((0, 0), caja[#padre.at(0)+ #h(2pt) #padre.at(1)−]),
    a: ((-1.1, -1.2), caja[#h1.at(0)+ #h(2pt) #h1.at(1)−]),
    b: ((1.1, -1.2), caja[#h2.at(0)+ #h(2pt) #h2.at(1)−]),
  ),
  (("p", "a", $R_1$), ("p", "b", $R_2$)),
)

#figure(
  caption: [Dos formas de dividir una región con 900 positivos y 100 negativos. En ambas, uno de los hijos es puro (solo tiene positivos), pero la división de la derecha aísla 500 positivos en lugar de 200.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      [División A #v(0.2cm) #arbol-conteo((900, 100), (700, 100), (200, 0))],
      [División B #v(0.2cm) #arbol-conteo((900, 100), (400, 100), (500, 0))],
    )
  ],
) <fig-dos-divisiones>

Intuitivamente, la división B es mejor: deja 500 positivos en una región pura, contra 200 de la división A. Pero el error de clasificación no ve ninguna diferencia. En el padre predecimos positivo y nos equivocamos en 100 de 1000 ejemplos. En la división A, los dos hijos predicen positivo y siguen equivocándose en los mismos 100 negativos: $800/1000 dot 100/800 + 200/1000 dot 0 = 0.1$. En la división B ocurre lo mismo: $500/1000 dot 100/500 + 0 = 0.1$. La ganancia es cero en los dos casos, como si ninguna de las dos preguntas sirviera para nada.

El problema es que el error de clasificación solo cambia si alguna región cambia de clase mayoritaria. Muchas preguntas buenas, que separan bien los datos sin cambiar la clase mayoritaria, quedan con ganancia cero. Necesitamos una medida más sensible.

== La impureza de Gini

Imaginemos el siguiente juego. Sacamos al azar un ejemplo de la región $R$, y en lugar de predecir siempre la clase mayoritaria, le asignamos una etiqueta al azar, sorteada con las mismas proporciones de la región: la clase $c$ con probabilidad $hat(p)_c$. ¿Con qué probabilidad nos equivocamos?

El ejemplo es de la clase $c$ con probabilidad $hat(p)_c$, y en ese caso nos equivocamos si la etiqueta sorteada es otra, lo que ocurre con probabilidad $1 - hat(p)_c$. Sumando sobre las clases:

$
  L_"Gini" (R) = sum_c hat(p)_c (1 - hat(p)_c) = 1 - sum_c hat(p)_c^2.
$

(La segunda forma sale de que $sum_c hat(p)_c = 1$.) A esto se le llama la impureza de Gini. Otra forma equivalente de verla: es la probabilidad de que dos ejemplos sacados al azar de la región (con reemplazo) sean de clases distintas.

- Si la región es pura, todos los ejemplos son de la misma clase y nunca nos equivocamos: $L_"Gini" = 0$.
- Si la región está muy mezclada, nos equivocamos mucho. Con dos clases, el máximo es $1/2$, cuando $hat(p) = 1/2$.

En clasificación binaria, si $hat(p)$ es la proporción de positivos, la fórmula queda

$
  L_"Gini" (R) = hat(p)(1 - hat(p)) + (1 - hat(p)) hat(p) = 2 hat(p) (1 - hat(p)).
$

Volvamos al ejemplo. El padre tiene $hat(p) = 0.9$, así que su impureza es $2 dot 0.9 dot 0.1 = #num(l-gini(0.9), digits: 3)$. En la división A, el hijo $(700+, 100-)$ tiene $hat(p) = 0.875$ e impureza $#num(l-gini(0.875), digits: 4)$, y el hijo puro tiene impureza 0. El promedio ponderado es $0.8 dot #num(l-gini(0.875), digits: 4) + 0.2 dot 0 = #num(impureza-hijos(l-gini, ((700, 100), (200, 0))), digits: 3)$. En la división B, el hijo $(400+, 100-)$ tiene $hat(p) = 0.8$ e impureza $#num(l-gini(0.8), digits: 2)$, y el promedio es $0.5 dot #num(l-gini(0.8), digits: 2) = #num(impureza-hijos(l-gini, ((400, 100), (500, 0))), digits: 2)$.

#let ganancia(L, hijos) = L(0.9) - impureza-hijos(L, hijos)
#let div-a = ((700, 100), (200, 0))
#let div-b = ((400, 100), (500, 0))

#figure(
  caption: [Ganancia de las dos divisiones de la @fig-dos-divisiones con tres medidas de impureza. El error de clasificación no distingue las dos divisiones. Gini y la entropía cruzada prefieren la división B.],
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header([Medida], [Padre], [Ganancia de A], [Ganancia de B]),
    [Error de clasificación], [#num(l-error(0.9), digits: 3)], [#num(ganancia(l-error, div-a), digits: 3)], [#num(ganancia(l-error, div-b), digits: 3)],
    [Gini], [#num(l-gini(0.9), digits: 3)], [#num(ganancia(l-gini, div-a), digits: 3)], [#num(ganancia(l-gini, div-b), digits: 3)],
    [Entropía cruzada (bits)], [#num(l-entropia(0.9), digits: 3)], [#num(ganancia(l-entropia, div-a), digits: 3)], [#num(ganancia(l-entropia, div-b), digits: 3)],
  ),
) <tab-ganancias>

Con Gini, la ganancia de la división B (#num(ganancia(l-gini, div-b), digits: 3)) es cuatro veces la de la división A (#num(ganancia(l-gini, div-a), digits: 3)). Gini sí premia la pregunta que aísla más positivos.

== Por qué Gini funciona: concavidad

Hay una explicación geométrica de por qué Gini ve la diferencia y el error de clasificación no. Pensemos en clasificación binaria y escribamos la impureza como función de la proporción de positivos, $L(hat(p))$.

Primero, la proporción de positivos del padre es el promedio ponderado de las proporciones de los hijos:

$
  hat(p)_p = (abs(R_1)) / (abs(R_p)) hat(p)_1 + (abs(R_2)) / (abs(R_p)) hat(p)_2,
$

porque los positivos del padre son los positivos de $R_1$ más los de $R_2$. En la división B, por ejemplo, $0.9 = 0.5 dot 0.8 + 0.5 dot 1$.

Ahora dibujemos la curva $L(hat(p))$ y marquemos los puntos de los dos hijos de una división, $(hat(p)_1, L(hat(p)_1))$ y $(hat(p)_2, L(hat(p)_2))$ (@fig-concavidad). La impureza promedio de los hijos es un punto sobre el segmento que los une, justo encima de $hat(p)_p$. La impureza del padre es el punto de la curva encima de $hat(p)_p$. La ganancia es la distancia vertical entre los dos.

#let panel-concavidad(L) = custom-plot(
  x-min: 0,
  x-max: 1,
  y-min: 0,
  y-max: 0.55,
  x-label: $hat(p)$,
  y-label: $L(hat(p))$,
  x-tick-step: 0.25,
  y-tick-step: 0.1,
  size: (5.5, 4.2),
  {
    import cetz.draw: *
    let p1 = 0.6
    let p2 = 1.0
    let pp = 0.8
    let medio = 0.5 * L(p1) + 0.5 * L(p2)
    plot.add(domain: (0, 1), samples: 200, L, style: (stroke: (paint: black, thickness: 1.4pt)))
    segmento((p1, L(p1)), (p2, L(p2)), estilo: (paint: c-neg, thickness: 1.2pt, dash: "dashed"))
    segmento((pp, 0), (pp, medio), estilo: (paint: gray, thickness: 0.6pt, dash: "dotted"))
    if L(pp) - medio > 0.001 {
      segmento((pp, medio), (pp, L(pp)), estilo: (paint: c-pos, thickness: 2pt))
    }
    plot.add(
      ((p1, L(p1)), (p2, L(p2))),
      mark: "o",
      mark-size: 0.13,
      mark-style: (stroke: none, fill: c-neg),
      style: (stroke: none),
    )
    plot.add(((pp, L(pp)),), mark: "o", mark-size: 0.13, mark-style: (stroke: none, fill: black), style: (stroke: none))
    plot.add(((pp, medio),), mark: "o", mark-size: 0.13, mark-style: (stroke: 0.9pt + c-neg, fill: white), style: (stroke: none))
    // Las etiquetas se colocan por su centro: con `anchor` cetz-plot las desplaza
    plot.annotate({
      content((p1 - 0.07, L(p1)), text(size: 8pt, fill: c-neg)[$R_1$])
      content((p2 - 0.06, L(p2) + 0.03), text(size: 8pt, fill: c-neg)[$R_2$])
      content((pp - 0.06, L(pp) + 0.035), text(size: 8pt)[$R_p$])
    })
  },
)

#figure(
  caption: [Una división en dos hijos del mismo tamaño, uno con 60% de positivos y otro puro ($hat(p)_1 = 0.6$, $hat(p)_2 = 1$), así que el padre tiene $hat(p)_p = 0.8$. Los puntos azules son los hijos, el círculo blanco es su impureza promedio y el punto negro es el padre. La ganancia es la distancia vertical entre los dos (en rojo). A la izquierda, con Gini, la curva está estrictamente por encima del segmento y la ganancia es positiva (#num(l-gini(0.8) - 0.5 * l-gini(0.6), digits: 2)). A la derecha, con el error de clasificación, la curva es una recta a cada lado de $1/2$: como los dos hijos quedan del mismo lado, el segmento coincide con la curva y la ganancia es cero.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 0.8cm,
      align: bottom,
      [Gini #v(0.1cm) #panel-concavidad(l-gini)],
      [Error de clasificación #v(0.1cm) #panel-concavidad(l-error)],
    )
  ],
) <fig-concavidad>

La curva de Gini es estrictamente cóncava (tiene forma de domo): cualquier segmento entre dos puntos distintos de la curva queda estrictamente por debajo de ella. Por lo tanto, cualquier división que cambie las proporciones, es decir, con $hat(p)_1 != hat(p)_2$, tiene ganancia positiva. Y cuanto más separados quedan $hat(p)_1$ y $hat(p)_2$, mayor es la ganancia.

La curva del error de clasificación, en cambio, es una pirámide formada por dos rectas. Si los dos hijos quedan del mismo lado de $1/2$ (los dos con mayoría positiva, por ejemplo), el segmento entre ellos queda exactamente sobre la recta y la ganancia es cero. Eso es lo que pasó en las dos divisiones del ejemplo.

== Entropía cruzada

La otra medida de impureza muy usada es la entropía cruzada (o simplemente entropía), que viene de la teoría de la información:

$
  L_"entropía" (R) = - sum_c hat(p)_c log_2 hat(p)_c.
$

Se interpreta como el número de bits que hacen falta, en promedio, para comunicar la clase de un ejemplo de la región si conocemos las proporciones $hat(p)_c$. Si la región es pura no hay nada que comunicar, y la entropía es 0. Si hay dos clases mezcladas por igual, hace falta 1 bit.

Como muestra la @fig-impurezas, la entropía y Gini tienen prácticamente la misma forma: las dos son estrictamente cóncavas, valen cero en las regiones puras y son máximas cuando las clases están balanceadas. En la práctica casi siempre escogen las mismas preguntas. Gini es un poco más barato de calcular (no tiene logaritmos) y es la opción por defecto en scikit-learn. Usaremos Gini en el resto de las notas.

#figure(
  caption: [Las tres medidas de impureza para clasificación binaria, en función de la proporción de positivos. La entropía está dividida entre 2 para que las tres tengan el mismo máximo. Gini y la entropía son curvas suaves y estrictamente cóncavas. El error de clasificación es una pirámide.],
  [
    #custom-plot(
      x-min: 0,
      x-max: 1,
      y-min: 0,
      y-max: 0.55,
      x-label: $hat(p)$,
      y-label: $L(hat(p))$,
      x-tick-step: 0.25,
      y-tick-step: 0.1,
      size: (7, 4.2),
      {
        plot.add(domain: (0, 1), samples: 200, l-error, style: (stroke: (paint: gray.darken(20%), thickness: 1.4pt, dash: "dashed")))
        plot.add(domain: (0, 1), samples: 200, l-gini, style: (stroke: (paint: c-pos, thickness: 1.6pt)))
        plot.add(domain: (0.0001, 0.9999), samples: 200, p => l-entropia(p) / 2, style: (stroke: (paint: c-neg, thickness: 1.6pt)))
      },
    )
    #v(0.2cm)
    #leyenda-lineas(
      linea-leyenda(gray.darken(20%), [error de clasificación], punteada: true),
      linea-leyenda(c-pos, [Gini]),
      linea-leyenda(c-neg, [entropía / 2]),
    )
  ],
) <fig-impurezas>

== Cuántas preguntas hay que probar

En cada nodo buscamos

$
  (j, t) = argmax_(j, t) "ganancia"(j, t).
$

Parece que hay infinitos umbrales $t$ posibles, pero no es así. Si ordenamos los ejemplos de la región según la feature $j$, cualquier umbral entre dos valores consecutivos divide los ejemplos de la misma forma. Con $m$ ejemplos hay solo $m - 1$ divisiones distintas por feature, y se suele tomar como umbral el punto medio entre los dos valores consecutivos. En total hay que evaluar a lo sumo $d (m - 1)$ preguntas, y el algoritmo simplemente las prueba todas.

Encontrar el árbol óptimo (el de menor error con un número dado de hojas) es un problema computacionalmente intratable (NP-completo). Por eso usamos el algoritmo codicioso: no garantiza el mejor árbol, pero es rápido y funciona bien.

= Parte 3: Regresión, variables categóricas y regularización

== Árboles de regresión

Los árboles también sirven para regresión. Cambiemos el ejemplo: en lugar de predecir si se puede esquiar, queremos predecir cuántos centímetros de nieve caen por día, con las mismas dos features, el mes y la latitud. La @fig-esqui-regresion muestra el mismo plano de la @fig-esqui, pero en lugar de $+$ y $-$ cada ejemplo tiene un número: los centímetros de nieve de ese lugar y ese mes. En los inviernos de cada hemisferio los números son grandes, y en el resto son ceros.

#let ce = d-esqui-reg
#let w-reg(k) = num(ce.regiones.at(k).at(4), digits: 1)

#figure(
  caption: [El problema del esquí como regresión, con #ce.m ejemplos: cada número es la cantidad de nieve (en cm por día) de un ejemplo. Las líneas son las regiones del árbol de la @fig-arbol-completo. Cada hoja predice el promedio de los números de su región: #w-reg(0), #w-reg(2) y #w-reg(5) cm en las tres regiones de invierno (arriba a la izquierda, arriba a la derecha y abajo al centro), y 0 en las demás.],
  [
    #panel-esqui(size: (7.5, 5.5), {
      for (x0, x1, y0, y1, _) in ce.regiones {
        plot.add(
          ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)),
          style: (stroke: (paint: black, thickness: 1.2pt)),
        )
      }
      plot.annotate({
        import cetz.draw: *
        for (x, y, v) in ce.puntos {
          content((x, y), text(size: 10pt, fill: if v > 0 { c-pos } else { gray.darken(30%) })[#v])
        }
      })
    })
  ],
) <fig-esqui-regresion>

Solo cambian dos cosas respecto a la clasificación. Primero, cada hoja predice el promedio de las etiquetas de los ejemplos que caen en ella:

$
  w_j = 1 / abs(R_j) sum_(i : fvec(x)^(\(i\)) in R_j) y^(\(i\)).
$

Segundo, la impureza de una región es el error cuadrático medio de esa predicción, que es la varianza de las etiquetas en la región:

$
  L_"cuadrática" (R_j) = 1 / abs(R_j) sum_(i : fvec(x)^(\(i\)) in R_j) (y^(\(i\)) - w_j)^2.
$

La ganancia se define igual que antes, con los mismos pesos.

Para ver mejor la forma de la predicción, quedémonos con una sola feature: la nieve en un lugar del hemisferio norte, según el mes. En la @fig-regresion se ve el resultado: una función constante a trozos, con un escalón por cada hoja. Con más hojas, la escalera sigue más de cerca a los datos.

#let panel-reg(p, y-label: none) = custom-plot(
  x-min: 1,
  x-max: 13,
  y-min: 0,
  y-max: 28,
  x-label: [mes],
  y-label: y-label,
  x-tick-step: 3,
  y-tick-step: 5,
  size: (4.4, 3.4),
  {
    puntos-reg(d-reg.puntos, tam: 0.07)
    curva(p.f, color: c-pos, grosor: 1.4pt)
  },
)

#figure(
  caption: [Árboles de regresión sobre #d-reg.m ejemplos, con profundidad 1 (#d-reg.paneles.p1.hojas hojas), 2 (#d-reg.paneles.p2.hojas hojas) y sin límite (#d-reg.paneles.p8.hojas hojas). El error cuadrático medio de entrenamiento es #num(d-reg.paneles.p1.ecm, digits: 1), #num(d-reg.paneles.p2.ecm, digits: 1) y #num(d-reg.paneles.p8.ecm, digits: 1). El árbol sin límite pasa exactamente por cada ejemplo, incluido el ruido.],
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.3cm,
    panel-reg(d-reg.paneles.p1, y-label: [cm de nieve]), panel-reg(d-reg.paneles.p2), panel-reg(d-reg.paneles.p8),
  ),
) <fig-regresion>

Hay una relación directa entre Gini y la pérdida cuadrática. En clasificación binaria, codifiquemos los positivos como $y = 1$ y los negativos como $y = 0$. El promedio de las etiquetas en la región es $hat(p)$, y su varianza es

$
  1 / abs(R) sum_i (y^(\(i\)) - hat(p))^2 = hat(p) (1 - hat(p))^2 + (1 - hat(p)) hat(p)^2 = hat(p) (1 - hat(p)),
$

porque una fracción $hat(p)$ de los ejemplos tiene error $1 - hat(p)$ y el resto tiene error $-hat(p)$. Es exactamente la mitad de la impureza de Gini. Es decir, un árbol de clasificación con Gini es un árbol de regresión que predice la probabilidad de la clase positiva con pérdida cuadrática. Esta idea, que todo se reduce a ajustar promedios con pérdida cuadrática, reaparecerá en boosting.

== Variables categóricas

Los árboles manejan variables categóricas de forma natural. En lugar de comparar con un umbral, la pregunta es si la categoría pertenece a un subconjunto. Por ejemplo, si en lugar de la latitud tenemos la ubicación como una de tres categorías, hemisferio norte (N), Ecuador (E) y hemisferio sur (S), una pregunta posible es "¿ubicación $in {N}$?", y otra es "¿ubicación $in {N, S}$?".

El problema es que, con $q$ categorías, hay $2^(q - 1) - 1$ formas de dividirlas en dos grupos. Con $q = 30$, son más de 500 millones. Para clasificación binaria hay un truco: se ordenan las categorías según su proporción de positivos y se prueban solo los $q - 1$ cortes de esa lista ordenada. Se puede demostrar que el mejor corte (con Gini o entropía) está entre ellos. En otros casos, con muchas categorías, se suelen agrupar o codificar de otra forma.

== Regularización y poda

Si dejamos que el árbol crezca sin restricciones, sigue dividiendo hasta que cada hoja es pura, lo que en el peor caso significa una hoja por ejemplo. El error de entrenamiento es cero, pero el árbol memorizó los datos, incluido el ruido (@fig-regresion, derecha). Los árboles de decisión son modelos de alta varianza: pequeños cambios en los datos producen árboles muy distintos.

Para regularizarlos se usan reglas de parada:

- *Tamaño mínimo de hoja:* no dividir una región si quedaría con menos de cierto número de ejemplos.
- *Profundidad máxima:* limitar el número de preguntas desde la raíz hasta una hoja.
- *Número máximo de hojas.*
- *Ganancia mínima:* no dividir si la mejor ganancia es menor que un umbral.

La última parece la más natural, pero suele ser una mala idea. Hay preguntas que por sí solas no ayudan, pero que hacen posible una segunda pregunta muy buena. En el problema del esquí, preguntar solo por el mes no separa casi nada (en el mismo mes se puede esquiar en un hemisferio pero no en el otro), pero combinado con la latitud es muy útil. La @fig-xor muestra el caso extremo.

#figure(
  caption: [Datos en los que cada clase ocupa dos esquinas opuestas. Cualquier pregunta individual deja los dos hijos casi tan mezclados como el padre: la mejor ganancia de Gini posible en la raíz es #num(d-xor.ganancia_max, digits: 3) (el padre tiene impureza #num(d-xor.gini_padre)). Pero después de la pregunta $x_1 < 2$, la pregunta $x_2 < 2$ en cada hijo separa todo perfectamente.],
  [
    #panel-2d(size: (4.5, 4.5), {
      segmento((2, 0), (2, 4), estilo: (paint: black, thickness: 1.3pt))
      segmento((0, 2), (4, 2), estilo: (paint: black, thickness: 1.3pt, dash: "dashed"))
      puntos-clases(d-xor, tam: 0.11)
    }, x-max: 4, y-max: 4)
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-xor>

La alternativa recomendada es la poda (pruning). Primero se deja crecer el árbol completo. Luego se recorren las hojas de abajo hacia arriba y se eliminan las divisiones que no mejoran el error sobre un conjunto de validación, juntando las dos hojas en una sola. Así el árbol puede descubrir interacciones como la de la @fig-xor, y solo después decidimos qué partes valen la pena.

== Evaluar un clasificador: exactitud y matriz de confusión

Hasta ahora hemos medido qué tan bueno es un clasificador con su exactitud (accuracy), la fracción de ejemplos que clasifica bien, o su complemento, el error. La exactitud es fácil de interpretar, pero puede engañar.

Volvamos al esquí, ahora con 400 ejemplos, de los que separamos #cf.m_test para prueba. Solo #cf.pos_test de esos #cf.m_test ejemplos son positivos: en la mayoría de los lugares y meses no se puede esquiar. Un "modelo" que ignora los datos y siempre responde "no se puede esquiar" acierta en todos los negativos, así que su exactitud es #pct-cf(cf.siempre_no). Parece buena, pero el modelo no sirve para nada: nunca encuentra un día para esquiar. Cuando una clase es mucho más frecuente que la otra (se dice que las clases están desbalanceadas), la exactitud por sí sola no basta.

La matriz de confusión separa los aciertos y los errores según la clase. Para clasificación binaria es una tabla de $2 times 2$: las filas son la clase real y las columnas la clase predicha.

- *Verdaderos positivos* (VP): positivos que el modelo predice como positivos. Días buenos para esquiar que el modelo detecta.
- *Verdaderos negativos* (VN): negativos predichos como negativos.
- *Falsos positivos* (FP): negativos que el modelo predice como positivos. Vamos a esquiar y no hay nieve.
- *Falsos negativos* (FN): positivos que el modelo predice como negativos. Nos perdemos un buen día para esquiar.

La exactitud es la suma de la diagonal dividida entre el total: $("VP" + "VN") slash ("VP" + "VN" + "FP" + "FN")$. La @fig-confusion compara las matrices del modelo que siempre dice "no" y de un árbol de profundidad 3.

#figure(
  kind: image,
  caption: [Matrices de confusión sobre los #cf.m_test ejemplos de prueba. Las filas son la clase real y las columnas la predicha. A la izquierda, el modelo que siempre predice "no" tiene exactitud #pct-cf(cf.siempre_no), pero cero verdaderos positivos. A la derecha, un árbol de profundidad 3 tiene exactitud #pct-cf(cf.arbol): no comete ningún falso positivo, pero se le escapan #cf.arbol.at(1).at(0) de los #cf.pos_test días buenos.],
  grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [Siempre "no" #v(0.15cm) #matriz-confusion(cf.siempre_no)],
    [Árbol de profundidad 3 #v(0.15cm) #matriz-confusion(cf.arbol)],
  ),
) <fig-confusion>

De la matriz se leen dos medidas que responden preguntas más específicas que la exactitud:

$
  "precisión" = "VP" / ("VP" + "FP"), quad quad "sensibilidad" = "VP" / ("VP" + "FN").
$

La precisión (precision) responde: de los días en que el modelo dijo "sí", ¿qué fracción eran buenos? La sensibilidad (recall) responde: de los días buenos, ¿qué fracción detectó el modelo? Para el árbol de la figura, la precisión es #pct-cf-precision(cf.arbol) y la sensibilidad es #pct-cf-sensibilidad(cf.arbol): cada vez que dice "sí" acierta, pero detecta poco más de la mitad de los días buenos. El modelo que siempre dice "no" tiene sensibilidad 0 (y su precisión ni siquiera está definida, porque nunca dice "sí").

Qué medida importa más depende del problema. En un examen médico para detectar una enfermedad, un falso negativo (no detectar a un enfermo) suele ser mucho más grave que un falso positivo, y se busca una sensibilidad alta. En un filtro de spam, un falso positivo (mandar a spam un correo importante) es peor, y se busca una precisión alta.

== Ventajas y desventajas

Ventajas:

- *Fáciles de explicar e interpretar.* Se puede dibujar el árbol y ver exactamente qué preguntas llevan a cada predicción.
- *Manejan variables categóricas y mezclas de variables de distinto tipo* sin preprocesamiento.
- *No les afectan las transformaciones monótonas de las features.* Como solo comparan con umbrales, medir la temperatura en grados Celsius o Fahrenheit, o tomar logaritmo de una feature, produce exactamente el mismo árbol. En particular, no hace falta estandarizar, a diferencia de las SVM con kernel gaussiano.
- *Rápidos* de entrenar y de evaluar.

Desventajas:

- *Alta varianza.* Tienden al overfitting, y pequeños cambios en los datos producen árboles muy distintos.
- *Les cuesta capturar estructuras aditivas.* Una frontera tan simple como $x_2 = x_1$, que la regresión logística encuentra de inmediato, obliga al árbol a aproximarla con una escalera de muchas preguntas (@fig-diagonal).
- Por estas dos razones, un árbol solo suele tener *baja precisión predictiva*.

#figure(
  caption: [Los positivos están arriba de la recta $x_2 = x_1$. A la izquierda, un clasificador lineal la encuentra con un solo parámetro de pendiente. A la derecha, un árbol completo necesita #d-diag.n_hojas hojas para separar los ejemplos de entrenamiento, y su frontera es una escalera que solo aproxima la diagonal.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      panel-2d(size: (4.5, 4.5), {
        segmento((0, 0), (5, 5))
        puntos-clases(d-diag, tam: 0.1)
      }),
      panel-2d(size: (4.5, 4.5), {
        regiones(d-diag.hojas)
        segmento((0, 0), (5, 5), estilo: (paint: gray, thickness: 0.8pt, dash: "dotted"))
        puntos-clases(d-diag, tam: 0.1)
      }),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-diagonal>

Lo interesante es que la primera desventaja tiene solución. El resto de la clase trata de cómo combinar muchos árboles para reducir su varianza (bagging y random forests) o su sesgo (boosting).

En scikit-learn, las clases `DecisionTreeClassifier` y `DecisionTreeRegressor` implementan estos árboles. El criterio por defecto en clasificación es `criterion="gini"`. Los parámetros `max_depth`, `min_samples_leaf` y `max_leaf_nodes` corresponden a las reglas de parada, y `ccp_alpha` a una forma de poda.

= Parte 4: Ensamblaje: bagging y random forests

== Por qué promediar modelos

Un ensamblaje (ensemble) es un modelo que combina las predicciones de varios modelos. La razón por la que funciona viene de la estadística básica.

Sean $X_1, ..., X_B$ variables aleatorias independientes e idénticamente distribuidas (i.i.d.), cada una con varianza $sigma^2$. La varianza de su promedio $overline(X) = 1/B sum_b X_b$ es

$
  "Var"(overline(X)) = sigma^2 / B.
$

Promediar $B$ variables independientes divide la varianza entre $B$. Si pensamos en cada $X_b$ como la predicción de un modelo entrenado con un conjunto de datos distinto, promediar muchos modelos reduce la varianza tanto como queramos.

El problema es que los modelos no son independientes: se entrenan con los mismos datos, o con datos parecidos. Supongamos que las $X_b$ siguen teniendo varianza $sigma^2$, pero que cada par tiene correlación $rho$, es decir, $"Cov"(X_b, X_(b')) = rho sigma^2$ para $b != b'$. Entonces

$
  "Var"(overline(X)) = 1 / B^2 sum_(b=1)^B sum_(b'=1)^B "Cov"(X_b, X_(b'))
  = 1 / B^2 (B sigma^2 + B(B - 1) rho sigma^2)
  = rho sigma^2 + (1 - rho) / B sigma^2.
$

(En la doble suma hay $B$ términos con $b = b'$, que son varianzas, y $B(B - 1)$ con $b != b'$, que son covarianzas.)

Al aumentar $B$, el segundo término desaparece, pero el primero no: la varianza nunca baja de $rho sigma^2$ (@fig-varianza). Esto dice exactamente qué hay que hacer para que un ensamblaje funcione:

+ Usar muchos modelos ($B$ grande), para eliminar el segundo término.
+ Hacer que los modelos estén poco correlacionados ($rho$ pequeño), para reducir el primero.

#let colores-rho = (c-neg, c-3, c-pos)

#figure(
  caption: [Varianza del promedio de $B$ modelos con varianza $sigma^2 = 1$, para tres valores de la correlación $rho$. Con modelos independientes la varianza tiende a cero. Con modelos correlacionados se estanca en $rho sigma^2$.],
  [
    #custom-plot(
      x-min: 1,
      x-max: 50,
      y-min: 0,
      y-max: 1,
      x-label: $B$,
      y-label: [$"Var"(overline(X))$],
      x-tick-step: 10,
      y-tick-step: 0.25,
      size: (7, 4),
      {
        for (k, r) in (0, 0.2, 0.5).enumerate() {
          plot.add(
            domain: (1, 50),
            samples: 200,
            B => r + (1 - r) / B,
            style: (stroke: (paint: colores-rho.at(k), thickness: 1.6pt)),
          )
        }
      },
    )
    #v(0.2cm)
    #leyenda-lineas(..range(3).map(k => linea-leyenda(colores-rho.at(k), $rho = #(0, 0.2, 0.5).at(k)$)))
  ],
) <fig-varianza>

Una forma de obtener modelos poco correlacionados es usar algoritmos distintos: por ejemplo, promediar una SVM, una regresión logística y un árbol. Es muy común en competencias como las de Kaggle. Pero también hay métodos que generan automáticamente muchos modelos distintos a partir de un solo algoritmo. El primero es bagging.

== Bootstrap

Idealmente, para obtener modelos independientes, tomaríamos varios conjuntos de entrenamiento independientes $S_1, S_2, ..., S_B$ de la población, y entrenaríamos un modelo con cada uno. Pero solo tenemos un conjunto de entrenamiento $S$, con $m$ ejemplos.

El bootstrap es una técnica de la estadística para simular esa situación. Supone que el conjunto de entrenamiento $S$ representa bien a la población, y genera conjuntos nuevos $Z_1, ..., Z_B$ sacando $m$ ejemplos de $S$ al azar con reemplazo. Como el muestreo es con reemplazo, cada $Z_b$ contiene algunos ejemplos repetidos y le faltan otros (@tab-bootstrap).

#figure(
  caption: [Tres muestras bootstrap de un conjunto de #d-boot.m ejemplos, numerados del 1 al #d-boot.m. Cada muestra tiene #d-boot.m elementos, pero solo #d-boot.unicos.at(0), #d-boot.unicos.at(1) y #d-boot.unicos.at(2) ejemplos distintos.],
  table(
    columns: 3,
    align: (left, left, center),
    table.header([Muestra], [Ejemplos], [Distintos]),
    ..range(3)
      .map(k => (
        $Z_#(k + 1)$,
        d-boot.muestras.at(k).map(str).join(", "),
        str(d-boot.unicos.at(k)),
      ))
      .flatten(),
  ),
) <tab-bootstrap>

¿Qué fracción de los ejemplos originales aparece en cada muestra? En cada uno de los $m$ sorteos, un ejemplo dado no sale con probabilidad $1 - 1 slash m$. La probabilidad de que no salga en ninguno es $(1 - 1 slash m)^m$, así que la probabilidad de que aparezca al menos una vez es

$
  1 - (1 - 1 / m)^m -> 1 - e^(-1) approx 0.632 quad "cuando" m -> infinity.
$

Es decir, cada muestra bootstrap contiene aproximadamente dos tercios de los ejemplos distintos del conjunto original. (Para $m = 10$, la fracción esperada es #num(d-boot.fraccion.at("10"), digits: 3).)

== Bagging

Bagging (de bootstrap aggregating) consiste en:

+ Generar $B$ muestras bootstrap $Z_1, ..., Z_B$.
+ Entrenar un modelo $G_b$ con cada muestra $Z_b$.
+ Promediar las predicciones:

$
  G(fvec(x)) = 1 / B sum_(b=1)^B G_b (fvec(x)).
$

En clasificación, se promedian las probabilidades que predice cada modelo, o cada modelo vota por una clase y gana la mayoría.

¿Qué efecto tiene sobre el sesgo y la varianza?

- *Reduce la varianza.* Los modelos se entrenan con datos distintos, así que están menos correlacionados que si usaran todos el mismo conjunto: el bootstrap reduce $rho$. Y al aumentar $B$, el término $(1 - rho) sigma^2 slash B$ desaparece. Un detalle importante: *aumentar $B$ no produce overfitting.* Más modelos solo acercan la varianza a su límite $rho sigma^2$. El único costo de un $B$ grande es computacional.
- *Aumenta un poco el sesgo.* Cada modelo ve solo alrededor del 63% de los ejemplos distintos, así que individualmente es algo peor que un modelo entrenado con todos los datos.

Bagging funciona especialmente bien con árboles de decisión. Un árbol completo tiene sesgo bajo y varianza alta: exactamente el tipo de modelo que se beneficia de reducir la varianza a cambio de un poco de sesgo. Además, los árboles son muy sensibles a los datos, así que las muestras bootstrap producen árboles bastante distintos entre sí, con $rho$ relativamente bajo. Con un modelo muy estable, como una regresión lineal, todas las muestras producirían casi el mismo modelo y bagging no ganaría casi nada.

#let panel-bag(p, y-label: none) = custom-plot(
  x-min: -1.8,
  x-max: 2.8,
  y-min: -1.4,
  y-max: 1.9,
  x-label: $x_1$,
  y-label: y-label,
  x-tick-step: 1,
  y-tick-step: 1,
  size: (4.5, 3.4),
  {
    for s in p.frontera {
      plot.add(s, style: (stroke: (paint: black, thickness: 1.2pt)))
    }
    puntos-clases(d-bag, tam: 0.08)
  },
)

#figure(
  caption: [Fronteras de decisión con #d-bag.m ejemplos ruidosos: un solo árbol completo (izquierda), bagging con $B = #d-bag.B$ árboles (centro) y un random forest con $B = #d-bag.B$ árboles que en cada división consideran una sola feature al azar (derecha). Los tres clasifican correctamente todos los ejemplos de entrenamiento, pero la exactitud sobre datos de prueba es #pct(d-bag.paneles.arbol.exactitud_test), #pct(d-bag.paneles.bagging.exactitud_test) y #pct(d-bag.paneles.rf.exactitud_test). El árbol individual tiene una frontera llena de entrantes que siguen el ruido. Al promediar, la frontera se suaviza.],
  [
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 0.6cm,
      panel-bag(d-bag.paneles.arbol, y-label: $x_2$), panel-bag(d-bag.paneles.bagging), panel-bag(d-bag.paneles.rf),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-bagging>

== Random forests

Bagging reduce $rho$, pero no tanto como quisiéramos. Supongamos que hay una feature mucho más predictiva que las demás. Entonces casi todos los árboles, aunque se entrenen con muestras bootstrap distintas, van a escoger esa misma feature en la raíz, y probablemente las mismas en los niveles siguientes. Los árboles terminan pareciéndose mucho, y $rho$ queda alto.

Un random forest (bosque aleatorio) es bagging con árboles más una fuente adicional de aleatoriedad: *en cada división, el árbol solo puede escoger entre un subconjunto de $k < d$ features sorteadas al azar.* El sorteo se repite en cada nodo de cada árbol. En el ejemplo del esquí, un árbol podría estar obligado a preguntar primero por el mes, porque en ese nodo la latitud no estaba disponible, y otro árbol a preguntar primero por la latitud.

Así, la feature dominante no puede aparecer en todos los árboles en la misma posición, y los árboles quedan más distintos entre sí. Cada árbol individual es un poco peor (sesgo un poco mayor), pero la correlación $rho$ baja, y el promedio suele ser mejor que el de bagging. Valores típicos son $k approx sqrt(d)$ en clasificación y $k approx d slash 3$ en regresión.

En scikit-learn, `RandomForestClassifier` y `RandomForestRegressor` implementan random forests. El parámetro `n_estimators` es $B$ y `max_features` es $k$. Con `max_features=None` se considera todas las features en cada división, y se obtiene bagging. Random forests es probablemente el mejor algoritmo "listo para usar": funciona bien con los parámetros por defecto y es difícil que haga mucho overfitting.

= Parte 5: Boosting

== Idea: combinar modelos débiles

Bagging y random forests promedian modelos que se entrenan en paralelo, cada uno sin saber nada de los demás, y reducen la varianza. Boosting hace lo contrario: entrena los modelos uno después de otro, y cada modelo nuevo trata de corregir los errores de los anteriores. El objetivo principal es reducir el sesgo.

Por eso, en lugar de árboles completos (sesgo bajo y varianza alta), boosting usa modelos con sesgo alto y varianza baja. El caso extremo es el tocón (decision stump), un árbol con una sola pregunta y dos hojas. Un tocón por sí solo es un modelo muy débil, apenas mejor que adivinar. La pregunta de boosting es: ¿se puede construir un modelo fuerte sumando muchos modelos débiles?

== AdaBoost

AdaBoost (adaptive boosting) fue el primer algoritmo de boosting exitoso. Usamos etiquetas $y in {-1, 1}$, como en las SVM. La idea es mantener un peso $w_i$ para cada ejemplo de entrenamiento y entrenar cada modelo nuevo dándole más importancia a los ejemplos en los que los anteriores se equivocaron.

+ Inicializar los pesos: $w_i = 1 slash m$ para $i = 1, ..., m$.
+ Para $b = 1, ..., B$:
  + Entrenar un clasificador $G_b$ con los ejemplos ponderados por los pesos $w_i$. (Para un árbol, basta con usar las proporciones ponderadas al calcular la impureza: cada ejemplo cuenta $w_i$ veces.)
  + Calcular su error ponderado:
    $ "err"_b = (sum_(i=1)^m w_i bb(1){y^(\(i\)) != G_b (fvec(x)^(\(i\)))}) / (sum_(i=1)^m w_i). $
  + Calcular su peso en la votación: $alpha_b = log((1 - "err"_b) / "err"_b)$.
  + Aumentar el peso de los ejemplos mal clasificados: $w_i <- w_i exp(alpha_b bb(1){y^(\(i\)) != G_b (fvec(x)^(\(i\)))})$.
+ El clasificador final es una votación ponderada:
  $ G(fvec(x)) = "signo"(sum_(b=1)^B alpha_b G_b (fvec(x))). $

Algunas observaciones:

- Si $"err"_b < 1 slash 2$, es decir, si el clasificador es mejor que adivinar, entonces $alpha_b > 0$. Cuanto menor es el error, mayor es $alpha_b$, y más pesa ese clasificador en la votación final.
- Los ejemplos bien clasificados conservan su peso, y los mal clasificados lo multiplican por $e^(alpha_b) = (1 - "err"_b) slash "err"_b$. Con un error del 10%, por ejemplo, el peso de cada error se multiplica por 9. Así, el siguiente clasificador se concentra en los ejemplos difíciles.

#let pesos-r(r) = d-ada.rondas.at(r).pesos

// Un panel por ronda: el tocón, sus dos regiones y los puntos con tamaño según el peso
#let panel-ronda(r) = {
  let ronda = d-ada.rondas.at(r)
  let t = ronda.umbral
  let izq = if ronda.izq == 1 { 1 } else { 0 }
  let der = 1 - izq
  panel-2d(size: (3.6, 3.6), {
    if ronda.feature == 0 {
      region(0, t, 0, 5, izq, alfa: 88%)
      region(t, 5, 0, 5, der, alfa: 88%)
      segmento((t, 0), (t, 5))
    } else {
      region(0, 5, 0, t, izq, alfa: 88%)
      region(0, 5, t, 5, der, alfa: 88%)
      segmento((0, t), (5, t))
    }
    for (i, p) in d-ada.puntos.enumerate() {
      let tam = 0.11 * calc.sqrt(ronda.pesos.at(i))
      puntos((p,), if d-ada.y.at(i) == 1 { 1 } else { -1 }, tam: tam)
    }
    marcar(ronda.mal.map(i => d-ada.puntos.at(i)), tam: 0.32)
  })
}

#figure(
  caption: [Tres rondas de AdaBoost con tocones. El tamaño de cada punto es proporcional a la raíz de su peso. En la ronda 1 todos pesan lo mismo, y el tocón (una pregunta sobre $x_2$) se equivoca en los dos ejemplos marcados con un círculo, con $"err"_1 = #num(d-ada.rondas.at(0).err, digits: 3)$ y $alpha_1 = #num(d-ada.rondas.at(0).alpha)$. En la ronda 2 esos ejemplos pesan 7 veces más que los demás, y el nuevo tocón (una pregunta sobre $x_1$) los clasifica bien a cambio de equivocarse en otros, de menor peso. Abajo a la derecha, la votación ponderada de los tres tocones, que ya clasifica correctamente todos los ejemplos.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 0.5cm,
      [Ronda 1 #panel-ronda(0)], [Ronda 2 #panel-ronda(1)],
      [Ronda 3 #panel-ronda(2)],
      [
        Votación final
        #panel-2d(size: (3.6, 3.6), {
          regiones(d-ada.regiones, alfa: 88%)
          for s in d-ada.frontera {
            plot.add(s, style: (stroke: (paint: black, thickness: 1.3pt)))
          }
          for (i, p) in d-ada.puntos.enumerate() {
            puntos((p,), if d-ada.y.at(i) == 1 { 1 } else { -1 }, tam: 0.11)
          }
        })
      ],
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-adaboost>

La frontera final de la @fig-adaboost no es un rectángulo ni una recta: sumando tres preguntas muy simples obtuvimos una frontera que ninguna de ellas podía producir sola. Con más rondas se pueden obtener fronteras muy complejas. Por eso boosting reduce el sesgo.

La @fig-error-B compara los métodos de esta clase en un problema con 10 features. Un tocón solo tiene un error de prueba de #pct(d-err.tocon), casi como adivinar. Un árbol completo tiene #pct(d-err.arbol). Bagging y random forests mejoran el árbol completo y se estabilizan a medida que crece $B$. AdaBoost con tocones empieza muy mal, pero sigue mejorando ronda tras ronda, y termina siendo el mejor de los tres.

#figure(
  caption: [Error de prueba en función del número de modelos $B$, con #d-err.m ejemplos de entrenamiento y #d-err.m_test de prueba. Los datos son un problema clásico del libro _The Elements of Statistical Learning_ (Hastie, Tibshirani y Friedman): 10 features gaussianas, y la clase depende de si la suma de sus cuadrados supera un umbral. Las rectas horizontales son un solo tocón y un solo árbol completo.],
  [
    #custom-plot(
      x-min: 0,
      x-max: 400,
      y-min: 0,
      y-max: 0.5,
      x-label: $B$,
      y-label: [error de prueba],
      x-tick-step: 100,
      y-tick-step: 0.1,
      size: (8, 4.5),
      {
        plot.add-hline(d-err.tocon, style: (stroke: (paint: gray, thickness: 1pt, dash: "dotted")))
        plot.add-hline(d-err.arbol, style: (stroke: (paint: gray.darken(30%), thickness: 1pt, dash: "dashed")))
        curva(d-err.bagging, color: c-neg, grosor: 1.5pt)
        curva(d-err.rf, color: c-3, grosor: 1.5pt)
        curva(d-err.adaboost, color: c-pos, grosor: 1.5pt)
      },
    )
    #v(0.2cm)
    #leyenda-lineas(
      linea-leyenda(c-neg, [bagging]),
      linea-leyenda(c-3, [random forest]),
      linea-leyenda(c-pos, [AdaBoost con tocones]),
      linea-leyenda(gray.darken(30%), [un árbol], punteada: true),
      linea-leyenda(gray, [un tocón], punteada: true),
    )
  ],
) <fig-error-B>

AdaBoost se puede derivar como un caso particular de una idea más general, que es la base de los métodos de boosting que se usan hoy: gradient boosting y XGBoost. Para llegar a ellos, conviene empezar por regresión.

== Boosting con pérdida cuadrática: ajustar los residuos

Volvamos a los datos de nieve de la Parte 3. Queremos construir un modelo como una suma de árboles pequeños:

$
  F_B (fvec(x)) = F_0 (fvec(x)) + h_1 (fvec(x)) + h_2 (fvec(x)) + dots.c + h_B (fvec(x)),
$

donde $F_0$ es un modelo inicial muy simple (por ejemplo, el promedio de todas las etiquetas) y cada $h_b$ es un árbol de regresión. Igual que en AdaBoost, construimos el modelo un árbol a la vez: $F_b = F_(b-1) + h_b$.

¿Qué árbol $h_b$ hay que agregar? Con pérdida cuadrática, queremos que $F_(b-1)(fvec(x)^(\(i\))) + h_b (fvec(x)^(\(i\)))$ se acerque lo más posible a $y^(\(i\))$. Es decir, queremos que

$
  h_b (fvec(x)^(\(i\))) approx y^(\(i\)) - F_(b-1)(fvec(x)^(\(i\))) = r_i.
$

La diferencia $r_i$ entre la etiqueta y la predicción actual se llama residuo: es lo que al modelo actual todavía le falta explicar. Así que cada árbol nuevo es simplemente un árbol de regresión entrenado para predecir los residuos del modelo actual (@fig-residuos). Es la misma idea de AdaBoost, "concentrarse en los errores del modelo anterior", pero en lugar de repesar los ejemplos cambiamos las etiquetas.

#figure(
  caption: [El primer paso de boosting con los datos de nieve. A la izquierda, el modelo inicial $F_0$ es el promedio de las etiquetas (#num(d-gb.f0, digits: 1) cm). A la derecha, los residuos $r_i = y^(\(i\)) - F_0 (fvec(x)^(\(i\)))$ y el primer árbol $h_1$ (de profundidad #d-gb.prof), entrenado para predecirlos.],
  grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    custom-plot(
      x-min: 1,
      x-max: 13,
      y-min: 0,
      y-max: 28,
      x-label: [mes],
      y-label: $y$,
      x-tick-step: 3,
      y-tick-step: 5,
      size: (5, 3.6),
      {
        puntos-reg(d-gb.puntos)
        plot.add-hline(d-gb.f0, style: (stroke: (paint: c-pos, thickness: 1.4pt)))
      },
    ),
    custom-plot(
      x-min: 1,
      x-max: 13,
      y-min: -12,
      y-max: 20,
      x-label: [mes],
      y-label: [residuo $r$],
      x-tick-step: 3,
      y-tick-step: 5,
      size: (5, 3.6),
      {
        plot.add-hline(0, style: (stroke: (paint: gray, thickness: 0.6pt)))
        puntos-reg(d-gb.primer.residuos)
        curva(d-gb.primer.h, color: c-neg, grosor: 1.4pt)
      },
    ),
  ),
) <fig-residuos>

En la práctica no se suma el árbol completo, sino una fracción:

$
  F_b (fvec(x)) = F_(b-1)(fvec(x)) + nu h_b (fvec(x)),
$

donde $0 < nu <= 1$ se llama tasa de aprendizaje (learning rate o shrinkage), y valores típicos son $nu = 0.1$ o $0.3$. Con $nu$ pequeño cada árbol corrige solo una parte del error y deja el resto para los siguientes, lo que hace el proceso más lento pero menos propenso al overfitting. La @fig-boosting-pasos muestra cómo se va formando el modelo.

#let panel-gb(b) = custom-plot(
  x-min: 1,
  x-max: 13,
  y-min: 0,
  y-max: 28,
  x-label: [mes],
  y-label: [],
  x-tick-step: 3,
  y-tick-step: 5,
  size: (3.3, 2.8),
  {
    puntos-reg(d-gb.puntos, tam: 0.06)
    curva(d-gb.instantes.at(str(b)), color: c-pos, grosor: 1.3pt)
  },
)

#figure(
  caption: [El modelo $F_B$ después de $B = 1$, 3, 10 y 50 árboles de profundidad #d-gb.prof, con $nu = #d-gb.nu$. Cada árbol tiene solo 4 hojas, pero la suma de muchos de ellos aproxima la forma de los datos.],
  grid(
    columns: 4,
    gutter: 0.2cm,
    ..(1, 3, 10, 50).map(b => [$B = #b$ #panel-gb(b)]),
  ),
) <fig-boosting-pasos>

A diferencia de bagging, en boosting *aumentar $B$ sí puede producir overfitting*: cada árbol nuevo ajusta los residuos que quedan, y cuando ya solo queda ruido, el modelo empieza a memorizarlo. En la @fig-gb-sobreajuste el error de entrenamiento baja hasta casi cero, mientras que el error de prueba llega a un mínimo y luego empieza a subir. Por eso $B$ se escoge con validación, o se detiene el entrenamiento cuando el error de validación deja de mejorar (early stopping).

#let (b-min, _, ecm-min) = d-gb.curva.fold(d-gb.curva.first(), (a, c) => if c.at(2) < a.at(2) { c } else { a })

#figure(
  caption: [Error cuadrático medio de entrenamiento y de prueba de boosting con los datos de nieve, en función del número de árboles (los primeros valores, cercanos a 28, quedan fuera de la gráfica). El error de prueba es mínimo con $B = #b-min$ (#num(ecm-min, digits: 1)) y luego sube hasta #num(d-gb.curva.last().at(2), digits: 1) con $B = #d-gb.curva.last().at(0)$, mientras que el de entrenamiento sigue bajando.],
  [
    #custom-plot(
      x-min: 0,
      x-max: 300,
      y-min: 0,
      y-max: 12,
      x-label: $B$,
      y-label: [error cuadrático medio],
      x-tick-step: 50,
      y-tick-step: 2,
      size: (7, 3.8),
      {
        curva(d-gb.curva.map(c => (c.at(0), c.at(1))), color: c-neg, grosor: 1.5pt)
        curva(d-gb.curva.map(c => (c.at(0), c.at(2))), color: c-pos, grosor: 1.5pt)
      },
    )
    #v(0.2cm)
    #leyenda-lineas(linea-leyenda(c-neg, [entrenamiento]), linea-leyenda(c-pos, [prueba]))
  ],
) <fig-gb-sobreajuste>

== Gradient boosting

Ajustar residuos tiene sentido con pérdida cuadrática, pero ¿qué hacemos con otras pérdidas, como la de la regresión logística? La clave es ver los residuos de otra forma.

Pensemos en las predicciones del modelo sobre los ejemplos de entrenamiento como $m$ números, $hat(y)_i = F_(b-1)(fvec(x)^(\(i\)))$, y en la pérdida total como función de ellos:

$
  cal(L) = sum_(i=1)^m ell(y^(\(i\)), hat(y)_i).
$

Si pudiéramos mover cada $hat(y)_i$ libremente, para reducir la pérdida daríamos un paso de descenso por gradiente: moveríamos cada $hat(y)_i$ en la dirección de

$
  r_i = - (partial ell(y^(\(i\)), hat(y)_i)) / (partial hat(y)_i).
$

Con la pérdida cuadrática $ell(y, hat(y)) = 1/2 (y - hat(y))^2$, la derivada es $-(y - hat(y))$, y $r_i = y^(\(i\)) - hat(y)_i$ es exactamente el residuo. Es decir, ajustar los residuos es dar un paso de descenso por gradiente.

Pero no podemos mover los $hat(y)_i$ libremente: necesitamos una función que también sirva para puntos nuevos. La solución es entrenar un árbol de regresión $h_b$ para predecir los $r_i$, y sumarlo con la tasa de aprendizaje, $F_b = F_(b-1) + nu h_b$. El árbol "generaliza" el paso de gradiente a todo el espacio.

Esto es gradient boosting: el mismo algoritmo de la sección anterior, pero en lugar de los residuos, cada árbol se entrena con los gradientes negativos de la pérdida, que se llaman pseudo-residuos. Para cambiar de problema basta con cambiar la pérdida. Por ejemplo, en clasificación binaria con $y in {0, 1}$, si el modelo predice el logit $hat(y)$ y la probabilidad es $sigma(hat(y))$, con la pérdida de la regresión logística el pseudo-residuo resulta ser $r_i = y^(\(i\)) - sigma(hat(y)_i)$: la diferencia entre la etiqueta y la probabilidad predicha. Otra vez, "lo que falta explicar".

En scikit-learn, `GradientBoostingRegressor` y `GradientBoostingClassifier` implementan gradient boosting con árboles. Los parámetros principales son `n_estimators` ($B$), `learning_rate` ($nu$) y `max_depth` (la profundidad de cada árbol, típicamente entre 2 y 6).

== XGBoost

XGBoost (extreme gradient boosting) es una implementación de gradient boosting con árboles que, desde que apareció en 2014, ha ganado una enorme cantidad de competencias y se usa constantemente en la industria. Es un caso raro en este curso: un algoritmo que está muy cerca del estado del arte para datos tabulares y que podemos entender con lo que ya sabemos. Sus diferencias principales con gradient boosting son tres.

*1. Regularización.* XGBoost minimiza la pérdida más una penalización por la complejidad de cada árbol:

$
  sum_(i=1)^m ell(y^(\(i\)), hat(y)_i) + sum_(b=1)^B (gamma J_b + 1/2 lambda sum_(j=1)^(J_b) w_(b j)^2),
$

donde $J_b$ es el número de hojas del árbol $b$ y $w_(b j)$ son los valores de sus hojas. El término $gamma J$ penaliza los árboles con muchas hojas, y $1/2 lambda sum w_j^2$ es una penalización $L_2$, como en la regresión ridge, sobre los valores de las hojas.

Veamos qué efecto tiene con pérdida cuadrática. Al agregar el árbol $b$, en cada hoja $R_j$ tenemos que escoger un valor $w_j$ que se suma a la predicción de todos los ejemplos de la hoja. Si $r_i$ son los residuos, hay que minimizar

$
  sum_(i in R_j) 1/2 (r_i - w_j)^2 + 1/2 lambda w_j^2.
$

Derivando respecto a $w_j$ e igualando a cero, $-sum_(i in R_j) (r_i - w_j) + lambda w_j = 0$, de donde

$
  w_j = (sum_(i in R_j) r_i) / (n_j + lambda),
$

donde $n_j$ es el número de ejemplos en la hoja. Sin regularización ($lambda = 0$), $w_j$ es el promedio de los residuos de la hoja, como en un árbol de regresión normal. Con $lambda > 0$, es un promedio "encogido" hacia cero. El efecto es mayor en las hojas con pocos ejemplos: si una hoja tiene un solo ejemplo con residuo 10, con $lambda = 1$ el valor de la hoja es $10 slash 2 = 5$; si tiene 100 ejemplos con residuo promedio 10, el valor es $1000 slash 101 approx 9.9$. Es decir, XGBoost desconfía de las hojas que se basan en pocos ejemplos.

Llamemos $S_j = sum_(i in R_j) r_i$ a la suma de los residuos de la hoja, así que $w_j = S_j slash (n_j + lambda)$. Sustituyendo este $w_j$ en la expresión anterior, el valor mínimo es

$
  1/2 sum_(i in R_j) r_i^2 - 1/2 S_j^2 / (n_j + lambda).
$

El primer término es el mismo sin importar cómo dividamos los ejemplos de la hoja, así que la ganancia de dividir una hoja en dos hijos $L$ y $R$ es

$
  "ganancia" = 1/2 (S_L^2 / (n_L + lambda) + S_R^2 / (n_R + lambda) - (S_L + S_R)^2 / (n_L + n_R + lambda)) - gamma.
$

Es el mismo esquema de la Parte 2: la pérdida del padre menos la de los hijos. Con $lambda = 0$ y $gamma = 0$ es (la mitad de) la reducción de la suma de errores cuadráticos, el mismo criterio de un árbol de regresión. El término $gamma$ hace que una división solo valga la pena si reduce la pérdida al menos en $gamma$. XGBoost hace crecer los árboles hasta una profundidad máxima y luego los poda, eliminando las divisiones con ganancia negativa.

*2. Segundo orden.* Para una pérdida cualquiera, XGBoost no usa solo el gradiente $g_i = partial ell slash partial hat(y)_i$, sino también la segunda derivada $h_i = partial^2 ell slash partial hat(y)_i^2$ (es un paso del método de Newton, en lugar de descenso por gradiente). Las fórmulas son las mismas de arriba, cambiando $sum r_i$ por $-sum g_i$ y el número de ejemplos $n_j$ por $sum h_i$:

$
  w_j = - (sum_(i in R_j) g_i) / (sum_(i in R_j) h_i + lambda).
$

Con pérdida cuadrática, $g_i = -r_i$ y $h_i = 1$, y recuperamos exactamente lo anterior. Con la pérdida logística, $h_i = sigma(hat(y)_i)(1 - sigma(hat(y)_i))$, que es pequeño cuando el modelo ya está muy seguro.

*3. Ideas de random forests.* Al igual que un random forest, XGBoost puede entrenar cada árbol con una fracción de los ejemplos (submuestreo de filas) y considerar solo una fracción de las features en cada árbol o en cada división (submuestreo de columnas). Esto reduce la correlación entre los árboles y el overfitting.

Además de esto, XGBoost tiene muchas optimizaciones de ingeniería (paralelismo, manejo de valores faltantes, búsqueda aproximada de umbrales) que lo hacen muy rápido, pero la parte de aprendizaje automático es la que acabamos de ver.

En la librería `xgboost`, las clases `XGBRegressor` y `XGBClassifier` tienen la misma interfaz que scikit-learn. Los parámetros principales son `n_estimators` ($B$), `learning_rate` ($nu$), `max_depth`, `reg_lambda` ($lambda$), `gamma` ($gamma$), `subsample` (fracción de ejemplos por árbol) y `colsample_bytree` (fracción de features por árbol).

= Resumen

- Un árbol de decisión divide el espacio de las features en regiones rectangulares haciendo preguntas de la forma $x_j < t$, de forma codiciosa y recursiva. Cada hoja predice la clase mayoritaria (clasificación) o el promedio de las etiquetas (regresión) de su región.
- En cada nodo se escoge la pregunta con mayor ganancia: la impureza del padre menos el promedio ponderado de la impureza de los hijos. El error de clasificación no sirve como impureza porque no es estrictamente cóncavo. La impureza de Gini, $sum_c hat(p)_c (1 - hat(p)_c)$, es la probabilidad de equivocarse al etiquetar un ejemplo al azar con las proporciones de la región, y es estrictamente cóncava. La entropía cruzada se comporta casi igual.
- En regresión, la impureza es el error cuadrático medio. En clasificación binaria, Gini es el doble de la varianza de las etiquetas $0$/$1$.
- Los árboles son interpretables, manejan variables categóricas y no necesitan estandarizar, pero tienen alta varianza. Se regularizan con reglas de parada y con poda sobre un conjunto de validación.
- El promedio de $B$ modelos con varianza $sigma^2$ y correlación $rho$ tiene varianza $rho sigma^2 + (1 - rho) sigma^2 slash B$. Bagging entrena cada modelo con una muestra bootstrap y reduce la varianza. Random forests además sortea las features disponibles en cada división, para reducir $rho$. En ambos, aumentar $B$ no produce overfitting.
- Boosting entrena modelos débiles de forma secuencial, cada uno enfocado en los errores de los anteriores, y reduce el sesgo. AdaBoost aumenta el peso de los ejemplos mal clasificados. Gradient boosting entrena cada árbol con los residuos (o, en general, con el gradiente negativo de la pérdida). En boosting, aumentar $B$ sí puede producir overfitting.
- XGBoost es gradient boosting con regularización ($gamma J + 1/2 lambda sum w_j^2$), que encoge los valores de las hojas hacia cero ($w_j = sum r_i slash (n_j + lambda)$), aproximación de segundo orden y submuestreo de ejemplos y features.

