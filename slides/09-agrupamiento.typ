// Slides con las figuras, el código y los ejercicios de notes/09-agrupamiento.typ
// Las figuras son las mismas de las notas (notes/09-figuras.typ).
// Compilar desde la raíz del repositorio:
//   typst compile --root . slides/09-agrupamiento.typ

#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../notes/09-figuras.typ": *

#set page(paper: "presentation-16-9", margin: (x: 1.6cm, y: 1.2cm))
#set text(lang: "es", size: 20pt)
#set smartquote(quotes: "“”")
#show raw: set text(font: "DejaVu Sans Mono")

#let argmin = math.op("argmin", limits: true)

#let frame(stroke) = (x, y) => (
  left: 0pt,
  right: 0pt,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)
#set table(stroke: frame(1pt + rgb("21222C")))

// Las figuras se dibujan con el tamaño de las notas y se amplían en la diapositiva
// (con el texto del tamaño de las notas, para que los ejes queden proporcionados)
#let ampliar(f, s) = {
  set text(size: 10pt)
  scale(s, reflow: true, f)
}
#let grande(f, s: 230%) = align(center + horizon, ampliar(f, s))

// Una diapositiva
#let slide(titulo, cuerpo) = {
  pagebreak(weak: true)
  text(size: 28pt, weight: "bold", titulo)
  v(0.3cm)
  block(width: 100%, height: 1fr, cuerpo)
}

#let ejercicio(titulo, figura, cuerpo, ancho-figura: 45%) = slide(titulo, align(horizon, grid(
  columns: (ancho-figura, 1fr),
  gutter: 1cm,
  align: horizon,
  align(center, figura), block(fill: luma(237), inset: 14pt, radius: 4pt, width: 100%, cuerpo),
)))

// Una diapositiva para cada parte del laboratorio
#let lab(parte, cuerpo) = slide([Laboratorio: Parte #parte], align(center + horizon, block(
  fill: luma(237),
  inset: 24pt,
  radius: 4pt,
  width: 80%,
  align(left, text(size: 24pt, cuerpo)),
)))

// Un bloque de código, leído del mismo archivo que usan las notas
#let bloque-codigo(nombre, desde: none, hasta: none, tam: 15pt) = block(
  fill: luma(245),
  stroke: 0.6pt + luma(200),
  inset: 14pt,
  radius: 4pt,
  align(left, {
    show raw: set text(size: tam)
    raw(fuente(nombre, desde: desde, hasta: hasta), lang: "python", block: true)
  }),
)

// Código y su salida: uno debajo del otro, o lado a lado
#let codigo-salida(nombre, desde: none, hasta: none, tam: 15pt, lado: false, ancho-codigo: 60%) = {
  let c = bloque-codigo(nombre, desde: desde, hasta: hasta, tam: tam)
  let o = salida(nombre, tam: tam, ancho: auto)
  if lado {
    align(center + horizon, grid(columns: (ancho-codigo, 1fr), gutter: 0.8cm, align: horizon + left, c, o))
  } else {
    align(center + horizon, stack(spacing: 0.4cm, c, o))
  }
}

// La distorsión J en cada paso de K-means, hasta el paso `hasta`
#let fig-J(hasta: none, size: (7, 4.2)) = {
  let P = d.pasos.pasos.slice(2)
  let n = if hasta == none { P.len() } else { hasta - 1 }
  let pts = P.slice(0, n).enumerate().map(((i, p)) => (i + 1, p.J))
  custom-plot(
    x-min: 0.5, x-max: P.len() + 0.5, y-min: 0, y-max: 420,
    x-label: [paso], y-label: $J$, x-tick-step: 1, y-tick-step: 100, size: size,
    plot.add(pts, style: (stroke: (paint: colores.at(0), thickness: 1.5pt)), mark: "o", mark-size: 0.14, mark-style: (fill: colores.at(0), stroke: none)),
  )
}

// Un paso de K-means: la figura grande y, al lado, qué se hizo y cuánto vale J
#let titulos-paso = (
  datos: [Los datos, sin etiquetas],
  inicio: [Centroides iniciales al azar],
  asignar: [*Asignar*: cada punto va al centroide más cercano],
  mover: [*Mover*: cada centroide va al promedio de su grupo],
)
#let slide-paso(i) = {
  let p = d.pasos.pasos.at(i)
  slide[K-means paso a paso][
    #align(center + horizon, grid(
      columns: (auto, 1fr),
      column-gutter: 1.2cm,
      align: horizon + left,
      ampliar(fig-paso(i), 300%),
      {
        text(size: 22pt, titulos-paso.at(p.tipo))
        if "J" in p {
          v(0.6cm)
          text(size: 24pt)[$J = #num(p.J)$]
        }
      },
    ))
  ]
}

// ---------------------------------------------------------------------------
// Portada

#align(center + horizon)[
  #text(size: 40pt, weight: "bold")[Agrupamiento]
  #v(0.4cm)
  #text(size: 22pt)[Tomas Santana]
  #v(0.2cm)
  #text(size: 18pt, fill: gray.darken(30%))[#datetime.today().display("[month repr:long], [year]")]
]

// ---------------------------------------------------------------------------
// Introducción

#slide[Aprendizaje no supervisado][
  #align(center + horizon, grid(
    columns: 2,
    column-gutter: 2cm,
    align: center + horizon,
    text(size: 26pt)[
      Supervisado: $(x^((i)), y^((i)))$ \
      #v(0.5cm)
      No supervisado: solo $x^((i))$
    ],
    ampliar(fig-elegir-k-datos(), 230%),
  ))
]

// ---------------------------------------------------------------------------
// Parte 1: K-means

#for i in range(d.pasos.pasos.len() - 1) { slide-paso(i) }

#slide[El algoritmo][
  #align(center + horizon)[
    #set text(size: 22pt)
    #set align(left)
    + Inicializar $mu_1, ..., mu_k$ al azar.
    + Repetir hasta que nada cambie:
      #v(0.3cm)
      *Asignar:* #h(0.5cm) $display(c^((i)) := argmin_j norm(x^((i)) - mu_j)^2)$
      #v(0.5cm)
      *Mover:* #h(0.5cm) $display(mu_j := (sum_(i=1)^n 1{c^((i)) = j} x^((i))) / (sum_(i=1)^n 1{c^((i)) = j}))$
  ]
]

#ejercicio(
  "Ejercicio 1",
  text(size: 24pt)[
    $x = 1, 2, 4, 7, 8, 10$ \
    $k = 2$ \
    $mu_1 = 1, quad mu_2 = 2$
  ],
  ancho-figura: 35%,
)[
  + Aplicar K-means hasta que nada cambie. En cada iteración, anotar a qué grupo va cada punto y los nuevos centroides.
  + Calcular $J = sum_i (x^((i)) - mu_(c^((i))))^2$ después de mover los centroides en cada iteración. ¿Qué pasa con $J$?
]

#slide[La distorsión][
  #align(center + horizon, grid(
    columns: 2,
    column-gutter: 1.5cm,
    align: horizon,
    text(size: 26pt)[$ J(c, mu) = sum_(i=1)^n norm(x^((i)) - mu_(c^((i))))^2 $],
    ampliar(fig-J(), 170%),
  ))
]

#lab(1)[
  Programar K-means en `agrupamiento.py`:
  - `assign_clusters`
  - `update_centroids`
  - `distortion`
  y ver cómo baja $J$.
]

#slide[Mínimos locales][
  #grande(fig-minimos(tam: 10pt), s: 200%)
]

#slide[¿Cuántos grupos?][
  #grande(fig-codo(size: (5.0, 3.6), tam: 10pt), s: 200%)
]

#slide[El coeficiente de silueta][
  #align(center + horizon)[
    #text(size: 22pt)[
      $a(i)$: distancia promedio a los ejemplos de su grupo \
      #v(0.2cm)
      $b(i)$: distancia promedio a los ejemplos del grupo más cercano
    ]
    #v(0.8cm)
    #text(size: 30pt)[$ s(i) = (b(i) - a(i)) / max(a(i), b(i)) $]
    #v(0.5cm)
    #text(size: 22pt)[$s(i) approx 1$: bien agrupado #h(1cm) $s(i) approx 0$: en la frontera #h(1cm) $s(i) < 0$: grupo equivocado]
  ]
]

#ejercicio(
  "Ejercicio 2",
  text(size: 24pt)[
    $x = 1, 2, 4, 7, 8, 10$ \
    grupos ${1, 2, 4}$ y ${7, 8, 10}$
  ],
  ancho-figura: 40%,
)[
  + Calcular $a$, $b$ y la silueta $s$ del punto $x = 4$.
  + Hacer lo mismo para el punto $x = 1$.
  + ¿Cuál de los dos puntos está mejor agrupado? ¿Tiene sentido viendo los datos?
]

#slide[K-means en scikit-learn][
  #codigo-salida("kmeans", tam: 12pt, lado: true, ancho-codigo: 80%)
]

#slide[Elegir $k$ en scikit-learn][
  #codigo-salida("elegir_k", tam: 13pt)
]

#slide[Estandarizar las features][
  #grande(fig-escala(tam: 10pt), s: 210%)
]

// Los colores de una foto

#slide[Los colores de una foto][
  #align(center + horizon, image("/notes/figs/09/foto-original.png", height: 92%))
]

#slide[Cada píxel es un punto][
  #grande(fig-nube(agrupado: false), s: 200%)
]

#slide[Cada píxel es un punto][
  #grande(fig-nube(tam: 10pt), s: 185%)
]

#for k in ("2", "4", "16") {
  slide[La foto con $k = #k$ colores][
    #align(center + horizon, stack(
      spacing: 0.4cm,
      image("/notes/figs/09/foto-k" + k + ".png", height: 82%),
      paleta(d.colores.paletas.at(k), lado: 0.9cm),
    ))
  ]
}

#slide[Comprimir la foto en scikit-learn][
  #codigo-salida("colores", tam: 13pt, lado: true, ancho-codigo: 76%)
]

#lab(2)[
  - Elegir $k$ con el codo y la silueta en los datos de vinos, sin estandarizar y estandarizando.
  - Comprimir la foto con distintos valores de $k$.
]

#slide[Limitación: grupos que no son redondos][
  #grande(fig-kmeans-lunas(tam: 10pt), s: 210%)
]

// ---------------------------------------------------------------------------
// Parte 2: DBSCAN

#slide[DBSCAN: núcleo, frontera y ruido][
  #align(center + horizon, grid(
    columns: (auto, 1fr),
    column-gutter: 1cm,
    align: horizon + left,
    ampliar(fig-tipos(tam: 9pt), 200%),
    text(size: 20pt)[
      `eps` $= #str(d.tipos.eps)$ \
      `min_samples` $= 3$ \
      #v(0.5cm)
      *Núcleo:* al menos `min_samples` ejemplos en su vecindario \
      #v(0.2cm)
      *Frontera:* no es núcleo, pero está cerca de uno \
      #v(0.2cm)
      *Ruido:* ninguno de los dos
    ],
  ))
]

#ejercicio(
  "Ejercicio 3",
  text(size: 24pt)[
    $x = 1, 2, 3, 5, 9, 10, 11, 15$ \
    `eps` $= 1.5$ \
    `min_samples` $= 3$
  ],
  ancho-figura: 42%,
)[
  + Escribir el vecindario de cada punto y decir si es núcleo, frontera o ruido.
  + ¿Qué grupos forma DBSCAN?
  + Repetir con `eps` $= 2.5$. ¿Qué cambia?
]

#slide[DBSCAN en las lunas][
  #grande(fig-dbscan-lunas(), s: 210%)
]

#slide[DBSCAN en scikit-learn][
  #codigo-salida("dbscan", tam: 15pt, lado: true, ancho-codigo: 77%)
]

#slide[El efecto de `eps`][
  #grande(fig-eps(tam: 10pt), s: 200%)
]

#slide[Elegir `eps`][
  #grande(fig-kdist(), s: 200%)
]

#slide[Limitación: densidades distintas][
  #grande(fig-densidades(), s: 210%)
]

// ---------------------------------------------------------------------------
// Parte 3: Agrupamiento jerárquico

#slide[Unir los grupos más cercanos][
  #grande(fig-dendrograma(corte: false, tam: 9pt), s: 200%)
]

#slide[Cortar el dendrograma][
  #grande(fig-dendrograma(tam: 9pt), s: 200%)
]

#ejercicio(
  "Ejercicio 4",
  text(size: 24pt)[
    $x = 1, 2, 4, 8, 9$ \
    enlace _single_
  ],
  ancho-figura: 35%,
)[
  + Aplicar el agrupamiento aglomerativo: anotar qué grupos se unen en cada paso y a qué distancia.
  + Dibujar el dendrograma.
  + ¿A qué altura hay que cortar para obtener 2 grupos? ¿Y 3 grupos?
]

#slide[El enlace: _single_ contra Ward][
  #grande(fig-enlace(tam: 10pt), s: 210%)
]

#slide[El enlace en scikit-learn][
  #codigo-salida("jerarquico", tam: 13pt)
]

// ---------------------------------------------------------------------------
// Parte 4: Densidad contra particional

#slide[Los tres métodos][
  #grande(fig-comparacion(size: (3.6, 2.6), tam: 10pt), s: 140%)
]

#slide[Los tres métodos][
  #align(center + horizon, text(size: 20pt, table(
    columns: 4,
    align: left,
    inset: 11pt,
    table.header([], [K-means], [DBSCAN], [Jerárquico]),
    [Hay que dar], [$k$], [`eps` y `min_samples`], [$k$, o dónde cortar],
    [Forma de los grupos], [redondos, de tamaño parecido], [cualquiera], [depende del enlace],
    [Ruido], [no: todo va a un grupo], [sí], [no],
    [Muchos datos], [rápido], [rápido], [lento],
    [Además da], [los centroides], [qué puntos son ruido], [el dendrograma],
  )))
]

#lab(3)[
  Correr K-means, DBSCAN y el jerárquico sobre bolas, lunas y círculos, y comparar los resultados.
]
