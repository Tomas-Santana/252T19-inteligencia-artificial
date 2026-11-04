// Slides con las figuras y los ejercicios de notes/07-redes-neuronales.typ
// Compilar desde la raíz del repositorio:
//   typst compile --root . slides/07-redes-neuronales.typ

#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *

#let fvec(x) = math.bold(math.upright(x))

#set page(paper: "presentation-16-9", margin: (x: 1.6cm, y: 1.2cm))
#set text(lang: "es", size: 20pt)
#set smartquote(quotes: "“”")

#let frame(stroke) = (x, y) => (
  left: 0pt,
  right: 0pt,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)
#set table(stroke: frame(1pt + rgb("21222C")))

// Datos de las figuras
#let d-lunas = json("../notes/figs/07/lunas.json")
#let d-perdida = json("../notes/figs/07/perdida.json")

#let pct(x) = str(calc.round(x * 100, digits: 1)) + "%"

// Estilo de las clases (el mismo de las notas)
#let c-neg = rgb("#1f77b4")
#let c-pos = rgb("#d62728")

#let sigmoide(z) = 1 / (1 + calc.exp(-z))
#let relu(z) = calc.max(0, z)

#let puntos(datos, clase, tam: 0.13) = plot.add(
  datos,
  mark: if clase < 1 { "o" } else { "triangle" },
  mark-size: tam,
  mark-style: (stroke: none, fill: if clase < 1 { c-neg } else { c-pos }),
  style: (stroke: none),
)

#let puntos-clases(d, tam: 0.13) = {
  puntos(d.neg, 0, tam: tam)
  puntos(d.pos, 1, tam: tam)
}

#let segmento(a, b, estilo: (paint: black, thickness: 1.4pt)) = plot.add((a, b), style: (stroke: estilo))

#let curva(pts, color: black, grosor: 1.4pt, punteada: false) = plot.add(
  pts,
  style: (stroke: (paint: color, thickness: grosor, dash: if punteada { "dashed" } else { none })),
)

#let leyenda-clases = {
  set text(size: 16pt)
  grid(
    columns: 2,
    column-gutter: 20pt,
    box[#text(fill: c-neg)[●] clase 0], box[#text(fill: c-pos)[▲] clase 1],
  )
}

#let linea-leyenda(color, etiqueta, punteada: false) = box[
  #box(width: 0.8cm, height: 0pt, baseline: -0.3em, stroke: (
    top: (paint: color, thickness: 1.6pt, dash: if punteada { "dashed" } else { none }),
  ))
  #etiqueta
]

#let leyenda-lineas(..items) = {
  set text(size: 16pt)
  grid(columns: items.pos().len(), column-gutter: 20pt, ..items.pos())
}

// Diagrama de una red (igual que en las notas)
#let red-diagrama(
  capas,
  nombres: none,
  etiquetas: (:),
  pesos: (:),
  radio: 0.4,
  dx: 3,
  dy: 1.5,
  pos-peso: 0.32,
) = cetz.canvas({
  import cetz.draw: *
  let alto = calc.max(..capas)
  let pos(l, i) = (l * dx, (capas.at(l) - 1) * dy / 2 - i * dy)
  for l in range(capas.len() - 1) {
    for i in range(capas.at(l)) {
      for j in range(capas.at(l + 1)) {
        let (a, b) = (pos(l, i), pos(l + 1, j))
        line(a, b, stroke: 0.5pt + gray.darken(20%))
        let k = str(l) + "-" + str(i) + "-" + str(j)
        if k in pesos {
          let t = pos-peso
          let p = (a.at(0) + t * (b.at(0) - a.at(0)), a.at(1) + t * (b.at(1) - a.at(1)))
          content(p, box(fill: white, inset: 1.5pt, text(size: 8.5pt, pesos.at(k))))
        }
      }
    }
  }
  for l in range(capas.len()) {
    for i in range(capas.at(l)) {
      let color = if l == 0 { gray.darken(30%) } else if l == capas.len() - 1 { c-pos } else { c-neg }
      circle(pos(l, i), radius: radio, fill: white, stroke: 0.9pt + color)
      let k = str(l) + "-" + str(i)
      if k in etiquetas { content(pos(l, i), text(size: 9pt, etiquetas.at(k))) }
    }
  }
  if nombres != none {
    for l in range(capas.len()) {
      content((l * dx, -(alto - 1) * dy / 2 - 0.85), text(size: 8.5pt, fill: gray.darken(40%), nombres.at(l)))
    }
  }
})

#let panel-xor(cuerpo, tam: 7.2, x-label: $x_1$, y-label: $x_2$, x-min: -0.5, x-max: 1.5, y-min: -0.5, y-max: 1.5) = custom-plot(
  x-min: x-min,
  x-max: x-max,
  y-min: y-min,
  y-max: y-max,
  x-label: x-label,
  y-label: y-label,
  x-tick-step: 0.5,
  y-tick-step: 0.5,
  size: (tam, tam),
  cuerpo,
)

#let xor-neg = ((0, 0), (1, 1))
#let xor-pos = ((0, 1), (1, 0))

// Las figuras se dibujan con el tamaño de las notas y se amplían en la diapositiva
#let grande(f, s: 230%) = align(center + horizon, scale(s, reflow: true, f))

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

// ---------------------------------------------------------------------------
// Portada

#align(center + horizon)[
  #text(size: 40pt, weight: "bold")[Redes neuronales]
  #v(0.4cm)
  #text(size: 22pt)[Tomas Santana]
  #v(0.2cm)
  #text(size: 18pt, fill: gray.darken(30%))[#datetime.today().display("[month repr:long], [year]")]
]

// ---------------------------------------------------------------------------
// Parte 1: Una neurona

#slide[Una neurona][
  #grande(red-diagrama(
    (3, 1),
    etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "0-2": $x_3$, "1-0": $a$),
    pesos: ("0-0-0": $w_1$, "0-1-0": $w_2$, "0-2-0": $w_3$),
    pos-peso: 0.5,
    nombres: ([entrada], [$a = g(fvec(w)^T fvec(x) + b)$]),
    dx: 4,
  ))
]

#slide[Funciones de activación][
  #align(center + horizon, grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [Sigmoide $g(z)$ #v(0.1cm) #scale(100%, reflow: true, custom-plot(
        x-min: -5,
        x-max: 5,
        y-min: -0.2,
        y-max: 1.2,
        x-label: $z$,
        x-tick-step: 2,
        y-tick-step: 0.5,
        size: (9, 6.4),
        plot.add(domain: (-5, 5), samples: 100, sigmoide, style: (stroke: (paint: c-neg, thickness: 1.8pt))),
      ))],
    [ReLU $max(z, 0)$ #v(0.1cm) #scale(100%, reflow: true, custom-plot(
        x-min: -3,
        x-max: 3,
        y-min: -0.5,
        y-max: 3,
        x-label: $z$,
        x-tick-step: 1,
        y-tick-step: 1,
        size: (9, 6.4),
        plot.add(domain: (-3, 3), samples: 61, relu, style: (stroke: (paint: c-pos, thickness: 1.8pt))),
      ))],
  ))
]

#slide[El límite de una neurona: XOR][
  #align(center + horizon, grid(
    columns: (1fr, 1fr),
    align: horizon + center,
    gutter: 1cm,
    table(
      columns: 3,
      align: center,
      inset: 10pt,
      table.header($x_1$, $x_2$, $y$),
      $0$, $0$, $0$,
      $0$, $1$, $1$,
      $1$, $0$, $1$,
      $1$, $1$, $0$,
    ),
    [
      #scale(100%, reflow: true, panel-xor({
        puntos(xor-neg, 0, tam: 0.22)
        puntos(xor-pos, 1, tam: 0.22)
      }))
      #leyenda-clases
    ],
  ))
]

// ---------------------------------------------------------------------------
// Parte 2: Redes neuronales

#let red-xor = red-diagrama(
  (2, 2, 1),
  etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "1-0": $0$, "1-1": $-1$, "2-0": $0$),
  pesos: ("0-0-0": $1$, "0-0-1": $1$, "0-1-0": $1$, "0-1-1": $1$, "1-0-0": $1$, "1-1-0": $-2$),
  nombres: ([entrada], [oculta (ReLU)], [salida]),
  dx: 4,
  pos-peso: 0.4,
)

#let red-xor-compacta = red-diagrama(
  (2, 2, 1),
  etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "1-0": $0$, "1-1": $-1$, "2-0": $0$),
  pesos: ("0-0-0": $1$, "0-0-1": $1$, "0-1-0": $1$, "0-1-1": $1$, "1-0-0": $1$, "1-1-0": $-2$),
  nombres: ([entrada], [oculta (ReLU)], [salida]),
  dx: 3,
  pos-peso: 0.4,
)

#slide[Una red que resuelve XOR][
  #grande(red-xor)
]

#ejercicio("Ejercicio 1", scale(140%, reflow: true, red-xor-compacta))[
  Calcula la predicción de la red para los cuatro ejemplos de XOR y completa la tabla. ¿La red acierta en todos?

  #set text(size: 17pt)
  #align(center, table(
    columns: 6,
    align: center + horizon,
    rows: (auto, 0.9cm),
    table.header($x_1$, $x_2$, $x_1 + x_2$, $a_1$, $a_2$, $hat(y)$),
    $0$, $0$, [], [], [], [],
    $0$, $1$, [], [], [], [],
    $1$, $0$, [], [], [], [],
    $1$, $1$, [], [], [], [],
  ))
]

#ejercicio(
  "Ejercicio 2",
  scale(140%, reflow: true, red-diagrama(
    (2, 2, 1),
    etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "1-0": $0$, "1-1": $-1$, "2-0": $0$),
    pesos: ("0-0-0": $1$, "0-0-1": $1$, "0-1-0": $1$, "0-1-1": $1$, "1-0-0": $1$, "1-1-0": $-2$),
    nombres: ([entrada], [oculta (sin ReLU)], [salida]),
    dx: 3,
    pos-peso: 0.4,
  )),
  ancho-figura: 40%,
)[
  Repite el Ejercicio 1 *sin la ReLU*:
  $
    a_1 & = x_1 + x_2 \
    a_2 & = x_1 + x_2 - 1 \
    hat(y) & = a_1 - 2 a_2
  $
  + Calcula $hat(y)$ para los cuatro ejemplos. ¿Sigue resolviendo XOR?
  + Sustituye $a_1$ y $a_2$ en la fórmula de $hat(y)$ y simplifica. ¿Qué tipo de función de $x_1$ y $x_2$ queda?
]

#slide[La capa oculta transforma los datos][
  #align(center + horizon, grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [Espacio de las entradas #v(0.1cm) #scale(100%, reflow: true, panel-xor({
        plot.add(
          ((-0.5, 1), (1, -0.5), (1.5, -0.5), (1.5, 0), (0, 1.5), (-0.5, 1.5), (-0.5, 1)),
          fill: true,
          fill-type: "shape",
          style: (stroke: none, fill: c-pos.transparentize(82%)),
        )
        segmento((-0.5, 1), (1, -0.5), estilo: (paint: black, thickness: 1pt, dash: "dashed"))
        segmento((0, 1.5), (1.5, 0), estilo: (paint: black, thickness: 1pt, dash: "dashed"))
        puntos(xor-neg, 0, tam: 0.22)
        puntos(xor-pos, 1, tam: 0.22)
      }))],
    [Espacio de las unidades ocultas #v(0.1cm) #scale(100%, reflow: true, panel-xor(
        x-label: $a_1$,
        y-label: $a_2$,
        x-min: -0.5,
        x-max: 2.5,
        y-min: -0.5,
        y-max: 2.5,
        {
          segmento((0, -0.25), (2.5, 1), estilo: (paint: black, thickness: 1.2pt))
          puntos(((0, 0), (2, 1)), 0, tam: 0.22)
          puntos(((1, 0),), 1, tam: 0.22)
          plot.annotate({
            import cetz.draw: *
            content((1.75, -0.2), text(size: 13pt)[$(0,1)$ y $(1,0)$])
            content((0.1, 0.35), text(size: 13pt)[$(0,0)$])
            content((2, 1.35), text(size: 13pt)[$(1,1)$])
          })
        },
      ))],
  ))
]

#slide[Redes totalmente conectadas][
  #grande(red-diagrama(
    (2, 2, 1),
    etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "1-0": $a_1$, "1-1": $a_2$, "2-0": $hat(y)$),
    nombres: ([entrada], [capa oculta], [salida]),
  ))
]

#let panel-lunas(p, y-label: none) = custom-plot(
  x-min: d-lunas.limites.at(0),
  x-max: d-lunas.limites.at(1),
  y-min: d-lunas.limites.at(2),
  y-max: d-lunas.limites.at(3),
  x-label: $x_1$,
  y-label: y-label,
  x-tick-step: 1,
  y-tick-step: 1,
  size: (6.6, 5.4),
  {
    puntos-clases(d-lunas, tam: 0.07)
    for s in p.frontera { curva(s, grosor: 1.4pt) }
  },
)

#slide[La red aprende las features][
  #align(center + horizon)[
    #set text(size: 18pt)
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 0.4cm,
      [Regresión logística #scale(100%, reflow: true, panel-lunas(d-lunas.paneles.logistica, y-label: $x_2$))
        exactitud #pct(d-lunas.paneles.logistica.exactitud)],
      [3 unidades ocultas #scale(100%, reflow: true, panel-lunas(d-lunas.paneles.h3))
        exactitud #pct(d-lunas.paneles.h3.exactitud)],
      [32 unidades ocultas #scale(100%, reflow: true, panel-lunas(d-lunas.paneles.h32))
        exactitud #pct(d-lunas.paneles.h32.exactitud)],
    )
    #v(0.3cm)
    #leyenda-clases
  ]
]

// ---------------------------------------------------------------------------
// Parte 3: Cómo aprende una red

#ejercicio(
  "Ejercicio 3",
  [
    #scale(200%, reflow: true, red-diagrama(
      (1, 1),
      etiquetas: ("0-0": $x$, "1-0": $hat(y)$),
      pesos: ("0-0-0": $w$),
      pos-peso: 0.5,
      nombres: ([$x = 2$], [$hat(y) = w x$]),
      dx: 4,
    ))
    #v(0.4cm)
    #align(center, text(size: 18pt)[$y = 10, quad L = (hat(y) - y)^2$ \ $(d L) / (d w) = 2(hat(y) - y) x$])
  ],
  ancho-figura: 34%,
)[
  Empezando con $w = 1$ y $alpha = 0.05$, completa tres pasos de descenso por gradiente.

  #set text(size: 16pt)
  #align(center, table(
    columns: (auto, 1.8cm, 2.4cm, 3.4cm, 4.4cm),
    align: center + horizon,
    rows: (auto, 0.85cm),
    table.header([Paso], $w$, $hat(y) = 2 w$, $L = (hat(y) - 10)^2$, $(d L) / (d w) = 4(hat(y) - 10)$),
    $0$, $1$, [], [], [],
    $1$, [], [], [], [],
    $2$, [], [], [], [],
    $3$, [], [], [], [],
  ))

  #set text(size: 20pt)
  ¿Hacia qué valor se acerca $w$? ¿Por qué?
]

#let pasos-w(alfa, n) = {
  let w = 1.0
  let res = ()
  for t in range(n + 1) {
    res.push((t, w))
    w = w - alfa * 4 * (2 * w - 10)
  }
  res
}

#slide[Descenso por gradiente: la tasa de aprendizaje][
  #align(center + horizon)[
    #scale(100%, reflow: true, custom-plot(
      x-min: 0,
      x-max: 10,
      y-min: -4,
      y-max: 17,
      x-label: [paso],
      y-label: $w$,
      x-tick-step: 2,
      y-tick-step: 5,
      size: (17, 7.4),
      {
        plot.add-hline(5, style: (stroke: (paint: gray, thickness: 0.6pt, dash: "dotted")))
        plot.add(pasos-w(0.05, 10), mark: "o", mark-size: 0.1, style: (stroke: (paint: c-neg, thickness: 1.4pt)), mark-style: (fill: c-neg, stroke: none))
        plot.add(pasos-w(0.3, 3), mark: "o", mark-size: 0.1, style: (stroke: (paint: c-pos, thickness: 1.4pt, dash: "dashed")), mark-style: (fill: c-pos, stroke: none))
      },
    ))
    #v(0.3cm)
    #leyenda-lineas(linea-leyenda(c-neg, $alpha = 0.05$), linea-leyenda(c-pos, $alpha = 0.3$, punteada: true))
  ]
]

#ejercicio(
  "Ejercicio 4",
  [
    #scale(190%, reflow: true, red-diagrama(
      (1, 1, 1),
      etiquetas: ("0-0": $x$, "1-0": $a$, "2-0": $hat(y)$),
      pesos: ("0-0-0": $w_1$, "1-0-0": $w_2$),
      pos-peso: 0.5,
      nombres: ([entrada], [ReLU], [salida]),
      dx: 3,
    ))
    #v(0.4cm)
    #align(center, text(size: 18pt)[
      $a = "ReLU"(w_1 x), quad hat(y) = w_2 a$ \
      $L = (hat(y) - y)^2$
    ])
  ],
)[
  Con $x = 2$ y $y = 10$, calcula el forward pass y las dos derivadas $(partial L) / (partial w_1)$ y $(partial L) / (partial w_2)$ si empezamos con $w_1 = -1$ y $w_2 = 1$.

  ¿Qué pasa con $w_1$ si damos un paso de descenso por gradiente? ¿Por qué?
]

#slide[Entrenar una red en PyTorch][
  #align(center + horizon, block(
    fill: luma(245),
    stroke: 0.6pt + luma(200),
    inset: 16pt,
    radius: 4pt,
    align(left, text(size: 17pt)[
      ```python
      modelo = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
      perdida = nn.BCEWithLogitsLoss()
      optimizador = torch.optim.SGD(modelo.parameters(), lr=0.1)

      for paso in range(1000):
          optimizador.zero_grad()   # 1. borrar los gradientes anteriores
          y_hat = modelo(X)         # 2. forward pass
          L = perdida(y_hat, y)     # 3. calcular la pérdida
          L.backward()              # 4. backpropagation
          optimizador.step()        # 5. un paso de descenso
      ```
    ]),
  ))
]

#slide[La pérdida durante el entrenamiento][
  #align(center + horizon)[
    #scale(100%, reflow: true, custom-plot(
      x-min: 0,
      x-max: 1000,
      y-min: 0,
      y-max: 0.8,
      x-label: [paso],
      y-label: [pérdida],
      x-tick-step: 200,
      y-tick-step: 0.2,
      size: (17, 7.6),
      curva(d-perdida.perdida, color: c-pos, grosor: 1.5pt),
    ))
    #v(0.3cm)
    #text(size: 18pt)[32 unidades ocultas, $alpha = 0.1$. Exactitud en prueba: #pct(d-perdida.exactitud)]
  ]
]
