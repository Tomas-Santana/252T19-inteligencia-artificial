// Slides con las figuras, el código y los ejercicios de notes/08-redes-convolucionales.typ
// Las figuras son las mismas de las notas (notes/08-figuras.typ).
// Compilar desde la raíz del repositorio:
//   typst compile --root . slides/08-redes-convolucionales.typ

#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../notes/08-figuras.typ": *

#let fvec(x) = math.bold(math.upright(x))

#set page(paper: "presentation-16-9", margin: (x: 1.6cm, y: 1.2cm))
#set text(lang: "es", size: 20pt)
#set smartquote(quotes: "“”")
#show raw: set text(font: "DejaVu Sans Mono")

#let frame(stroke) = (x, y) => (
  left: 0pt,
  right: 0pt,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)
#set table(stroke: frame(1pt + rgb("21222C")))

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
#let codigo-salida(
  nombre,
  desde: none,
  hasta: none,
  tam: 15pt,
  salida-de: auto,
  lineas: none,
  lado: false,
  ancho-codigo: 60%,
) = {
  let s = if salida-de == auto { nombre } else { salida-de }
  let c = bloque-codigo(nombre, desde: desde, hasta: hasta, tam: tam)
  let o = salida(s, tam: tam, ancho: auto, lineas: lineas)
  if lado {
    align(center + horizon, grid(columns: (ancho-codigo, 1fr), gutter: 0.8cm, align: horizon + left, c, o))
  } else {
    align(center + horizon, stack(spacing: 0.4cm, c, o))
  }
}

#let r-mlp = d-mnist.resultados.mlp
#let r-cnn = d-mnist.resultados.cnn

// ---------------------------------------------------------------------------
// Portada

#align(center + horizon)[
  #text(size: 40pt, weight: "bold")[Redes convolucionales]
  #v(0.4cm)
  #text(size: 22pt)[Tomas Santana]
  #v(0.2cm)
  #text(size: 18pt, fill: gray.darken(30%))[#datetime.today().display("[month repr:long], [year]")]
]

// ---------------------------------------------------------------------------
// Parte 1: Repaso de backpropagation

#slide[Repaso: backpropagation][
  #grande(fig-grafo(tam: 9pt), s: 170%)
]

#ejercicio(
  "Ejercicio 1",
  text(size: 22pt)[
    $z = w_1 x$ \
    $a = "ReLU"(z)$ \
    $hat(y) = w_2 a$ \
    $L = (hat(y) - y)^2$
  ],
  ancho-figura: 30%,
)[
  Con $x = 1$, $y = 4$, $w_1 = 2$ y $w_2 = 3$:
  + Calcular el forward pass: $z$, $a$, $hat(y)$ y $L$.
  + Calcular, de derecha a izquierda, las derivadas de $L$ respecto a $hat(y)$, $w_2$, $a$, $z$ y $w_1$.
  + Dar un paso de descenso por gradiente con $alpha = 0.01$. ¿Los pesos suben o bajan? ¿Por qué?
]

#slide[`backward()` en PyTorch][
  #codigo-salida("backward", desde: 3, tam: 17pt, lado: true, ancho-codigo: 66%)
]

// ---------------------------------------------------------------------------
// Parte 2: Redes como clases de PyTorch

#slide[Una red como clase: `nn.Module`][
  #align(center + horizon, bloque-codigo("mlp", desde: 5, hasta: 14, tam: 17pt))
]

#slide[Usar la red][
  #codigo-salida("mlp", desde: 17, hasta: 21, tam: 17pt)
]

#slide[Entrenar por mini-batches][
  #align(center + horizon, bloque-codigo("entrenamiento", desde: 13, tam: 15pt))
]

#ejercicio(
  "Ejercicio 2",
  text(size: 22pt)[
    50 000 ejemplos \
    mini-batches de 64 \
    3 épocas
  ],
  ancho-figura: 30%,
)[
  + ¿Cuántos pasos da el entrenamiento?
  + ¿Y si los mini-batches son de 128?
  + ¿Cuántas veces ve la red cada ejemplo en los dos casos?
]

#slide[Evaluar el modelo][
  #align(center + horizon, stack(
    spacing: 0.5cm,
    bloque-codigo("entrenamiento", desde: 6, hasta: 10, tam: 16pt),
    salida("entrenar-mlp", tam: 16pt, titulo: [`MyMLP` en MNIST], ancho: auto),
  ))
]

// ---------------------------------------------------------------------------
// Parte 3: Redes convolucionales

#slide[El problema de las imágenes][
  #grande(fig-desplazado(ancho: 3.2cm, tam: 16pt), s: 120%)
]

#for (paso, pos) in ((1, (0, 0)), (2, (0, 1)), (3, (0, 2)), (9, (2, 2))) {
  slide[La convolución][
    #align(center + horizon, stack(
      spacing: 0.9cm,
      scale(190%, reflow: true, fig-conv(X-conv, K-conv, pos: pos, visibles: paso)),
      text(size: 20pt, calculo-conv(X-conv, K-conv, pos)),
    ))
  ]
}

#slide[La convolución en PyTorch][
  #codigo-salida("convolucion", desde: 4, tam: 12pt, lado: true, ancho-codigo: 66%)
]

#ejercicio(
  "Ejercicio 3",
  text(size: 22pt)[
    $X = mat(1, 2, 0, 1; 0, 1, 3, 1; 2, 0, 1, 0; 1, 1, 0, 2)$ \
    #v(0.3cm)
    $K = mat(1, 0; 0, -1)$
  ],
  ancho-figura: 38%,
)[
  + ¿De qué tamaño es la salida de $X star K$? Calcular sus entradas.
  + Aplicar la ReLU a la salida.
  + Si $X$ es un tensor de forma $(1, 1, 4, 4)$, ¿qué forma tiene la salida de `nn.Conv2d(1, 1, kernel_size=2)`?
]

#slide[Qué detecta un filtro][
  #grande(fig-bordes(ancho: 4.2cm, tam: 18pt), s: 100%)
]

#slide[Varios filtros: canales][
  #grande(fig-canales(1, 28, 8, 3, 26, tam: 9pt), s: 210%)
]

#slide[Varios filtros en PyTorch][
  #codigo-salida("canales", desde: 4, tam: 17pt)
]

#slide[Entradas con varios canales][
  #grande(fig-canales(8, 14, 16, 3, 12, tam: 9pt), s: 230%)
]

#slide[Entradas con varios canales en PyTorch][
  #codigo-salida("canales2", desde: 4, tam: 17pt)
]

#slide[Número de parámetros][
  #align(center + horizon)[
    #text(size: 30pt)[$(C_"in" k^2 + 1) C_"out"$]
    #v(1.2cm)
    #text(size: 22pt)[
      `Conv2d(1, 8, 3)`: #h(0.5cm) $(1 dot 9 + 1) dot 8 = 80$ \
      #v(0.2cm)
      `Conv2d(8, 16, 3)`: #h(0.5cm) $(8 dot 9 + 1) dot 16 = 1168$
    ]
  ]
]

#slide[Padding][
  #grande(fig-conv(X-conv, K-conv, pos: (0, 0), padding: 1, s: 0.5, tam: 9.5pt), s: 175%)
]

#slide[Tamaño de la salida][
  #align(center + horizon, stack(
    spacing: 0.8cm,
    text(size: 28pt)[$m = floor((n + 2 p - k) / s) + 1$],
    codigo-salida("padding", desde: 4, tam: 16pt),
  ))
]

#slide[Max pooling][
  #grande(fig-pooling(X-pool), s: 200%)
]

#slide[Max pooling en PyTorch][
  #codigo-salida("pooling", desde: 4, tam: 16pt)
]

#ejercicio(
  "Ejercicio 4",
  text(size: 22pt)[$Z = mat(2, 0, 1, 5; 3, 1, 0, 2; 0, 4, 2, 2; 1, 1, 3, 0)$],
  ancho-figura: 38%,
)[
  + Aplicar max pooling de $2 times 2$ a $Z$.
  + Un tensor de forma $(64, 16, 14, 14)$ pasa por `nn.MaxPool2d(2)`. ¿Qué forma tiene la salida? ¿Cuántos parámetros tiene la capa?
]

#slide[Una red convolucional completa][
  #grande(fig-arquitectura(tam: 8pt), s: 140%)
]

#slide[`MyCNN`][
  #align(center + horizon, bloque-codigo("cnn", desde: 5, tam: 16pt))
]

#slide[Un forward de prueba: las formas][
  #codigo-salida("formas", desde: 4, hasta: 18, lineas: (1, 7), tam: 14pt, lado: true, ancho-codigo: 60%)
]

#slide[Un forward de prueba: los parámetros][
  #codigo-salida("formas", desde: 20, lineas: (8, 13), tam: 17pt)
]

#ejercicio(
  "Ejercicio 5",
  text(size: 18pt)[
    entrada $1 times 28 times 28$ \
    `Conv2d(1, 6, kernel_size=5)` \
    ReLU, `MaxPool2d(2)` \
    `Conv2d(6, 16, kernel_size=5)` \
    ReLU, `MaxPool2d(2)` \
    aplanar \
    `Linear(?, 10)`
  ],
  ancho-figura: 45%,
)[
  + Calcular la forma de la salida de cada capa.
  + ¿Cuántas entradas tiene la capa densa?
  + ¿Cuántos parámetros tiene cada capa? ¿Y la red completa?
]

#slide[MLP contra CNN en MNIST][
  #align(center + horizon, text(size: 22pt, table(
    columns: 4,
    align: (left, right, center, center),
    inset: 12pt,
    table.header([Red], [Parámetros], [Validación], [Prueba]),
    [`MyMLP`], [#miles(r-mlp.parametros)], [#pct(r-mlp.validacion)], [#pct(r-mlp.prueba)],
    [`MyCNN`], [#miles(r-cnn.parametros)], [#pct(r-cnn.validacion)], [#pct(r-cnn.prueba)],
  )))
]

#slide[Lo que aprendió la primera capa][
  #grande(fig-filtros(ancho-filtro: 2.3cm, ancho-mapa: 2.3cm, tam: 14pt, gutter: 12pt), s: 100%)
]

// ---------------------------------------------------------------------------
// Parte 4: Sobreajuste y regularización

#slide[Sobreajuste][
  #align(center + horizon)[
    #fig-sobreajuste(size: (17, 7.6))
    #v(0.3cm)
    #leyenda-sobreajuste(tam: 16pt)
  ]
]

#slide[Dropout: `train()` y `eval()`][
  #codigo-salida("dropout", desde: 4, tam: 17pt)
]

#slide[Weight decay y early stopping][
  #align(center + horizon, block(
    fill: luma(245),
    stroke: 0.6pt + luma(200),
    inset: 14pt,
    radius: 4pt,
    align(left, {
      show raw: set text(size: 18pt)
      ```python
      # weight decay: regularización L2, como ridge
      optimizador = torch.optim.SGD(modelo.parameters(), lr=0.1,
                                    weight_decay=1e-4)

      # early stopping: guardar los pesos de la mejor época
      if perdida_val < mejor_perdida:
          mejor_perdida = perdida_val
          torch.save(modelo.state_dict(), "mejor.pt")
      ```
    }),
  ))
]
