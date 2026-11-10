// Figuras de la clase 8, compartidas por las notas (notes/08-redes-convolucionales.typ)
// y las diapositivas (slides/08-redes-convolucionales.typ).
// Compilar siempre con --root . desde la raíz del repositorio.
//
// Los datos vienen de `python notes/scripts/08-redes-convolucionales-figuras.py`.

#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *

#let d-mnist = json("/notes/figs/08/mnist.json")
#let d-salidas = json("/notes/figs/08/salidas.json")

#let c-neg = rgb("#1f77b4")
#let c-pos = rgb("#d62728")
#let c-3 = rgb("#2ca02c")
#let c-4 = rgb("#ff7f0e")

// Reduce una figura si es más ancha que el espacio disponible
#let ajustar(cuerpo) = layout(espacio => {
  let m = measure(cuerpo)
  if m.width > espacio.width {
    scale(espacio.width / m.width * 100%, reflow: true, cuerpo)
  } else { cuerpo }
})

#let pct(x) = {
  let t = str(calc.round(x * 100, digits: 1))
  if not t.contains(".") { t += ".0" }
  t + "%"
}
#let miles(n) = {
  let s = str(n)
  let partes = ()
  while s.len() > 3 {
    partes.insert(0, s.slice(s.len() - 3))
    s = s.slice(0, s.len() - 3)
  }
  partes.insert(0, s)
  partes.join("\u{202F}")
}

// ---------------------------------------------------------------------------
// Código y salidas
// ---------------------------------------------------------------------------

// Texto de un fragmento de notes/code/08, opcionalmente solo las líneas [desde, hasta] (desde 1)
#let fuente(nombre, desde: none, hasta: none) = {
  let lineas = read("/notes/code/08/" + nombre + ".py").trim(at: end).split("\n")
  let a = if desde == none { 0 } else { desde - 1 }
  let b = if hasta == none { lineas.len() } else { hasta }
  lineas.slice(a, b).join("\n")
}

#let codigo(nombre, desde: none, hasta: none) = block(breakable: false, raw(
  fuente(nombre, desde: desde, hasta: hasta),
  lang: "python",
  block: true,
))

// La salida real de un fragmento (guardada al ejecutarlo)
#let salida(nombre, tam: 8.5pt, titulo: [Salida], ancho: 100%, lineas: none) = block(
  width: ancho,
  fill: luma(245),
  stroke: (left: 2pt + luma(170)),
  inset: (x: 10pt, y: 7pt),
  breakable: false,
  align(left, {
    text(size: tam * 0.9, fill: luma(110), weight: "bold", titulo)
    linebreak()
    set text(font: "DejaVu Sans Mono", size: tam)
    let todas = d-salidas.at(nombre).split("\n")
    let lineas = if lineas == none { todas } else { todas.slice(lineas.at(0) - 1, lineas.at(1)) }
    for (i, l) in lineas.enumerate() {
      l.replace(" ", "\u{00A0}")
      if i < lineas.len() - 1 { linebreak() }
    }
  }),
)

// ---------------------------------------------------------------------------
// Imágenes
// ---------------------------------------------------------------------------

// Una matriz de píxeles 0-255 (alto = oscuro al imprimir: se invierte)
#let imagen(px, ancho: 1cm, pixelado: false, borde: none) = {
  // Algunos visores ignoran `scaling: "pixelated"`, así que agrandamos la
  // imagen repitiendo cada píxel (vecino más cercano) para evitar el suavizado.
  if pixelado {
    let k = calc.max(1, calc.ceil(64 / px.at(0).len()))
    px = px.map(fila => {
      let f = fila.map(p => range(k).map(_ => p)).flatten()
      range(k).map(_ => f)
    }).flatten().chunks(px.at(0).len() * k)
  }
  let h = px.len()
  let w = px.at(0).len()
  box(stroke: borde, image(
    bytes(px.flatten().map(p => 255 - p)),
    format: (encoding: "luma8", width: w, height: h),
    width: ancho,
    scaling: if pixelado { "pixelated" } else { "smooth" },
  ))
}

#let digito(px, ancho: 1.6cm) = imagen(px, ancho: ancho)

// ---------------------------------------------------------------------------
// Matrices de números y la convolución
// ---------------------------------------------------------------------------

#let fmt(v) = if v < 0 { "−" + str(-v) } else { str(v) }

#let conv2d(X, K) = {
  let n = X.len()
  let m = X.at(0).len()
  let k = K.len()
  range(n - k + 1).map(i => range(m - k + 1).map(j => {
    let s = 0
    for u in range(k) { for v in range(k) { s += X.at(i + u).at(j + v) * K.at(u).at(v) } }
    s
  }))
}

#let rellenar(X, p) = {
  let m = X.at(0).len()
  let ceros = range(m + 2 * p).map(_ => 0)
  range(p).map(_ => ceros) + X.map(f => range(p).map(_ => 0) + f + range(p).map(_ => 0)) + range(p).map(_ => ceros)
}

// Dibuja una matriz con su esquina superior izquierda en (x0, y0).
// `gris`: número de filas/columnas del borde que son padding.
// `visibles`: cuántas celdas mostrar (en orden de lectura); el resto quedan vacías.
#let matriz(M, x0, y0, s, gris: 0, visibles: none, colores: none, tam: 11pt) = {
  import cetz.draw: *
  let n = M.len()
  let m = M.at(0).len()
  for i in range(n) {
    for j in range(m) {
      let es-pad = i < gris or j < gris or i >= n - gris or j >= m - gris
      let fondo = if colores != none { colores.at(i).at(j) } else if es-pad { luma(235) } else { white }
      rect((x0 + j * s, y0 - i * s), (x0 + (j + 1) * s, y0 - (i + 1) * s), fill: fondo, stroke: 0.5pt + luma(150))
      let visible = visibles == none or i * m + j < visibles
      if visible {
        content(
          (x0 + (j + 0.5) * s, y0 - (i + 0.5) * s),
          text(size: tam, fill: if es-pad { luma(140) } else { black }, fmt(M.at(i).at(j))),
        )
      }
    }
  }
}

#let ventana(x0, y0, s, i, j, k, color) = {
  import cetz.draw: *
  rect(
    (x0 + j * s, y0 - i * s),
    (x0 + (j + k) * s, y0 - (i + k) * s),
    fill: color.transparentize(80%),
    stroke: 1.8pt + color,
  )
}

// Figura de la convolución: entrada ⋆ filtro = salida.
// `pos`: la posición (fila, columna) de la ventana que se resalta, o none.
// `visibles`: cuántas celdas de la salida mostrar (para recorrerla paso a paso).
#let fig-conv(X, K, pos: (0, 0), padding: 0, visibles: none, s: 0.62, tam: 11pt, etiquetas: true) = {
  let Xp = rellenar(X, padding)
  let Y = conv2d(Xp, K)
  let n = Xp.len()
  let k = K.len()
  let no = Y.len()
  cetz.canvas({
    import cetz.draw: *
    let alto = n * s
    let xk = n * s + 1.2
    let yk = -(alto - k * s) / 2
    let xy = xk + k * s + 1.2
    let yy = -(alto - no * s) / 2
    matriz(Xp, 0, 0, s, gris: padding, tam: tam)
    matriz(K, xk, yk, s, tam: tam)
    matriz(Y, xy, yy, s, visibles: visibles, tam: tam)
    if pos != none {
      let (i, j) = pos
      ventana(0, 0, s, i, j, k, c-neg)
      ventana(xy, yy, s, i, j, 1, c-neg)
    }
    content((n * s + 0.6, -alto / 2), text(size: tam * 1.6)[$star$])
    content((xk + k * s + 0.6, -alto / 2), text(size: tam * 1.6)[$=$])
    if etiquetas {
      let t = tam * 0.85
      let p = if padding > 0 { [ (con padding)] } else { [] }
      content((n * s / 2, -alto - 0.45), text(size: t)[entrada, $#str(n) times #str(n)$#p])
      content((xk + k * s / 2, yk - k * s - 0.45), text(size: t)[filtro, $#str(k) times #str(k)$])
      content((xy + no * s / 2, yy - no * s - 0.45), text(size: t)[salida, $#str(no) times #str(no)$])
    }
  })
}

// El cálculo de una celda de la salida, como suma de productos
#let calculo-conv(X, K, pos) = {
  let (i, j) = pos
  let k = K.len()
  let filas = ()
  let total = 0
  for u in range(k) {
    let terminos = ()
    for v in range(k) {
      let x = X.at(i + u).at(j + v)
      let w = K.at(u).at(v)
      total += x * w
      let w-txt = if w < 0 { "(" + fmt(w) + ")" } else { fmt(w) }
      terminos.push(fmt(x) + " dot " + w-txt)
    }
    filas.push(terminos.join(" + "))
  }
  eval("$" + filas.join(" + ") + " = " + fmt(total) + "$")
}

// Max pooling de 2 × 2 con colores por región
#let fig-pooling(X, s: 0.7, tam: 12pt) = {
  let tonos = (c-neg, c-pos, c-3, c-4).map(c => c.transparentize(75%))
  let n = X.len()
  let col-X = range(n).map(i => range(n).map(j => tonos.at(2 * calc.floor(i / 2) + calc.floor(j / 2))))
  let m = calc.quo(n, 2)
  let Y = range(m).map(i => range(m).map(j => calc.max(
    X.at(2 * i).at(2 * j),
    X.at(2 * i).at(2 * j + 1),
    X.at(2 * i + 1).at(2 * j),
    X.at(2 * i + 1).at(2 * j + 1),
  )))
  let col-Y = range(m).map(i => range(m).map(j => tonos.at(2 * i + j)))
  cetz.canvas({
    import cetz.draw: *
    matriz(X, 0, 0, s, colores: col-X, tam: tam)
    let xy = n * s + 2.6
    let yy = -(n * s - m * s) / 2
    line((n * s + 0.3, -n * s / 2), (xy - 0.3, -n * s / 2), stroke: 1pt, mark: (end: ">", fill: black))
    content(((n * s + xy) / 2, -n * s / 2 + 0.35), text(size: tam * 0.85)[max $2 times 2$])
    matriz(Y, xy, yy, s, colores: col-Y, tam: tam)
    content((n * s / 2, -n * s - 0.45), text(size: tam * 0.85)[entrada, $#str(n) times #str(n)$])
    content((xy + m * s / 2, yy - m * s - 0.45), text(size: tam * 0.85)[salida, $#str(m) times #str(m)$])
  })
}

// ---------------------------------------------------------------------------
// Tensores como bloques
// ---------------------------------------------------------------------------

// Un tensor C × H × W dibujado como una pila de C hojas (como máximo 6).
// El punto `p` es la esquina inferior izquierda de la hoja del frente.
#let tensor-bloque(p, C, H, W, esc: 0.08, color: c-neg, etiqueta: auto, tam: 9pt, max-hojas: 6) = {
  import cetz.draw: *
  let (x, y) = p
  let hojas = calc.min(C, max-hojas)
  let d = 0.13
  let w = W * esc
  let h = H * esc
  for t in range(hojas - 1, -1, step: -1) {
    rect(
      (x + t * d, y + t * d),
      (x + t * d + w, y + t * d + h),
      fill: color.lighten(80% - 8% * calc.min(t, 5)),
      stroke: 0.6pt + color.darken(10%),
    )
  }
  let txt = if etiqueta == auto { $#str(C) times #str(H) times #str(W)$ } else { etiqueta }
  content((x + w / 2 + (hojas - 1) * d / 2, y - 0.35), text(size: tam, txt))
}

// Un vector (después de aplanar) como una barra vertical
#let vector-bloque(p, n, alto, color: c-neg, etiqueta: auto, tam: 9pt) = {
  import cetz.draw: *
  let (x, y) = p
  rect((x, y), (x + 0.22, y + alto), fill: color.lighten(75%), stroke: 0.6pt + color.darken(10%))
  content((x + 0.11, y - 0.35), text(size: tam, if etiqueta == auto { str(n) } else { etiqueta }))
}

#let flecha(a, b, etiqueta, tam: 8pt, arriba: 0.3) = {
  import cetz.draw: *
  line(a, b, stroke: 0.9pt, mark: (end: ">", fill: black))
  content(((a.at(0) + b.at(0)) / 2, (a.at(1) + b.at(1)) / 2 + arriba), text(size: tam, etiqueta))
}

// Una capa convolucional: entrada ⋆ filtros = salida
#let fig-canales(C-in, H-in, filtros, k, H-out, esc: 0.08, tam: 9pt) = cetz.canvas({
  import cetz.draw: *
  let h-in = H-in * esc
  tensor-bloque((0, 0), C-in, H-in, H-in, esc: esc, color: luma(110), tam: tam)
  let x1 = h-in + 0.13 * (calc.min(C-in, 6) - 1) + 0.9
  content((x1 - 0.45, h-in / 2), text(size: tam * 1.6)[$star$])
  // los filtros: varios cubos pequeños, uno debajo de otro
  let fk = 0.36
  let paso = 0.62
  let arriba = h-in / 2 + paso * 1.2
  for t in range(3) {
    let yt = arriba - t * paso
    tensor-bloque((x1, yt - fk / 2), C-in, k, k, esc: fk / k, color: c-pos, etiqueta: [], max-hojas: 3)
  }
  content((x1 + 0.25, arriba - 3 * paso + 0.05), text(size: tam)[$dots.v$])
  content(
    (x1 + 0.3, arriba - 3 * paso - 0.55),
    text(size: tam)[#filtros filtros de $#str(C-in) times #str(k) times #str(k)$],
  )
  let x2 = x1 + 1.7
  content((x2 - 0.4, h-in / 2), text(size: tam * 1.6)[$=$])
  let h-out = H-out * esc
  tensor-bloque((x2, (h-in - h-out) / 2), filtros, H-out, H-out, esc: esc, color: c-neg, tam: tam)
})

// La arquitectura de MyCNN
#let fig-arquitectura(esc: 0.055, tam: 8pt, largo: 1.25) = cetz.canvas({
  import cetz.draw: *
  let pasos = (
    ("t", 1, 28, luma(110)),
    ("f", [conv $3 times 3$ \ + ReLU]),
    ("t", 8, 28, c-neg),
    ("f", [max pool \ $2 times 2$]),
    ("t", 8, 14, c-neg),
    ("f", [conv $3 times 3$ \ + ReLU]),
    ("t", 16, 14, c-neg),
    ("f", [max pool \ $2 times 2$]),
    ("t", 16, 7, c-neg),
    ("f", [aplanar]),
    ("v", 784, 2.0),
    ("f", [densa]),
    ("v", 10, 0.9),
  )
  let x = 0
  let base = 0
  for p in pasos {
    if p.at(0) == "t" {
      let (_, C, H, color) = p
      let h = H * esc
      tensor-bloque((x, base - h / 2), C, H, H, esc: esc, color: color, tam: tam)
      x += h + 0.13 * (calc.min(C, 6) - 1) + 0.25
    } else if p.at(0) == "v" {
      let (_, n, alto) = p
      vector-bloque((x, base - alto / 2), n, alto, color: if n == 10 { c-pos } else { c-neg }, tam: tam)
      x += 0.22 + 0.25
    } else {
      flecha((x, base), (x + largo, base), text(size: tam * 0.95, p.at(1)), tam: tam, arriba: 0.5)
      x += largo + 0.25
    }
  }
})

// ---------------------------------------------------------------------------
// Figuras con datos de MNIST
// ---------------------------------------------------------------------------

#let fig-desplazado(ancho: 2.2cm, tam: 9pt) = {
  set text(size: tam)
  grid(
    columns: 3,
    column-gutter: 1cm,
    row-gutter: 6pt,
    align: center,
    digito(d-mnist.original, ancho: ancho),
    digito(d-mnist.desplazado, ancho: ancho),
    digito(d-mnist.cercano, ancho: ancho),
    [un 4], [el mismo 4, desplazado \ #d-mnist.desplazamiento píxeles], [el #d-mnist.cercano_clase más parecido \ al primer 4],
    [], [distancia al primero: \ *#miles(calc.round(d-mnist.d_desplazado))*], [distancia al primero: \ *#miles(calc.round(d-mnist.d_cercano))*],
  )
}

#let mat-k(K) = math.mat(..K.map(f => f.map(v => [#fmt(v)])))

#let fig-bordes(ancho: 2.2cm, tam: 10pt) = {
  set text(size: tam)
  let Kv = ((-1, 0, 1), (-1, 0, 1), (-1, 0, 1))
  let Kh = ((1, 1, 1), (0, 0, 0), (-1, -1, -1))
  grid(
    columns: 5,
    column-gutter: 0.45cm,
    row-gutter: 0.35cm,
    align: center + horizon,
    grid.cell(rowspan: 2, digito(d-mnist.ejemplo, ancho: ancho)),
    $star$,
    $#mat-k(Kv)$,
    $arrow.r$,
    digito(d-mnist.bordes_v, ancho: ancho),
    $star$,
    $#mat-k(Kh)$,
    $arrow.r$,
    digito(d-mnist.bordes_h, ancho: ancho),
  )
}

#let fig-filtros(ancho-filtro: 1.15cm, ancho-mapa: 1.15cm, tam: 8pt, gutter: 6pt) = {
  set text(size: tam)
  grid(
    columns: 8,
    column-gutter: gutter,
    row-gutter: 5pt,
    align: center,
    ..d-mnist.filtros.map(f => imagen(f, ancho: ancho-filtro, pixelado: true, borde: 0.4pt + luma(150))),
    ..range(8).map(i => [filtro #(i + 1)]),
    grid.cell(colspan: 8, v(2pt)),
    ..d-mnist.mapas.map(m => imagen(m, ancho: ancho-mapa, pixelado: true)),
  )
}

#let fig-sobreajuste(size: (8, 4.4), tam: 9pt) = {
  let s = d-mnist.sobreajuste
  let tope = 1.2
  let filtrar(pts) = pts.filter(p => p.at(1) <= tope)
  let estilo(color, punteada: false) = (stroke: (paint: color, thickness: 1.5pt, dash: if punteada { "dashed" } else { none }))
  custom-plot(
    x-min: 0,
    x-max: 150,
    y-min: 0,
    y-max: tope,
    x-label: [época],
    y-label: [pérdida],
    x-tick-step: 25,
    y-tick-step: 0.4,
    size: size,
    {
      plot.add(filtrar(s.sin.train), style: estilo(c-neg, punteada: true))
      plot.add(filtrar(s.sin.val), style: estilo(c-pos))
      plot.add(filtrar(s.con.val), style: estilo(c-3))
    },
  )
}

#let linea-leyenda(color, etiqueta, punteada: false) = box[
  #box(width: 0.6cm, height: 0pt, baseline: -0.3em, stroke: (
    top: (paint: color, thickness: 1.6pt, dash: if punteada { "dashed" } else { none }),
  ))
  #etiqueta
]

#let leyenda-sobreajuste(tam: 9pt) = {
  set text(size: tam)
  grid(
    columns: 3,
    column-gutter: 14pt,
    linea-leyenda(c-neg, [entrenamiento], punteada: true),
    linea-leyenda(c-pos, [validación]),
    linea-leyenda(c-3, [validación con dropout]),
  )
}

// ---------------------------------------------------------------------------
// Repaso: el grafo de la red de dos pesos
// ---------------------------------------------------------------------------

#let nodo-grafo(p, etiqueta, tam) = {
  import cetz.draw: *
  circle(p, radius: 0.42, fill: white, stroke: 0.9pt + black)
  content(p, text(size: tam * 1.1, etiqueta))
}

#let fig-grafo(tam: 9pt) = cetz.canvas({
  import cetz.draw: *
  let xs = (0, 3.2, 6.4, 9.6, 12.8)
  let nombres = ($x$, $z$, $a$, $hat(y)$, $L$)
  let valores = ($2$, $2$, $2$, $2$, $64$)
  let ops = ($times w_1$, $"ReLU"$, $times w_2$, $(dot - 10)^2$)
  for k in range(4) {
    line((xs.at(k) + 0.45, 0), (xs.at(k + 1) - 0.45, 0), stroke: 0.9pt + black, mark: (end: ">", fill: black))
    content(((xs.at(k) + xs.at(k + 1)) / 2, 0.32), text(size: tam * 0.95, ops.at(k)))
  }
  for k in range(5) {
    nodo-grafo((xs.at(k), 0), nombres.at(k), tam)
    content((xs.at(k), 0.82), text(size: tam)[$= #valores.at(k)$])
  }
  let yb = -1.25
  for k in range(1, 4) {
    line((xs.at(k + 1) - 0.3, yb), (xs.at(k) + 0.3, yb), stroke: 0.9pt + c-pos, mark: (end: ">", fill: c-pos))
  }
  let etiquetas-b = (
    $$,
    $display((partial L) / (partial z)) = -16$,
    $display((partial L) / (partial a)) = -16$,
    $display((partial L) / (partial hat(y))) = -16$,
    $$,
  )
  for k in (1, 2, 3) {
    content((xs.at(k), yb - 0.7), text(size: tam, fill: c-pos, etiquetas-b.at(k)))
  }
  content(((xs.at(3) + xs.at(4)) / 2, yb + 0.32), text(size: tam * 0.85, fill: c-pos)[$2(hat(y) - 10)$])
  content(((xs.at(2) + xs.at(3)) / 2, yb + 0.32), text(size: tam * 0.85, fill: c-pos)[$times w_2$])
  content(((xs.at(1) + xs.at(2)) / 2, yb + 0.32), text(size: tam * 0.85, fill: c-pos)[$times 1$ (pendiente de ReLU)])
  content(((xs.at(0) + xs.at(1)) / 2, yb - 1.7), box(stroke: 0.8pt + c-pos, inset: 4pt, text(
    size: tam,
    fill: c-pos,
  )[$display((partial L) / (partial w_1)) = -16 dot x = -32$]))
  content(((xs.at(2) + xs.at(3)) / 2, yb - 1.7), box(stroke: 0.8pt + c-pos, inset: 4pt, text(
    size: tam,
    fill: c-pos,
  )[$display((partial L) / (partial w_2)) = -16 dot a = -32$]))
})

// Datos de los ejemplos
#let X-conv = ((0, 1, 2, 2, 0), (0, 0, 2, 1, 0), (1, 0, 2, 2, 0), (0, 0, 1, 2, 1), (0, 1, 2, 2, 0))
#let K-conv = ((-1, 0, 1), (-1, 0, 1), (-1, 0, 1))
#let X-pool = ((1, 3, 2, 0), (4, 2, 1, 1), (0, 1, 5, 2), (2, 0, 3, 4))
