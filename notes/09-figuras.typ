// Figuras de la clase 9 (agrupamiento), compartidas por las notas
// (notes/09-agrupamiento.typ) y las diapositivas.
// Compilar siempre con --root . desde la raíz del repositorio.
//
// Los datos vienen de `python notes/scripts/09-agrupamiento-figuras.py`.

#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *

#let d = json("/notes/figs/09/datos.json")
#let d-salidas = json("/notes/figs/09/salidas.json")

// Un color y una marca por grupo; el ruido (grupo -1) va en gris con una x
#let colores = (rgb("#1f77b4"), rgb("#d62728"), rgb("#2ca02c"), rgb("#ff7f0e"))
#let marcas = ("o", "triangle", "square", "diamond")
#let c-gris = luma(120)

#let num(x, digits: 1) = str(calc.round(x, digits: digits))

// ---------------------------------------------------------------------------
// Código y salidas
// ---------------------------------------------------------------------------

#let fuente(nombre, desde: none, hasta: none) = {
  let lineas = read("/notes/code/09/" + nombre + ".py").trim(at: end).split("\n")
  let a = if desde == none { 0 } else { desde - 1 }
  let b = if hasta == none { lineas.len() } else { hasta }
  lineas.slice(a, b).join("\n")
}

#let codigo(nombre, desde: none, hasta: none) = block(breakable: false, raw(
  fuente(nombre, desde: desde, hasta: hasta),
  lang: "python",
  block: true,
))

#let salida(nombre, tam: 8.5pt, titulo: [Salida], ancho: 100%) = block(
  width: ancho,
  fill: luma(245),
  stroke: (left: 2pt + luma(170)),
  inset: (x: 10pt, y: 7pt),
  breakable: false,
  align(left, {
    text(size: tam * 0.9, fill: luma(110), weight: "bold", titulo)
    linebreak()
    set text(font: "DejaVu Sans Mono", size: tam)
    let lineas = d-salidas.at(nombre).split("\n")
    for (i, l) in lineas.enumerate() {
      l.replace(" ", "\u{00A0}")
      if i < lineas.len() - 1 { linebreak() }
    }
  }),
)

// ---------------------------------------------------------------------------
// Puntos y paneles
// ---------------------------------------------------------------------------

// Los puntos de X, coloreados según su grupo c (o en gris si c es none)
#let puntos(X, c: none, tam: 0.09) = {
  if c == none {
    plot.add(X, mark: "o", mark-size: tam, mark-style: (stroke: none, fill: luma(90)), style: (stroke: none))
  } else {
    let k = calc.max(..c) + 1
    for j in range(k) {
      let pts = X.enumerate().filter(((i, p)) => c.at(i) == j).map(((i, p)) => p)
      if pts.len() > 0 {
        plot.add(
          pts,
          mark: marcas.at(calc.rem(j, 4)),
          mark-size: tam,
          mark-style: (stroke: none, fill: colores.at(calc.rem(j, 4))),
          style: (stroke: none),
        )
      }
    }
    let ruido = X.enumerate().filter(((i, p)) => c.at(i) == -1).map(((i, p)) => p)
    if ruido.len() > 0 {
      plot.add(ruido, mark: "x", mark-size: tam * 1.1, mark-style: (stroke: 0.8pt + c-gris), style: (stroke: none))
    }
  }
}

// Centroides: una cruz grande del color de su grupo
#let centroides(mu, tam: 0.32, gris: false) = {
  for (j, m) in mu.enumerate() {
    let color = if gris { black } else { colores.at(calc.rem(j, 4)) }
    plot.add((m,), mark: "x", mark-size: tam + 0.08, mark-style: (stroke: 4pt + white), style: (stroke: none))
    plot.add((m,), mark: "x", mark-size: tam, mark-style: (stroke: 2.2pt + color), style: (stroke: none))
  }
}

// Límites de un conjunto de puntos, con un margen
#let limites(X, margen: 0.1) = {
  let xs = X.map(p => p.at(0))
  let ys = X.map(p => p.at(1))
  let (x0, x1, y0, y1) = (calc.min(..xs), calc.max(..xs), calc.min(..ys), calc.max(..ys))
  let (mx, my) = ((x1 - x0) * margen, (y1 - y0) * margen)
  (x0 - mx, x1 + mx, y0 - my, y1 + my)
}

#let panel(X, cuerpo, size: (4.4, 3.6), lim: auto, x-label: none, y-label: none) = {
  let (x0, x1, y0, y1) = if lim == auto { limites(X) } else { lim }
  custom-plot(
    x-min: x0,
    x-max: x1,
    y-min: y0,
    y-max: y1,
    x-label: x-label,
    y-label: y-label,
    size: size,
    cuerpo,
  )
}

#let con-titulo(titulo, cuerpo, tam: 9pt) = align(center, stack(spacing: 4pt, text(size: tam, titulo), cuerpo))

// ---------------------------------------------------------------------------
// K-means
// ---------------------------------------------------------------------------

// Un paso individual de K-means (usado en las diapositivas)
#let fig-paso(i, size: (4.4, 3.6)) = {
  let X = d.pasos.X
  let P = d.pasos.pasos
  let todos = X + P.filter(p => p.mu != none).map(p => p.mu).flatten().chunks(2)
  let lim = limites(todos, margen: 0.08)
  let flechas(antes, despues) = plot.annotate({
    import cetz.draw: *
    for (a, b) in antes.zip(despues) {
      line(a, b, stroke: 0.9pt + black, mark: (end: ">", fill: black, scale: 0.6))
    }
  })
  let p = P.at(i)
  panel(
    X,
    {
      puntos(X, c: p.c)
      if "antes" in p { flechas(p.antes, p.mu) }
      if p.mu != none { centroides(p.mu) }
    },
    size: size,
    lim: lim,
  )
}

// K-means paso a paso: (a) datos, (b) centroides iniciales, (c) asignar,
// (d) mover, (e) asignar de nuevo, (f) resultado final
#let fig-pasos(size: (4.4, 3.6), tam: 9pt) = {
  let X = d.pasos.X
  let P = d.pasos.pasos
  let todos = X + P.filter(p => p.mu != none).map(p => p.mu).flatten().chunks(2)
  let lim = limites(todos, margen: 0.08)
  let flechas(antes, despues) = plot.annotate({
    import cetz.draw: *
    for (a, b) in antes.zip(despues) {
      line(a, b, stroke: 0.9pt + black, mark: (end: ">", fill: black, scale: 0.6))
    }
  })
  let paneles = (
    ([(a) Datos], { puntos(X) }),
    ([(b) Centroides iniciales], { puntos(X); centroides(P.at(1).mu) }),
    ([(c) Asignar], { puntos(X, c: P.at(2).c); centroides(P.at(2).mu) }),
    ([(d) Mover], { puntos(X, c: P.at(3).c); flechas(P.at(3).antes, P.at(3).mu); centroides(P.at(3).mu) }),
    ([(e) Asignar], { puntos(X, c: P.at(4).c); centroides(P.at(4).mu) }),
    ([(f) Resultado final], { puntos(X, c: P.last().c); centroides(P.last().mu) }),
  )
  grid(
    columns: 3,
    column-gutter: 0.3cm,
    row-gutter: 0.35cm,
    ..paneles.map(((t, cuerpo)) => con-titulo(t, panel(X, cuerpo, size: size, lim: lim), tam: tam)),
  )
}

#let fig-minimos(size: (5.2, 4.2), tam: 9pt) = {
  let m = d.minimos
  let lim = limites(m.X, margen: 0.08)
  grid(
    columns: 2,
    column-gutter: 0.8cm,
    con-titulo([$J = #num(m.bueno.J)$], panel(m.X, { puntos(m.X, c: m.bueno.c); centroides(m.bueno.mu) }, size: size, lim: lim), tam: tam),
    con-titulo([$J = #num(m.malo.J)$], panel(m.X, { puntos(m.X, c: m.malo.c); centroides(m.malo.mu) }, size: size, lim: lim), tam: tam),
  )
}

#let fig-codo(size: (5.6, 3.6), tam: 9pt) = {
  let codo = d.elegir_k.codo.map(((k, j)) => (k, j / 1000))
  let sil = d.elegir_k.silueta
  let jmax = calc.max(..codo.map(p => p.at(1)))
  let estilo = (stroke: (paint: colores.at(0), thickness: 1.5pt))
  let marca = (mark: "o", mark-size: 0.12, mark-style: (fill: colores.at(0), stroke: none))
  grid(
    columns: 2,
    column-gutter: 0.8cm,
    con-titulo([Método del codo], custom-plot(
      x-min: 0.5, x-max: 8.5, y-min: 0, y-max: calc.ceil(jmax / 5) * 5,
      x-label: $k$, y-label: [$J$ (miles)], x-tick-step: 1, y-tick-step: 5, size: size,
      {
        plot.add(codo, style: estilo, ..marca)
        plot.add(((4, codo.at(3).at(1)),), mark: "o", mark-size: 0.3, mark-style: (fill: none, stroke: 1.5pt + colores.at(1)), style: (stroke: none))
      },
    ), tam: tam),
    con-titulo([Silueta], custom-plot(
      x-min: 1.5, x-max: 8.5, y-min: 0.3, y-max: 0.9,
      x-label: $k$, y-label: [silueta], x-tick-step: 1, y-tick-step: 0.2, size: size,
      {
        plot.add(sil, style: estilo, ..marca)
        plot.add(((4, sil.at(2).at(1)),), mark: "o", mark-size: 0.3, mark-style: (fill: none, stroke: 1.5pt + colores.at(1)), style: (stroke: none))
      },
    ), tam: tam),
  )
}

#let fig-elegir-k-datos(size: (4.4, 3.6)) = {
  let e = d.elegir_k
  panel(e.X, { puntos(e.X) }, size: size)
}

#let fig-escala(size: (5.2, 3.8), tam: 9pt) = {
  let e = d.escala
  let lim = (15, 70, 15, 85)
  grid(
    columns: 2,
    column-gutter: 0.8cm,
    con-titulo([Sin estandarizar], panel(e.X, { puntos(e.X, c: e.sin) }, size: size, lim: lim, x-label: [edad (años)], y-label: [ingreso (miles)]), tam: tam),
    con-titulo([Estandarizando], panel(e.X, { puntos(e.X, c: e.con) }, size: size, lim: lim, x-label: [edad (años)], y-label: [ingreso (miles)]), tam: tam),
  )
}

#let paleta(colores-rgb, lado: 0.32cm) = box(grid(
  columns: colores-rgb.len(),
  ..colores-rgb.map(((r, g, b)) => box(width: lado, height: lado, fill: rgb(r, g, b))),
))

#let fig-colores(ancho: 3.7cm, tam: 9pt) = {
  let foto(archivo) = image("/notes/figs/09/" + archivo, width: ancho)
  let p = d.colores.paletas
  grid(
    columns: 4,
    column-gutter: 0.25cm,
    row-gutter: 0.2cm,
    align: center,
    foto("foto-original.png"), foto("foto-k2.png"), foto("foto-k4.png"), foto("foto-k16.png"),
    text(size: tam)[original], text(size: tam)[$k = 2$], text(size: tam)[$k = 4$], text(size: tam)[$k = 16$],
    [], paleta(p.at("2"), lado: ancho / 16), paleta(p.at("4"), lado: ancho / 16), paleta(p.at("16"), lado: ancho / 16),
  )
}

// Nube de píxeles de la foto en el cubo RGB (proyección 3D isométrica)
#let fig-nube(agrupado: true, lado: 2.3, tam: 9pt) = {
  let c = d.colores
  let paleta4 = c.paletas.at("4")
  let cos30 = calc.sqrt(3) / 2
  let proj(p) = {
    let (rn, gn, bn) = (p.at(0) / 255, p.at(1) / 255, p.at(2) / 255)
    ((gn - rn) * cos30 * lado, (bn - 0.5 * (rn + gn)) * lado)
  }
  let ordenados = c.nube.zip(c.nube_c).sorted(key: item => item.at(0).at(0) + item.at(0).at(1) + item.at(0).at(2))
  let cubo = cetz.canvas({
    import cetz.draw: *
    let o = proj((0, 0, 0))
    let px = proj((255, 0, 0))
    let py = proj((0, 255, 0))
    let pz = proj((0, 0, 255))
    let pxy = proj((255, 255, 0))
    let pxz = proj((255, 0, 255))
    let pyz = proj((0, 255, 255))
    let pxyz = proj((255, 255, 255))

    let estilo-eje = (paint: luma(140), thickness: 0.7pt)
    let estilo-arista = (paint: luma(195), thickness: 0.6pt)
    line(o, px, stroke: estilo-eje)
    line(o, py, stroke: estilo-eje)
    line(o, pz, stroke: estilo-eje)
    line(px, pxy, py, pyz, pz, pxz, px, stroke: estilo-arista)

    for (p, grupo) in ordenados {
      let col = if agrupado { paleta4.at(grupo) } else { p }
      let (r, g, b) = col
      circle(proj(p), radius: 0.055, fill: rgb(r, g, b), stroke: 0.25pt + luma(60).transparentize(60%))
    }

    if agrupado {
      for (r, g, b) in paleta4 {
        let pos = proj((r, g, b))
        circle(pos, radius: 0.20, fill: white, stroke: none)
        circle(pos, radius: 0.16, fill: rgb(r, g, b), stroke: 1pt + black)
      }
    }

    let estilo-frente = (paint: luma(180), thickness: 0.5pt, dash: "dashed")
    line(pxy, pxyz, stroke: estilo-frente)
    line(pxz, pxyz, stroke: estilo-frente)
    line(pyz, pxyz, stroke: estilo-frente)

    content((px.at(0) - 0.22, px.at(1) - 0.1), text(size: tam, weight: "bold", fill: rgb("#c0392b"))[R])
    content((py.at(0) + 0.22, py.at(1) - 0.1), text(size: tam, weight: "bold", fill: rgb("#27ae60"))[G])
    content((pz.at(0), pz.at(1) + 0.22), text(size: tam, weight: "bold", fill: rgb("#2980b9"))[B])
  })
  if agrupado {
    con-titulo([K-means con $k = 4$], cubo, tam: tam)
  } else {
    cubo
  }
}

#let fig-kmeans-lunas(size: (5.2, 3.8), tam: 9pt) = {
  let l = d.lunas
  let lim = limites(l.X, margen: 0.05)
  grid(
    columns: 2,
    column-gutter: 0.8cm,
    con-titulo([Datos], panel(l.X, { puntos(l.X, tam: 0.07) }, size: size, lim: lim), tam: tam),
    con-titulo([K-means con $k = 2$], panel(l.X, { puntos(l.X, c: l.kmeans, tam: 0.07) }, size: size, lim: lim), tam: tam),
  )
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------

#let circulo(centro, r, estilo) = plot.add(
  domain: (0, 2 * calc.pi),
  samples: 60,
  t => (centro.at(0) + r * calc.cos(t), centro.at(1) + r * calc.sin(t)),
  style: estilo,
)

#let fig-tipos(size: (6.6, 5.06), tam: 9pt) = {
  let t = d.tipos
  let lim = (0.0, 6.0, 0.0, 4.6)  // misma escala en x y en y: size proporcional
  let col(tipo) = if tipo == "nucleo" { colores.at(0) } else if tipo == "frontera" { colores.at(3) } else { c-gris }
  let grafica = custom-plot(
    x-min: lim.at(0), x-max: lim.at(1), y-min: lim.at(2), y-max: lim.at(3), size: size,
    {
      for (p, tipo) in t.P.zip(t.tipo) {
        circulo(p, t.eps, (stroke: (paint: col(tipo).transparentize(40%), thickness: 0.7pt, dash: if tipo == "nucleo" { none } else { "dashed" })))
      }
      for (p, tipo) in t.P.zip(t.tipo) {
        if tipo == "ruido" {
          plot.add((p,), mark: "x", mark-size: 0.2, mark-style: (stroke: 1.4pt + c-gris), style: (stroke: none))
        } else {
          plot.add((p,), mark: "o", mark-size: 0.17, mark-style: (stroke: 1.2pt + col(tipo), fill: if tipo == "nucleo" { col(tipo) } else { white }), style: (stroke: none))
        }
      }
    },
  )
  let leyenda = {
    set text(size: tam)
    grid(
      columns: 3,
      column-gutter: 14pt,
      [#text(fill: colores.at(0))[●] núcleo], [#text(fill: colores.at(3))[○] frontera], [#text(fill: c-gris)[✕] ruido],
    )
  }
  align(center, stack(spacing: 0.3cm, grafica, leyenda))
}

#let fig-dbscan-lunas(size: (6, 4.2)) = {
  let l = d.lunas
  panel(l.X, { puntos(l.X, c: l.dbscan, tam: 0.07) }, size: size, lim: limites(l.X, margen: 0.05))
}

#let fig-eps(size: (4.4, 3.4), tam: 9pt) = {
  let l = d.lunas
  let lim = limites(l.X, margen: 0.05)
  grid(
    columns: 3,
    column-gutter: 0.3cm,
    ..l.eps.zip(l.eps_paneles).map(((e, c)) => con-titulo(
      [`eps` $= #str(e)$],
      panel(l.X, { puntos(l.X, c: c, tam: 0.06) }, size: size, lim: lim),
      tam: tam,
    )),
  )
}

#let fig-kdist(size: (7, 3.6), eps: 0.15) = {
  let k = d.lunas.k_dist
  custom-plot(
    x-min: 0, x-max: k.len(), y-min: 0, y-max: 0.5,
    x-label: [puntos, ordenados por distancia], y-label: [distancia al 5.º vecino],
    x-tick-step: 50, y-tick-step: 0.1, size: size,
    {
      plot.add-hline(eps, style: (stroke: (paint: colores.at(1), thickness: 1pt, dash: "dashed")))
      plot.add(k.filter(p => p.at(1) <= 0.5), style: (stroke: (paint: colores.at(0), thickness: 1.6pt)))
    },
  )
}

#let fig-densidades(size: (6.4, 3.8)) = {
  let e = d.densidades
  panel(e.X, { puntos(e.X, c: e.c, tam: 0.07) }, size: size)
}

// ---------------------------------------------------------------------------
// Agrupamiento jerárquico
// ---------------------------------------------------------------------------

#let fig-dendrograma(size-puntos: (5, 4.2), alto: 4.2, ancho: 6, tam: 9pt, corte: true) = {
  let e = d.dendrograma
  let grupo-de(nombre) = e.grupos.at(e.nombres.position(n => n == nombre))
  let grafica-puntos = custom-plot(
    x-min: 0.4, x-max: 5.2, y-min: 0.4, y-max: 5.2, size: size-puntos,
    {
      for (i, p) in e.P.enumerate() {
        let g = e.grupos.at(i)
        plot.add((p,), mark: marcas.at(g), mark-size: 0.17, mark-style: (stroke: none, fill: if corte { colores.at(g) } else { luma(90) }), style: (stroke: none))
      }
      plot.annotate({
        import cetz.draw: *
        for (i, p) in e.P.enumerate() {
          content((p.at(0) + 0.28, p.at(1) + 0.22), text(size: tam, e.nombres.at(i)))
        }
      })
    },
  )
  let hmax = calc.max(..e.dcoord.flatten()) * 1.08
  let n = e.hojas.len()
  let grafica-dendro = cetz.canvas({
    import cetz.draw: *
    let sx(x) = (x - 5) / (10 * (n - 1)) * ancho
    let sy(y) = y / hmax * alto
    // eje vertical
    line((-0.3, 0), (-0.3, alto), stroke: 0.6pt)
    for h in (0, 0.5, 1, 1.5, 2, 2.5) {
      if h <= hmax {
        line((-0.38, sy(h)), (-0.3, sy(h)), stroke: 0.6pt)
        content((-0.45, sy(h)), anchor: "east", text(size: tam * 0.85, str(h)))
      }
    }
    content((-1.25, alto / 2), angle: 90deg, text(size: tam)[distancia al unir])
    for (xs, ys) in e.icoord.zip(e.dcoord) {
      line(..xs.zip(ys).map(((x, y)) => (sx(x), sy(y))), stroke: 1.1pt + luma(60))
    }
    for (i, h) in e.hojas.enumerate() {
      let x = sx(5 + 10 * i)
      content((x, -0.3), text(size: tam, fill: if corte { colores.at(grupo-de(h)) } else { black }, weight: "bold", h))
    }
    if corte {
      line((-0.3, sy(e.corte)), (ancho + 0.3, sy(e.corte)), stroke: (paint: colores.at(1), thickness: 1pt, dash: "dashed"))
      content((ancho + 0.35, sy(e.corte)), anchor: "west", text(size: tam * 0.9, fill: colores.at(1))[corte])
    }
  })
  grid(columns: 2, column-gutter: 1cm, align: horizon, grafica-puntos, grafica-dendro)
}

#let fig-enlace(size: (5.2, 3.8), tam: 9pt) = {
  let e = d.enlace
  let lim = limites(e.X, margen: 0.05)
  grid(
    columns: 2,
    column-gutter: 0.8cm,
    con-titulo([Enlace _single_], panel(e.X, { puntos(e.X, c: e.single, tam: 0.07) }, size: size, lim: lim), tam: tam),
    con-titulo([Enlace de Ward], panel(e.X, { puntos(e.X, c: e.ward, tam: 0.07) }, size: size, lim: lim), tam: tam),
  )
}

// ---------------------------------------------------------------------------
// Comparación final
// ---------------------------------------------------------------------------

#let fig-comparacion(size: (3.9, 3.2), tam: 9pt) = {
  let c = d.comparacion
  let conjuntos = ("bolas", "lunas", "circulos")
  let algoritmos = (("kmeans", [K-means]), ("dbscan", [DBSCAN]), ("single", [Jerárquico (_single_)]))
  grid(
    columns: 3,
    column-gutter: 0.3cm,
    row-gutter: 0.3cm,
    align: center + horizon,
    ..algoritmos.map(((_, t)) => text(size: tam, weight: "bold", t)),
    ..conjuntos
      .map(nombre => algoritmos.map(((alg, _)) => {
        let X = c.at(nombre).X
        panel(X, { puntos(X, c: c.at(nombre).at(alg), tam: 0.05) }, size: size, lim: limites(X, margen: 0.05))
      }))
      .flatten(),
  )
}
