// OUTLINE (3 horas: 150 min de teoría + 30 min de laboratorio)
// 1. Márgenes (25 min)
// 1.1. Intuición: confianza y distancia a la frontera
// 1.2. Nueva notación: y en {-1, 1}, w y b
// 1.3. Margen funcional
// 1.4. Margen geométrico (derivación completa)
// 2. El clasificador de margen óptimo (25 min)
// 2.1. Primer planteamiento
// 2.2. Truco de escala: min 1/2 ||w||^2
// 3. El problema dual (40 min)
// 3.1. Por qué w es una combinación de los ejemplos
// 3.2. Todo en términos de productos internos
// 3.3. Dualidad de Lagrange (versión completa; en clase se puede dar
//      solo el Lagrangiano, las derivadas y las condiciones KKT)
// 3.4. El dual del clasificador de margen óptimo
// 3.5. Vectores de soporte
// 4. Kernels (40 min)
// 4.1. La receta
// 4.2. Kernels polinomiales
// 4.3. Kernels como medida de similitud: el kernel gaussiano
// 4.4. Kernels válidos y teorema de Mercer
// 4.5. Máquinas de soporte vectorial
// 5. Margen suave y aplicaciones (20 min)
// 6. Laboratorio: SVC con margen suave y selección de kernel (30 min)
//
// Los datos de las figuras se generan con `python generar_figuras.py`
// y se leen desde la carpeta figs/.

#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *

#let fvec(x) = math.bold(math.upright(x))

#set text(lang: "es")
#set smartquote(quotes: "“”")

#let phi = sym.phi.alt

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
  title: [Máquinas de soporte vectorial y kernels],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

// Datos de las figuras
#let d-sep = json("figs/05/separables.json")
#let d-gauss = json("figs/05/gaussiano.json")
#let d-out = json("figs/05/outlier.json")
#let d-c = json("figs/05/efecto_C.json")

#let num(x, digits: 2) = str(calc.round(x, digits: digits))

// Estilo de las clases (el mismo en todas las figuras)
#let c-neg = rgb("#1f77b4")
#let c-pos = rgb("#d62728")

#let puntos(datos, clase, tam: 0.13) = plot.add(
  datos,
  mark: if clase < 0 { "o" } else { "triangle" },
  mark-size: tam,
  mark-style: (stroke: none, fill: if clase < 0 { c-neg } else { c-pos }),
  style: (stroke: none),
)

#let puntos-clases(d, tam: 0.13) = {
  puntos(d.neg, -1, tam: tam)
  puntos(d.pos, 1, tam: tam)
}

// Vectores de soporte: círculo vacío alrededor del punto
#let soporte(sv, tam: 0.3) = plot.add(
  sv,
  mark: "o",
  mark-size: tam,
  mark-style: (stroke: 0.9pt + black, fill: none),
  style: (stroke: none),
)

// Recta w_1 x_1 + w_2 x_2 + b = c
#let recta(w, b, c: 0, dominio: (0, 5), estilo: (paint: black, thickness: 1.5pt)) = plot.add(
  domain: dominio,
  x1 => (c - b - w.at(0) * x1) / w.at(1),
  style: (stroke: estilo),
)

#let margenes(w, b, dominio: (0, 5)) = {
  for c in (-1, 1) {
    recta(w, b, c: c, dominio: dominio, estilo: (paint: gray.darken(30%), thickness: 0.9pt, dash: "dashed"))
  }
}

#let leyenda-clases = {
  set text(size: 9pt)
  grid(
    columns: 2,
    column-gutter: 16pt,
    box[#text(fill: c-neg)[●] $y = -1$],
    box[#text(fill: c-pos)[▲] $y = 1$],
  )
}

#let panel-2d(cuerpo, x-min: 0, x-max: 5, y-min: 0, y-max: 5, size: (5.5, 5.5), x-label: $x_1$, y-label: $x_2$) = custom-plot(
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

En el estudio de los clasificadores lineales dos preguntas quedaron abiertas: 

1. Cuando los datos se pueden separar con una recta, hay infinitas rectas que lo hacen, ¿cuál es la mejor? 
2. Con una transformación de features $phi(fvec(x))$ se pueden obtener fronteras no lineales, pero $phi(fvec(x))$ puede tener millones de componentes, ¿se puede evitar calcularlo? 

Las máquinas de soporte vectorial (SVM, por sus siglas en inglés) responden ambas.

= Parte 1: Márgenes

== Intuición

En regresión logística, el modelo predice $y = 1$ cuando el score $theta^T fvec(x)$ es positivo, y la probabilidad que asigna crece con el score. Un score muy positivo significa que el modelo está muy seguro. Y como el score es proporcional a la distancia del punto a la frontera, estar seguro significa estar lejos de la frontera.

#let p-conf = ((4.45, 4.4), (3.25, 3.05), (2.05, 2.8))

#figure(
  caption: [Tres puntos nuevos (A, B, C), todos del lado donde el modelo predice $y = 1$. Estamos muy seguros de la predicción en A, que está lejos de la frontera. En C, que está muy cerca, un pequeño cambio en la frontera bastaría para cambiar la predicción.],
  [
    #panel-2d({
      import cetz.draw: *
      recta(d-sep.opt.w, d-sep.opt.b)
      puntos-clases(d-sep)
      plot.add(p-conf, mark: "o", mark-size: 0.14, mark-style: (stroke: none, fill: black), style: (stroke: none))
      plot.annotate({
        for (k, nombre) in ("A", "B", "C").enumerate() {
          let (x, y) = p-conf.at(k)
          content((x - 0.12, y + 0.12), anchor: "south-east", text(size: 10pt)[*#nombre*])
        }
      })
    })
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-confianza>

Esto sugiere un criterio para escoger la frontera: entre todas las que separan los datos, preferir una que deje todos los ejemplos de entrenamiento lo más lejos posible de ella. Así, todas las predicciones sobre el conjunto de entrenamiento serían correctas y seguras. 

== Notación

En el contexto de SVM es costumbre usar una notación un poco distinta a la vista hasta el momento para problemas de regresión lineal y logística:

- Las etiquetas son $y in {-1, 1}$, en lugar de ${0, 1}$.
- En lugar de $theta$, usamos $fvec(w) in RR^d$ y $b in RR$, y escribimos el score como $fvec(w)^T fvec(x) + b$. Es decir, $b$ es lo que antes era $theta_0$, y $fvec(w) = (theta_1, ..., theta_d)$. Ya no se usa la convención $x_0 = 1$: el vector $fvec(x) in RR^d$ solo contiene las $d$ features.
- El clasificador es

$
  h_(fvec(w), b) (fvec(x)) = g(fvec(w)^T fvec(x) + b), quad quad g(z) = cases(
    1 quad & "si" z >= 0,
    -1 quad & "si" z < 0.
  )
$

A diferencia de la regresión logística, este clasificador predice directamente la clase, sin pasar por una probabilidad.

Se sigue utilizando $m$ para el número de ejemplos de entrenamiento $(fvec(x)^(\(i\)), y^(\(i\)))$.

== Margen funcional

Dado un ejemplo de entrenamiento $(fvec(x)^(\(i\)), y^(\(i\)))$, definimos el margen funcional de $(fvec(w), b)$ respecto a ese ejemplo como

$
  hat(gamma)^(\(i\)) = y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b).
$

La idea es la siguiente. Si $y^(\(i\)) = 1$, queremos que el score $fvec(w)^T fvec(x)^(\(i\)) + b$ sea un número positivo grande. Si $y^(\(i\)) = -1$, queremos que sea un número negativo grande. 

En ambos casos, lo que queremos es que el producto $y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b)$ sea positivo y grande. 

Además, $hat(gamma)^(\(i\)) > 0$ exactamente cuando el ejemplo está bien clasificado. Así, un margen funcional grande representa una predicción correcta y segura.

Para todo el conjunto de entrenamiento, el margen funcional es el del peor ejemplo:

$
  hat(gamma) = min_(i = 1, ..., m) hat(gamma)^(\(i\)).
$

El margen funcional presenta un problema. Supongamos que $fvec(w) = (2, 1)$ y $b = -4$. La frontera es la recta $2 x_1 + x_2 - 4 = 0$. Si multiplicamos ambos por 10, $fvec(w) = (20, 10)$ y $b = -40$, la frontera es $20 x_1 + 10 x_2 - 40 = 0$, que es exactamente la misma recta. El clasificador no cambió en absoluto, porque $g$ solo depende del signo del score. Pero todos los márgenes funcionales se multiplicaron por 10. 

Es decir, podemos hacer el margen funcional tan grande como queramos, alterando el score sin necesariamente mejorar las predicciones. Se necesita entonces una medida que no dependa de la escala de $fvec(w)$ y $b$.

== Margen geométrico

La medida natural es la distancia del punto a la frontera. En la @fig-margen-geometrico, el punto A corresponde a un ejemplo positivo $fvec(x)^(\(i\))$, y queremos calcular su distancia $gamma^(\(i\))$ a la frontera, que es la longitud del segmento AB.

#figure(
  caption: [Margen geométrico. El vector $fvec(w)$ es perpendicular a la frontera. La distancia $gamma^(\(i\))$ del punto A a la frontera es la longitud del segmento AB, donde B es el punto de la frontera más cercano a A.],
  [
    #panel-2d({
      import cetz.draw: *
      let w = d-sep.opt.w
      let nw = calc.sqrt(w.at(0) * w.at(0) + w.at(1) * w.at(1))
      recta(w, d-sep.opt.b)
      puntos-clases(d-sep)
      let (ax, ay) = d-sep.A
      let (bx, by) = d-sep.B
      plot.add(((ax, ay), (bx, by)), style: (stroke: (paint: black, thickness: 1pt, dash: "dashed")))
      // Vector w desde un punto de la frontera
      let p = (bx + 1.3 * w.at(1) / nw, by - 1.3 * w.at(0) / nw)
      vector(p, (p.at(0) + w.at(0) * 0.9, p.at(1) + w.at(1) * 0.9), label: $fvec(w)$, label-position: "head")
      plot.add(((ax, ay), (bx, by)), mark: "o", mark-size: 0.13, mark-style: (stroke: none, fill: black), style: (stroke: none))
      plot.annotate({
        content((ax + 0.12, ay), anchor: "west", text(size: 10pt)[*A*])
        content((bx + 0.12, by - 0.05), anchor: "north-west", text(size: 10pt)[*B*])
        content(((ax + bx) / 2 + 0.12, (ay + by) / 2 - 0.1), anchor: "north-west", text(size: 10pt)[$gamma^(\(i\))$])
      })
    })
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-margen-geometrico>

Como se ha visto anteriormente, el vector $fvec(w)$ es perpendicular a la frontera. Si $fvec(u)$ y $fvec(v)$ son dos puntos de la frontera, entonces $fvec(w)^T fvec(u) + b = 0$ y $fvec(w)^T fvec(v) + b = 0$; restando, $fvec(w)^T (fvec(u) - fvec(v)) = 0$. Es decir, $fvec(w)$ es perpendicular a cualquier vector que esté en la dirección de la frontera.

Entonces $fvec(w) slash norm(fvec(w))$ es un vector de longitud 1 perpendicular a la frontera, que apunta hacia el lado de los ejemplos positivos. Para llegar de A a B, avanzamos una distancia $gamma^(\(i\))$ en la dirección opuesta:

$
  B = fvec(x)^(\(i\)) - gamma^(\(i\)) fvec(w) / norm(fvec(w)).
$

Como B está sobre la frontera, cumple $fvec(w)^T B + b = 0$:

$
  fvec(w)^T (fvec(x)^(\(i\)) - gamma^(\(i\)) fvec(w) / norm(fvec(w))) + b = 0.
$

Despejando:

$
  gamma^(\(i\)) = (fvec(w)^T fvec(x)^(\(i\)) + b) / norm(fvec(w))
$

Este cálculo fue para un ejemplo positivo. Para un ejemplo negativo bien clasificado, el score es negativo, y basta multiplicar por $y^(\(i\)) = -1$ para obtener una distancia positiva. En general, definimos el margen geométrico de $(fvec(w), b)$ respecto al ejemplo $i$ como

$
  gamma^(\(i\)) = 
  y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) / norm(fvec(w)) = 
  hat(gamma)^(\(i\)) / norm(fvec(w)).
$

El margen geométrico es positivo si el ejemplo está bien clasificado, y en ese caso es su distancia a la frontera. Además, no cambia si multiplicamos $fvec(w)$ y $b$ por una constante positiva $c$: el numerador y el denominador se multiplican por $c$. 

En el ejemplo anterior, con $fvec(w) = (2, 1)$, el margen geométrico es el margen funcional dividido entre $sqrt(5)$; con $fvec(w) = (20, 10)$ es el margen funcional (10 veces mayor) dividido entre $sqrt(500) = 10 sqrt(5)$. El resultado es el mismo.

Si $norm(fvec(w)) = 1$, los dos márgenes coinciden.

Igual que antes, el margen geométrico del conjunto de entrenamiento es el del peor ejemplo:

$
  gamma = min_(i = 1, ..., m) gamma^(\(i\)).
$

= Parte 2: El clasificador de margen óptimo

== Primer planteamiento

Supongamos por ahora que los datos son linealmente separables. Queremos la frontera que maximiza el margen geométrico, es decir, la que deja al ejemplo más cercano lo más lejos posible. Podemos escribirlo así:

$
  max_(gamma, fvec(w), b) quad & gamma \
  "sujeto a" quad & y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) >= gamma, quad i = 1, ..., m \
  & norm(fvec(w)) = 1.
$

Queremos maximizar $gamma$ de forma que todos los ejemplos tengan margen funcional al menos $gamma$. La condición $norm(fvec(w)) = 1$ hace que el margen funcional sea igual al geométrico, así que todos los ejemplos quedan a distancia al menos $gamma$ de la frontera.

El problema es que la restricción $norm(fvec(w)) = 1$ es difícil de manejar: el conjunto de vectores de norma 1 es una esfera, que no es convexa, y los algoritmos de optimización estándar no la aceptan.

== Truco de escala

Podemos quitar esa restricción si maximizamos directamente el margen geométrico, $hat(gamma) slash norm(fvec(w))$:

$
  max_(hat(gamma), fvec(w), b) quad & hat(gamma) / norm(fvec(w)) \
  "sujeto a" quad & y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) >= hat(gamma), quad i = 1, ..., m.
$

Ya no hay restricción sobre la norma, pero la función objetivo $hat(gamma) slash norm(fvec(w))$ es difícil de optimizar.

Aquí puede usarse la observación de la Parte 1: podemos multiplicar $fvec(w)$ y $b$ por cualquier constante positiva sin cambiar la frontera. Al hacerlo, el margen funcional $hat(gamma)$ se multiplica por la misma constante. Así que siempre podemos escoger una escala de $fvec(w)$ y $b$ de forma que el margen funcional del conjunto sea exactamente 1:

$
  hat(gamma) = 1.
$

Con $hat(gamma) = 1$, el objetivo es maximizar $1 slash norm(fvec(w))$, que es lo mismo que minimizar $norm(fvec(w))$, y también lo mismo que minimizar $1/2 norm(fvec(w))^2$. De esta forma podemos escribir nuestro objetivo como:

$
  min_(fvec(w), b) quad & 1/2 norm(fvec(w))^2 \
  "sujeto a" quad & y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) >= 1, quad i = 1, ..., m.
$

Este es el clasificador de margen óptimo. La función objetivo es cuadrática y convexa (es decir, existe un mínimo global), y las restricciones son lineales en $fvec(w)$ y $b$. A los problemas de este tipo se les llama problemas de programación cuadrática, y existen algoritmos eficientes para resolverlos.

Su solución deja todos los ejemplos con margen funcional al menos 1. Los ejemplos más cercanos a la frontera tienen margen funcional exactamente 1, es decir, están sobre una de las rectas $fvec(w)^T fvec(x) + b = 1$ o $fvec(w)^T fvec(x) + b = -1$. La distancia de la frontera a cada una de esas rectas es $1 slash norm(fvec(w))$, que es el margen geométrico.

Nótese que $1/2 norm(fvec(w))^2$ es exactamente la penalización de la regularización $L_2$. El clasificador de margen óptimo busca el $fvec(w)$ más pequeño posible que clasifique todos los ejemplos con margen funcional al menos 1.

#let panel-margen(w, b) = panel-2d({
  margenes(w, b)
  recta(w, b)
  puntos-clases(d-sep)
})

#figure(
  caption: [Dos fronteras que separan los mismos datos, con las rectas $fvec(w)^T fvec(x) + b = plus.minus 1$ (punteadas). A la izquierda, una frontera cualquiera, con margen geométrico #num(d-sep.otra.margen). A la derecha, el clasificador de margen óptimo, con margen geométrico #num(d-sep.opt.margen). Los ejemplos marcados con un círculo son los vectores de soporte: los únicos que están sobre las rectas punteadas.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      panel-margen(d-sep.otra.w, d-sep.otra.b),
      panel-2d({
        margenes(d-sep.opt.w, d-sep.opt.b)
        recta(d-sep.opt.w, d-sep.opt.b)
        puntos-clases(d-sep)
        soporte(d-sep.opt.sv)
      }),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-margen-optimo>

El problema anterior se puede resolver con cualquier software de programación cuadrática. Reescribamos ahora el problema de otra forma, llamada el problema dual, que tiene dos ventajas. Primero, muestra que la solución depende solo de unos pocos ejemplos (los vectores de soporte de la @fig-margen-optimo). Segundo, y más importante, solo depende de los datos a través de productos internos $chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r = (fvec(x)^(\(i\)))^T fvec(x)^(\(j\))$. 

= Parte 3: El problema dual

== Por qué $fvec(w)$ es una combinación de los ejemplos

Veamos que el $fvec(w)$ óptimo se puede escribir como una combinación lineal de los ejemplos de entrenamiento:

$
  fvec(w) = sum_(i=1)^m alpha_i y^(\(i\)) fvec(x)^(\(i\)),
$

para ciertos números $alpha_1, ..., alpha_m$. Escribimos los coeficientes como $alpha_i y^(\(i\))$, y no simplemente como $alpha_i$, porque así los $alpha_i$ resultarán ser no negativos. Hay dos formas de convencerse de esto.

La primera viene del entrenamiento con gradientes. En el pasado vimos que, en regresión lineal y en regresión logística, cada paso del descenso o ascenso por gradiente le suma a los parámetros un múltiplo de algunos ejemplos $fvec(x)^(\(i\))$. Si empezamos con todos los parámetros en cero, después de cualquier número de pasos los parámetros son una suma de múltiplos de los ejemplos. Es natural esperar algo parecido aquí.

La segunda es geométrica, y en este caso es una demostración. Supongamos que todos los ejemplos de entrenamiento están en una recta que pasa por el origen, como en la @fig-span. Podemos descomponer cualquier $fvec(w)$ en dos partes: $fvec(w)_parallel$, en la dirección de la recta, y $fvec(w)_perp$, perpendicular a ella. Como cada $fvec(x)^(\(i\))$ está sobre la recta, $fvec(w)_perp^T fvec(x)^(\(i\)) = 0$, y por lo tanto

$
  fvec(w)^T fvec(x)^(\(i\)) = fvec(w)_parallel^T fvec(x)^(\(i\)).
$

Es decir, $fvec(w)_perp$ no cambia ningún score de entrenamiento, y por lo tanto no cambia si las restricciones se cumplen. Pero sí aumenta la función objetivo, porque $norm(fvec(w))^2 = norm(fvec(w)_parallel)^2 + norm(fvec(w)_perp)^2$. Así que en el óptimo $fvec(w)_perp = 0$, y $fvec(w)$ está en la dirección de la recta.

#let u-span = (0.894, 0.447)
#let u-perp = (-0.447, 0.894)
#let t-neg = (-2.4, -1.8, -1.3)
#let t-pos = (1.0, 1.6, 2.3)
#let w-par = (0.8696 * 0.894, 0.8696 * 0.447)
#let b-span = 0.1304
#let w-tilt = (w-par.at(0) + 0.6 * u-perp.at(0), w-par.at(1) + 0.6 * u-perp.at(1))

#figure(
  caption: [Ejemplos que están todos sobre una recta que pasa por el origen (gris). Las dos fronteras clasifican exactamente igual a los ejemplos de entrenamiento, con los mismos scores. Pero la frontera inclinada (naranja) tiene un $fvec(w)$ con una componente perpendicular a la recta, que solo aumenta $norm(fvec(w))$ y por lo tanto reduce el margen geométrico.],
  [
    #panel-2d(
      x-min: -2.5,
      x-max: 2.5,
      y-min: -2,
      y-max: 2,
      size: (7, 5.6),
      {
        import cetz.draw: *
        plot.add(domain: (-2.5, 2.5), x => 0.5 * x, style: (stroke: (paint: gray, thickness: 0.8pt, dash: "dotted")))
        recta(w-par, b-span, dominio: (-2.5, 2.5))
        recta(w-tilt, b-span, dominio: (-2.5, 2.5), estilo: (paint: orange.darken(10%), thickness: 1.5pt, dash: "dashed"))
        puntos(t-neg.map(t => (t * u-span.at(0), t * u-span.at(1))), -1)
        puntos(t-pos.map(t => (t * u-span.at(0), t * u-span.at(1))), 1)
        let p0 = (-0.15 * u-span.at(0), -0.15 * u-span.at(1))
        vector(p0, (p0.at(0) + 1.4 * w-par.at(0), p0.at(1) + 1.4 * w-par.at(1)), label: $fvec(w)_parallel$, label-position: "head")
        vector(p0, (p0.at(0) + 1.4 * w-tilt.at(0), p0.at(1) + 1.4 * w-tilt.at(1)), color: orange.darken(10%), label: text(fill: orange.darken(10%))[$fvec(w)$], label-position: "head")
      },
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-span>

En general, los ejemplos no están en una recta, pero el argumento es el mismo: cualquier componente de $fvec(w)$ perpendicular a todos los $fvec(x)^(\(i\))$ no afecta las restricciones y aumenta $norm(fvec(w))^2$. En el óptimo, $fvec(w)$ queda en el espacio generado por los ejemplos de entrenamiento, que es lo mismo que decir que es una combinación lineal de ellos.

== Productos internos

Susituimos $fvec(w) = sum_i alpha_i y^(\(i\)) fvec(x)^(\(i\))$ en el problema de margen óptimo. La función objetivo queda

$
  1/2 norm(fvec(w))^2 = 1/2 fvec(w)^T fvec(w)
  = 1/2 (sum_(i=1)^m alpha_i y^(\(i\)) fvec(x)^(\(i\)))^T (sum_(j=1)^m alpha_j y^(\(j\)) fvec(x)^(\(j\)))
  = 1/2 sum_(i=1)^m sum_(j=1)^m alpha_i alpha_j y^(\(i\)) y^(\(j\)) chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r,
$

y cada restricción queda

$
  y^(\(i\)) (sum_(j=1)^m alpha_j y^(\(j\)) chevron.l fvec(x)^(\(j\)), fvec(x)^(\(i\)) chevron.r + b) >= 1.
$

Donde $chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r$ es el producto interno $(fvec(x)^(\(i\)))^T fvec(x)^(\(j\))$.

Ahora las incógnitas son los $alpha_i$ y $b$, y los datos aparecen únicamente a través de los productos internos $chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r$. Lo mismo pasa al predecir sobre un punto nuevo $fvec(x)$:

$
  fvec(w)^T fvec(x) + b = sum_(i=1)^m alpha_i y^(\(i\)) chevron.l fvec(x)^(\(i\)), fvec(x) chevron.r + b.
$

Usando la teoría de optimización con restricciones (dualidad de Lagrange), este problema se puede simplificar todavía más, y se obtiene el problema dual:

$
  max_alpha quad & W(alpha) = sum_(i=1)^m alpha_i - 1/2 sum_(i=1)^m sum_(j=1)^m y^(\(i\)) y^(\(j\)) alpha_i alpha_j chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r \
  "sujeto a" quad & alpha_i >= 0, quad i = 1, ..., m \
  & sum_(i=1)^m alpha_i y^(\(i\)) = 0.
$

== Dualidad de Lagrange

Dejemos de lado por un momento las SVM y veamos cómo resolver problemas de optimización con restricciones.

=== Multiplicadores de Lagrange

Consideremos primero un problema con restricciones de igualdad:

$
  min_fvec(w) quad & f(fvec(w)) \
  "sujeto a" quad & h_i (fvec(w)) = 0, quad i = 1, ..., l.
$

El método de los multiplicadores de Lagrange define el Lagrangiano

$
  cal(L)(fvec(w), beta) = f(fvec(w)) + sum_(i=1)^l beta_i h_i (fvec(w)),
$

donde los $beta_i$ se llaman multiplicadores de Lagrange. Luego se igualan a cero sus derivadas parciales,

$
  (partial cal(L)) / (partial w_i) = 0, quad quad (partial cal(L)) / (partial beta_i) = 0,
$

y se despejan $fvec(w)$ y $beta$. Por ejemplo, para minimizar $w_1^2 + w_2^2$ sujeto a $w_1 + w_2 = 1$:

$
  cal(L)(fvec(w), beta) = w_1^2 + w_2^2 + beta (w_1 + w_2 - 1).
$

Las derivadas dan $2 w_1 + beta = 0$, $2 w_2 + beta = 0$ y $w_1 + w_2 = 1$. De las dos primeras, $w_1 = w_2$, y con la tercera, $w_1 = w_2 = 1 slash 2$ (y $beta = -1$). Es el punto de la recta $w_1 + w_2 = 1$ más cercano al origen.

=== El problema primal

Ahora generalizamos a problemas que también tienen restricciones de desigualdad. Llamaremos a este el problema primal:

$
  min_fvec(w) quad & f(fvec(w)) \
  "sujeto a" quad & g_i (fvec(w)) <= 0, quad i = 1, ..., k \
  & h_i (fvec(w)) = 0, quad i = 1, ..., l.
$

Definimos el Lagrangiano generalizado

$
  cal(L)(fvec(w), alpha, beta) = f(fvec(w)) + sum_(i=1)^k alpha_i g_i (fvec(w)) + sum_(i=1)^l beta_i h_i (fvec(w)),
$

donde los $alpha_i$ y $beta_i$ son los multiplicadores de Lagrange. Consideremos la cantidad

$
  theta_cal(P) (fvec(w)) = max_(alpha, beta : alpha_i >= 0) cal(L)(fvec(w), alpha, beta).
$

Si $fvec(w)$ no cumple alguna restricción, por ejemplo $g_i (fvec(w)) > 0$, podemos hacer $alpha_i$ tan grande como queramos, y el máximo es $infinity$. Lo mismo pasa si $h_i (fvec(w)) != 0$: tomamos $beta_i$ muy grande con el mismo signo que $h_i (fvec(w))$. En cambio, si $fvec(w)$ cumple todas las restricciones, cada término $alpha_i g_i (fvec(w))$ es menor o igual a cero (lo mejor es tomar $alpha_i = 0$) y cada $h_i (fvec(w))$ es cero, así que el máximo es $f(fvec(w))$. Es decir,

$
  theta_cal(P) (fvec(w)) = cases(
    f(fvec(w)) quad & "si" fvec(w) "cumple las restricciones",
    infinity quad & "si no".
  )
$

Por lo tanto, minimizar $theta_cal(P)$ es exactamente el problema primal:

$
  p^* = min_fvec(w) theta_cal(P) (fvec(w)) = min_fvec(w) max_(alpha, beta : alpha_i >= 0) cal(L)(fvec(w), alpha, beta).
$

A $p^*$ se le llama el valor del problema primal.

=== El problema dual

Ahora intercambiamos el orden del mínimo y el máximo. Definimos

$
  theta_cal(D) (alpha, beta) = min_fvec(w) cal(L)(fvec(w), alpha, beta),
$

y el problema dual

$
  d^* = max_(alpha, beta : alpha_i >= 0) theta_cal(D) (alpha, beta) = max_(alpha, beta : alpha_i >= 0) min_fvec(w) cal(L)(fvec(w), alpha, beta).
$

Siempre se cumple que $d^* <= p^*$. Para verlo, tomemos cualquier $fvec(w)'$ y cualesquiera $alpha, beta$ con $alpha_i >= 0$:

$
  theta_cal(D) (alpha, beta) = min_fvec(w) cal(L)(fvec(w), alpha, beta) <= cal(L)(fvec(w)', alpha, beta) <= max_(alpha, beta : alpha_i >= 0) cal(L)(fvec(w)', alpha, beta) = theta_cal(P) (fvec(w)').
$

Como esto vale para todos los $alpha, beta$ y todos los $fvec(w)'$, el máximo del lado izquierdo es menor o igual al mínimo del lado derecho.

Bajo ciertas condiciones, además, $d^* = p^*$, y podemos resolver el dual en lugar del primal. Basta con que $f$ y las $g_i$ sean convexas, que las $h_i$ sean afines (de la forma $fvec(a)_i^T fvec(w) + c_i$), y que las restricciones $g_i$ se puedan cumplir de forma estricta, es decir, que exista algún $fvec(w)$ con $g_i (fvec(w)) < 0$ para todo $i$.

=== Condiciones KKT

Bajo esas condiciones, existen $fvec(w)^*$, $alpha^*$ y $beta^*$ tales que $fvec(w)^*$ es solución del primal, $(alpha^*, beta^*)$ es solución del dual, y $p^* = d^* = cal(L)(fvec(w)^*, alpha^*, beta^*)$. Además, cumplen las condiciones de Karush-Kuhn-Tucker (KKT):

$
  (partial cal(L)) / (partial w_i) (fvec(w)^*, alpha^*, beta^*) & = 0, quad i = 1, ..., d \
  (partial cal(L)) / (partial beta_i) (fvec(w)^*, alpha^*, beta^*) & = 0, quad i = 1, ..., l \
  alpha_i^* g_i (fvec(w)^*) & = 0, quad i = 1, ..., k \
  g_i (fvec(w)^*) & <= 0, quad i = 1, ..., k \
  alpha_i^* & >= 0, quad i = 1, ..., k.
$

Y al revés: si unos $fvec(w)^*, alpha^*, beta^*$ cumplen las condiciones KKT, entonces son soluciones del primal y del dual.

La tercera condición, $alpha_i^* g_i (fvec(w)^*) = 0$, se llama complementariedad. Dice que, para cada restricción, o su multiplicador es cero o la restricción se cumple con igualdad. Es decir, si $alpha_i^* > 0$, entonces $g_i (fvec(w)^*) = 0$. Se dice que la restricción está activa. Esta condición es la que explicará los vectores de soporte.

== El dual del clasificador de margen óptimo

Volvamos al problema de margen óptimo:

$
  min_(fvec(w), b) quad & 1/2 norm(fvec(w))^2 \
  "sujeto a" quad & y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) >= 1, quad i = 1, ..., m.
$

Escribimos cada restricción en la forma $g_i <= 0$:

$
  g_i (fvec(w), b) = -y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) + 1 <= 0.
$

No hay restricciones de igualdad, así que solo hay multiplicadores $alpha_i$. El Lagrangiano es

$
  cal(L)(fvec(w), b, alpha) = 1/2 norm(fvec(w))^2 - sum_(i=1)^m alpha_i [y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) - 1].
$

Para encontrar $theta_cal(D)$, minimizamos $cal(L)$ respecto a $fvec(w)$ y $b$ con $alpha$ fijo, igualando las derivadas a cero:

$
  gradient_fvec(w) cal(L)(fvec(w), b, alpha) = fvec(w) - sum_(i=1)^m alpha_i y^(\(i\)) fvec(x)^(\(i\)) = 0,
$

de donde

$
  fvec(w) = sum_(i=1)^m alpha_i y^(\(i\)) fvec(x)^(\(i\)).
$

Esta es exactamente la forma que habíamos anticipado, y ahora sabemos qué son los $alpha_i$: los multiplicadores de Lagrange de cada restricción. Por eso son no negativos. La derivada respecto a $b$ da

$
  (partial cal(L)) / (partial b) = -sum_(i=1)^m alpha_i y^(\(i\)) = 0.
$

Sustituyamos $fvec(w)$ en el Lagrangiano. El primer término es el que calculamos antes:

$
  1/2 norm(fvec(w))^2 = 1/2 sum_(i,j=1)^m y^(\(i\)) y^(\(j\)) alpha_i alpha_j chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r.
$

El segundo término es

$
  sum_(i=1)^m alpha_i [y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) - 1]
  = sum_(i,j=1)^m y^(\(i\)) y^(\(j\)) alpha_i alpha_j chevron.l fvec(x)^(\(j\)), fvec(x)^(\(i\)) chevron.r + b sum_(i=1)^m alpha_i y^(\(i\)) - sum_(i=1)^m alpha_i.
$

Restando ambos:

$
  cal(L)(fvec(w), b, alpha) = sum_(i=1)^m alpha_i - 1/2 sum_(i,j=1)^m y^(\(i\)) y^(\(j\)) alpha_i alpha_j chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r - b sum_(i=1)^m alpha_i y^(\(i\)).
$

Por la condición de la derivada respecto a $b$, el último término es cero. Juntando esto con $alpha_i >= 0$ y con $sum_i alpha_i y^(\(i\)) = 0$, obtenemos el problema dual:

$
  max_alpha quad & W(alpha) = sum_(i=1)^m alpha_i - 1/2 sum_(i,j=1)^m y^(\(i\)) y^(\(j\)) alpha_i alpha_j chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r \
  "sujeto a" quad & alpha_i >= 0, quad i = 1, ..., m \
  & sum_(i=1)^m alpha_i y^(\(i\)) = 0.
$

Las condiciones para que $p^* = d^*$ se cumplen. La función objetivo $1/2 norm(fvec(w))^2$ es convexa, y las $g_i$ son afines (y por lo tanto convexas). Y si los datos son separables, las restricciones se pueden cumplir de forma estricta: basta tomar una solución con margen funcional 1 y multiplicar $fvec(w)$ y $b$ por 2, para que todos los márgenes funcionales sean al menos $2 > 1$.

El dual es un problema cuadrático en las $m$ variables $alpha_i$. Para resolverlo existen algoritmos especializados, como el algoritmo SMO (sequential minimal optimization), que es el que usa scikit-learn. No los veremos en detalle.

== Vectores de soporte

La condición de complementariedad dice que $alpha_i > 0$ solo si $g_i (fvec(w)) = 0$, es decir, solo si

$
  y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) = 1.
$

Los únicos ejemplos con $alpha_i > 0$ son los que tienen margen funcional exactamente 1: los que están sobre las rectas punteadas de la @fig-margen-optimo. A estos ejemplos se les llama vectores de soporte. En la figura hay #d-sep.opt.n_sv de #(2 * d-sep.neg.len()) ejemplos. Todos los demás tienen $alpha_i = 0$ y no aparecen en $fvec(w) = sum_i alpha_i y^(\(i\)) fvec(x)^(\(i\))$: podríamos moverlos o eliminarlos (sin cruzar el margen) y la frontera no cambiaría. El nombre viene de que los vectores de soporte son los que "sostienen" la frontera.

Una vez resuelto el dual, obtenemos $fvec(w)$ con la fórmula anterior. Para obtener $b$, tomamos cualquier vector de soporte $fvec(x)^(\(s\))$. Como su margen funcional es 1, $y^(\(s\)) (fvec(w)^T fvec(x)^(\(s\)) + b) = 1$, y como $y^(\(s\)) = plus.minus 1$, multiplicando ambos lados por $y^(\(s\))$:

$
  b = y^(\(s\)) - fvec(w)^T fvec(x)^(\(s\)).
$

En la práctica se promedia este valor sobre todos los vectores de soporte, para reducir errores numéricos.

Para predecir sobre un punto nuevo $fvec(x)$ calculamos

$
  fvec(w)^T fvec(x) + b = sum_(i=1)^m alpha_i y^(\(i\)) chevron.l fvec(x)^(\(i\)), fvec(x) chevron.r + b,
$

y predecimos $y = 1$ si es positivo. Como $alpha_i = 0$ para todos los ejemplos que no son vectores de soporte, la suma solo tiene que recorrer los vectores de soporte, que suelen ser pocos.

En resumen: tanto el entrenamiento (el problema dual) como la predicción dependen de los datos solo a través de productos internos entre ejemplos. Esto es lo que vamos a explotar ahora.

= Parte 4: Kernels

== La receta

En el pasado vimos que una transformación de features $phi(fvec(x))$ permite obtener fronteras no lineales con un modelo lineal, pero que $phi(fvec(x))$ puede tener tantas componentes que calcularlo es impráctico. La idea de los kernels sigue cuatro pasos:

+ Escribir el algoritmo de forma que los datos solo aparezcan en productos internos $chevron.l fvec(x), fvec(z) chevron.r$. Para las SVM, eso es lo que acabamos de hacer.
+ Escoger una transformación $phi$ que lleve las $d$ features originales a un espacio de dimensión mucho mayor.
+ Encontrar una forma eficiente de calcular $K(fvec(x), fvec(z)) = phi(fvec(x))^T phi(fvec(z))$ sin calcular $phi(fvec(x))$ ni $phi(fvec(z))$. A $K$ se le llama kernel.
+ Reemplazar cada $chevron.l fvec(x), fvec(z) chevron.r$ del algoritmo por $K(fvec(x), fvec(z))$.

El resultado es el mismo algoritmo, pero trabajando con las features $phi(fvec(x))$, y sin haberlas calculado nunca. El paso 3 parece mágico, así que veamos ejemplos.

== Kernels polinomiales

Tomemos $d = 3$, es decir, $fvec(x) = (x_1, x_2, x_3)$, y la transformación que contiene todos los productos de dos features:

$
  phi(fvec(x)) = (x_1 x_1, x_1 x_2, x_1 x_3, x_2 x_1, x_2 x_2, x_2 x_3, x_3 x_1, x_3 x_2, x_3 x_3).
$

Calcular $phi(fvec(x))$ requiere 9 multiplicaciones, y en general $d^2$. El producto interno de dos de estos vectores es

$
  phi(fvec(x))^T phi(fvec(z)) = sum_(i=1)^3 sum_(j=1)^3 (x_i x_j)(z_i z_j).
$

Ahora calculemos $(fvec(x)^T fvec(z))^2$:

$
  (fvec(x)^T fvec(z))^2
  = (sum_(i=1)^3 x_i z_i) (sum_(j=1)^3 x_j z_j)
  = sum_(i=1)^3 sum_(j=1)^3 x_i z_i x_j z_j
  = sum_(i=1)^3 sum_(j=1)^3 (x_i x_j)(z_i z_j).
$

Es lo mismo. Entonces

$
  K(fvec(x), fvec(z)) = (fvec(x)^T fvec(z))^2
$

es el kernel correspondiente a esta $phi$. Calcular $phi(fvec(x))^T phi(fvec(z))$ directamente cuesta del orden de $d^2$ operaciones. Calcular $(fvec(x)^T fvec(z))^2$ cuesta del orden de $d$: un producto interno en $RR^d$ y un cuadrado. Con $d = 10 000$, la diferencia es entre $10^8$ y $10^4$ operaciones.

Un ejemplo parecido es

$
  K(fvec(x), fvec(z)) = (fvec(x)^T fvec(z) + c)^2 = (fvec(x)^T fvec(z))^2 + 2 c fvec(x)^T fvec(z) + c^2 = sum_(i,j=1)^d (x_i x_j)(z_i z_j) + sum_(i=1)^d (sqrt(2c) x_i)(sqrt(2c) z_i) + c dot c,
$

con $c >= 0$. Corresponde a una $phi$ que contiene todos los productos $x_i x_j$, las features originales multiplicadas por $sqrt(2c)$, y la constante $c$. Para $d = 3$:

$
  phi(fvec(x)) = (x_1 x_1, x_1 x_2, ..., x_3 x_3, sqrt(2c) x_1, sqrt(2c) x_2, sqrt(2c) x_3, c).
$

El parámetro $c$ controla el peso relativo entre los términos de primer y segundo grado.

Más en general, el kernel polinomial

$
  K(fvec(x), fvec(z)) = (fvec(x)^T fvec(z) + c)^k
$

corresponde a una $phi$ con todos los monomios de grado hasta $k$, que son $binom(d + k, k)$. Con $d = 1000$ y $k = 3$, eso es más de $10^8$ features. Pero calcular $K$ sigue costando del orden de $d$ operaciones.

#let neg-1d = (-2.6, -2.1, -1.7, 1.6, 2.0, 2.5)
#let pos-1d = (-0.9, -0.5, -0.1, 0.3, 0.8)

#figure(
  caption: [A la izquierda, datos con una sola feature que ninguna frontera lineal (un punto) puede separar. A la derecha, los mismos datos transformados con $phi(x) = (x, x^2)$, que corresponde al kernel $K(x, z) = x z + x^2 z^2$. En el espacio transformado, la recta horizontal $x^2 = 1.6$ los separa. En el espacio original, esa frontera corresponde a los dos puntos $x = plus.minus sqrt(1.6)$.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      custom-plot(
        x-min: -3,
        x-max: 3,
        y-min: -1,
        y-max: 1,
        x-label: $x$,
        y-label: [],
        x-tick-step: 1,
        y-tick-step: none,
        size: (5.5, 2),
        {
          puntos(neg-1d.map(x => (x, 0)), -1)
          puntos(pos-1d.map(x => (x, 0)), 1)
        },
      ),
      custom-plot(
        x-min: -3,
        x-max: 3,
        y-min: 0,
        y-max: 7,
        x-label: $x$,
        y-label: $x^2$,
        x-tick-step: 1,
        y-tick-step: 1,
        size: (5.5, 5),
        {
          plot.add(domain: (-3, 3), x => x * x, style: (stroke: (paint: gray, thickness: 0.6pt, dash: "dotted")))
          plot.add-hline(1.6, style: (stroke: (paint: black, thickness: 1.5pt)))
          puntos(neg-1d.map(x => (x, x * x)), -1)
          puntos(pos-1d.map(x => (x, x * x)), 1)
        },
      ),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-levantar>

La @fig-levantar muestra la idea geométrica. Al llevar los datos a un espacio de más dimensiones, problemas que no se podían separar con una frontera lineal se vuelven separables. La SVM encuentra la frontera de margen máximo en ese espacio, y esa frontera, vista en el espacio original, ya no es lineal.

== Kernels como medida de similitud

Hay otra forma de pensar en los kernels. Si $phi(fvec(x))$ y $phi(fvec(z))$ apuntan en direcciones parecidas, su producto interno $K(fvec(x), fvec(z))$ es grande. Si son casi perpendiculares, es cercano a cero. Así, $K(fvec(x), fvec(z))$ se puede ver como una medida de qué tan parecidos son $fvec(x)$ y $fvec(z)$.

Esta interpretación sugiere inventar kernels directamente como medidas de similitud. El más usado es el kernel gaussiano:

$
  K(fvec(x), fvec(z)) = exp(-norm(fvec(x) - fvec(z))^2 / (2 sigma^2)).
$

Vale 1 cuando $fvec(x) = fvec(z)$ y se acerca a 0 a medida que los puntos se alejan. El parámetro $sigma$ dice a qué distancia los puntos dejan de considerarse parecidos (@fig-similitud). También se le llama kernel RBF (radial basis function).

#let colores-sigma = (c-neg, green.darken(20%), c-pos)

#figure(
  caption: [Kernel gaussiano en función de la distancia entre $fvec(x)$ y $fvec(z)$, para $sigma = 0.5$, 1 y 2. Con $sigma$ pequeño, solo los puntos muy cercanos se consideran parecidos.],
  [
    #custom-plot(
      x-min: 0,
      x-max: 5,
      y-min: 0,
      y-max: 1,
      x-label: $norm(fvec(x) - fvec(z))$,
      y-label: $K(fvec(x), fvec(z))$,
      x-tick-step: 1,
      y-tick-step: 0.25,
      size: (7, 4),
      {
        for (k, s) in (0.5, 1, 2).enumerate() {
          plot.add(
            domain: (0, 5),
            samples: 100,
            r => calc.exp(-r * r / (2 * s * s)),
            style: (stroke: (paint: colores-sigma.at(k), thickness: 1.6pt)),
          )
        }
      },
    )
    #v(0.2cm)
    #set text(size: 9pt)
    #grid(
      columns: 3,
      column-gutter: 14pt,
      ..range(3).map(k => box[
        #box(width: 0.6cm, height: 0pt, baseline: -0.3em, stroke: (top: (paint: colores-sigma.at(k), thickness: 1.6pt)))
        $sigma = #(0.5, 1, 2).at(k)$
      ])
    )
  ],
) <fig-similitud>

¿A qué $phi$ corresponde este kernel? Veámoslo con una sola feature, $x, z in RR$. Desarrollando el cuadrado, $(x - z)^2 = x^2 - 2 x z + z^2$, así que

$
  K(x, z) = exp(-x^2 / (2 sigma^2)) exp(-z^2 / (2 sigma^2)) exp((x z) / sigma^2).
$

Usando la serie de Taylor $e^t = sum_(n=0)^infinity t^n slash n!$ en el último factor:

$
  exp((x z) / sigma^2) = sum_(n=0)^infinity (x z)^n / (sigma^(2n) n!) = sum_(n=0)^infinity x^n / (sigma^n sqrt(n!)) dot z^n / (sigma^n sqrt(n!)).
$

Entonces $K(x, z) = sum_(n=0)^infinity phi_n (x) phi_n (z)$, con

$
  phi_n (x) = exp(-x^2 / (2 sigma^2)) x^n / (sigma^n sqrt(n!)), quad n = 0, 1, 2, ...
$

Es decir, el kernel gaussiano es el producto interno de un vector $phi(x)$ con infinitas componentes, que contiene todas las potencias de $x$. Ninguna computadora puede guardar ese vector, pero calcular $K(x, z)$ es trivial.

== ¿Qué funciones son kernels válidos?

No cualquier medida de similitud sirve. Para que el algoritmo tenga sentido, $K$ tiene que ser el producto interno de alguna transformación $phi$, aunque no la conozcamos explícitamente. ¿Cómo saber si una función $K$ cumple esto?

Supongamos que $K$ es un kernel válido, es decir, $K(fvec(x), fvec(z)) = phi(fvec(x))^T phi(fvec(z))$ para alguna $phi$. Tomemos cualquier conjunto finito de puntos $fvec(x)^(\(1\)), ..., fvec(x)^(\(n\))$ y formemos la matriz $n times n$ con entradas

$
  K_(i j) = K(fvec(x)^(\(i\)), fvec(x)^(\(j\))).
$

A esta matriz se le llama matriz kernel. (Usamos la misma letra para la función y para la matriz porque están muy relacionadas.)

La matriz kernel es simétrica, porque $K_(i j) = phi(fvec(x)^(\(i\)))^T phi(fvec(x)^(\(j\))) = phi(fvec(x)^(\(j\)))^T phi(fvec(x)^(\(i\))) = K_(j i)$. Además, si $phi_k (fvec(x))$ es la componente $k$ de $phi(fvec(x))$, para cualquier vector $fvec(z) in RR^n$:

$
  fvec(z)^T K fvec(z)
  &= sum_i sum_j z_i K_(i j) z_j \
  &= sum_i sum_j z_i phi(fvec(x)^(\(i\)))^T phi(fvec(x)^(\(j\))) z_j \
  &= sum_i sum_j z_i sum_k phi_k (fvec(x)^(\(i\))) phi_k (fvec(x)^(\(j\))) z_j \
  &= sum_k sum_i sum_j z_i phi_k (fvec(x)^(\(i\))) phi_k (fvec(x)^(\(j\))) z_j \
  &= sum_k (sum_i z_i phi_k (fvec(x)^(\(i\))))^2 \
  &>= 0.
$

En el penúltimo paso usamos que $sum_(i,j) a_i a_j = (sum_i a_i)^2$, con $a_i = z_i phi_k (fvec(x)^(\(i\)))$. Como $fvec(z)$ es arbitrario, la matriz kernel es semidefinida positiva.

Es decir, si $K$ es un kernel válido, entonces su matriz kernel es simétrica y semidefinida positiva para cualquier conjunto de puntos. Resulta que lo contrario también es cierto:

*Teorema (Mercer).* Sea $K : RR^d times RR^d -> RR$. Para que $K$ sea un kernel válido, es necesario y suficiente que, para cualquier conjunto finito de puntos $fvec(x)^(\(1\)), ..., fvec(x)^(\(n\))$, la matriz kernel correspondiente sea simétrica y semidefinida positiva.

No demostraremos la otra dirección. El teorema da una forma de verificar que una función es un kernel válido sin tener que encontrar $phi$.

Por ejemplo, $K(fvec(x), fvec(z)) = norm(fvec(x) - fvec(z))^2$ parece una medida razonable de diferencia, pero no es un kernel válido. Con dos puntos distintos a distancia $a$, la matriz kernel es

$
  mat(0, a^2; a^2, 0),
$

y con $fvec(z) = (1, -1)$ obtenemos $fvec(z)^T K fvec(z) = -2 a^2 < 0$.

== Máquinas de soporte vectorial

Ya tenemos todas las piezas. Una máquina de soporte vectorial es el clasificador de margen óptimo con un kernel. Siguiendo la receta, reemplazamos los productos internos del problema dual por $K$:

$
  max_alpha quad & W(alpha) = sum_(i=1)^m alpha_i - 1/2 sum_(i,j=1)^m y^(\(i\)) y^(\(j\)) alpha_i alpha_j K(fvec(x)^(\(i\)), fvec(x)^(\(j\))) \
  "sujeto a" quad & alpha_i >= 0, quad i = 1, ..., m \
  & sum_(i=1)^m alpha_i y^(\(i\)) = 0,
$

y para predecir sobre un punto nuevo calculamos

$
  fvec(w)^T phi(fvec(x)) + b = sum_(i=1)^m alpha_i y^(\(i\)) K(fvec(x)^(\(i\)), fvec(x)) + b.
$

El vector $fvec(w) = sum_i alpha_i y^(\(i\)) phi(fvec(x)^(\(i\)))$ vive en el espacio de las features $phi$, que puede tener dimensión infinita, así que nunca lo calculamos: solo guardamos los $alpha_i$ de los vectores de soporte. El parámetro $b$ se obtiene igual que antes, a partir de un vector de soporte: $b = y^(\(s\)) - sum_i alpha_i y^(\(i\)) K(fvec(x)^(\(i\)), fvec(x)^(\(s\)))$.

La @fig-gaussiano muestra una SVM con kernel gaussiano para tres valores de $sigma$. El parámetro $sigma$ controla la complejidad del modelo. Con $sigma$ grande, todos los puntos se parecen entre sí y la frontera es casi lineal. Con $sigma$ muy pequeño, cada ejemplo solo se parece a sí mismo, la frontera rodea ejemplos individuales y casi todos los ejemplos se vuelven vectores de soporte: overfitting. Igual que el grado de un polinomio o $lambda$, $sigma$ se escoge con validación.

#let panel-gauss(p) = custom-plot(
  x-min: -1.6,
  x-max: 2.6,
  y-min: -1.2,
  y-max: 1.7,
  x-label: $x_1$,
  y-label: $x_2$,
  x-tick-step: 1,
  y-tick-step: 1,
  size: (4.5, 3.6),
  {
    for s in p.frontera {
      plot.add(s, style: (stroke: (paint: black, thickness: 1.3pt)))
    }
    puntos(d-gauss.neg, -1, tam: 0.09)
    puntos(d-gauss.pos, 1, tam: 0.09)
    soporte(p.sv, tam: 0.15)
  },
)

#figure(
  caption: [SVM con kernel gaussiano sobre #d-gauss.m ejemplos, para $sigma = #d-gauss.paneles.grande.sigma$, #d-gauss.paneles.medio.sigma y #d-gauss.paneles.pequeno.sigma (de izquierda a derecha), con $C = 10$ (ver la Parte 5). Los círculos marcan los vectores de soporte: #d-gauss.paneles.grande.n_sv, #d-gauss.paneles.medio.n_sv y #d-gauss.paneles.pequeno.n_sv. La exactitud sobre datos de prueba es #calc.round(d-gauss.paneles.grande.exactitud_test * 100)%, #calc.round(d-gauss.paneles.medio.exactitud_test * 100)% y #calc.round(d-gauss.paneles.pequeno.exactitud_test * 100)%, mientras que la de entrenamiento es #calc.round(d-gauss.paneles.grande.exactitud_train * 100)%, #calc.round(d-gauss.paneles.medio.exactitud_train * 100)% y #calc.round(d-gauss.paneles.pequeno.exactitud_train * 100)%.],
  [
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 0.3cm,
      panel-gauss(d-gauss.paneles.grande),
      panel-gauss(d-gauss.paneles.medio),
      panel-gauss(d-gauss.paneles.pequeno),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-gaussiano>

La idea de los kernels no es exclusiva de las SVM. Cualquier algoritmo que se pueda escribir usando solo productos internos entre ejemplos se puede "kernelizar" reemplazándolos por $K$. Por ejemplo, la regresión lineal entrenada con descenso por gradiente, o el perceptrón, que veremos más adelante.

= Parte 5: Margen suave

== El problema de los outliers

Hasta ahora supusimos que los datos son linealmente separables, ya sea en el espacio original o en el espacio de $phi$. Eso no siempre ocurre. Y aunque ocurra, separar perfectamente los datos no siempre es buena idea. En la @fig-outlier, un solo ejemplo atípico (outlier) hace que la frontera gire bruscamente y que el margen se reduzca a una fracción del original.

#let panel-out(p, con-outlier: true) = panel-2d(size: (4.3, 4.3), {
  margenes(p.w, p.b)
  recta(p.w, p.b)
  puntos-clases(d-out, tam: 0.11)
  if con-outlier {
    puntos((d-out.outlier,), -1, tam: 0.11)
  }
  soporte(p.sv, tam: 0.24)
})

#figure(
  caption: [A la izquierda, el clasificador de margen óptimo. Al centro, el mismo después de agregar un ejemplo negativo atípico: la frontera gira y el ancho del margen ($2 slash norm(fvec(w))$) pasa de #num(d-out.paneles.sin_outlier.ancho) a #num(d-out.paneles.duro.ancho). A la derecha, la SVM de margen suave con $C = 0.5$ con los mismos datos: acepta clasificar mal el outlier a cambio de un margen de ancho #num(d-out.paneles.suave.ancho).],
  [
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 0.3cm,
      panel-out(d-out.paneles.sin_outlier, con-outlier: false),
      panel-out(d-out.paneles.duro),
      panel-out(d-out.paneles.suave),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-outlier>

== Formulación

Para que el algoritmo funcione con datos no separables y sea menos sensible a outliers, permitimos que algunos ejemplos tengan margen funcional menor que 1, pero con un costo. Introducimos una variable de holgura $xi_i >= 0$ por cada ejemplo:

$
  min_(fvec(w), b, xi) quad & 1/2 norm(fvec(w))^2 + C sum_(i=1)^m xi_i \
  "sujeto a" quad & y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) >= 1 - xi_i, quad i = 1, ..., m \
  & xi_i >= 0, quad i = 1, ..., m.
$

Ahora un ejemplo puede tener margen funcional $1 - xi_i$, y por ello la función objetivo aumenta en $C xi_i$. El valor de $xi_i$ indica qué tanto viola el ejemplo el margen:

- $xi_i = 0$: el ejemplo está fuera del margen, del lado correcto.
- $0 < xi_i <= 1$: el ejemplo está dentro del margen, pero del lado correcto de la frontera.
- $xi_i > 1$: el ejemplo está del lado equivocado de la frontera, es decir, mal clasificado.

A esta versión se le llama SVM de margen suave, en contraste con la de margen duro de la Parte 2. Como $sum xi_i$ es una suma de valores no negativos, es una penalización de tipo $L_1$ sobre las holguras.

El parámetro $C > 0$ controla el balance entre dos objetivos: que $norm(fvec(w))^2$ sea pequeño (margen grande) y que pocos ejemplos violen el margen. Con $C$ grande, cualquier violación es muy costosa y nos acercamos al margen duro. Con $C$ pequeño, el modelo prefiere un margen amplio aunque varios ejemplos queden dentro de él o mal clasificados. Si dividimos la función objetivo entre $C$, queda $sum xi_i + 1/(2C) norm(fvec(w))^2$, así que $1 slash C$ hace el papel del $lambda$ de la regularización: un $C$ pequeño significa mucha regularización. Igual que $lambda$, $C$ se escoge con validación.

#let panel-c(p) = panel-2d({
  margenes(p.w, p.b)
  recta(p.w, p.b)
  puntos-clases(d-c, tam: 0.1)
  soporte(p.sv, tam: 0.22)
})

#figure(
  caption: [SVM lineal de margen suave sobre #d-c.m ejemplos que se solapan, con $C = #d-c.paneles.pequeno.C$ (izquierda) y $C = #d-c.paneles.grande.C$ (derecha). Con $C$ pequeño el margen es más ancho (#num(d-c.paneles.pequeno.ancho) contra #num(d-c.paneles.grande.ancho)) y hay más vectores de soporte (#d-c.paneles.pequeno.n_sv contra #d-c.paneles.grande.n_sv), porque más ejemplos quedan dentro del margen.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      panel-c(d-c.paneles.pequeno),
      panel-c(d-c.paneles.grande),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-efecto-c>

== El problema dual

Igual que antes, formamos el Lagrangiano, ahora con multiplicadores $alpha_i >= 0$ para las restricciones de margen y $r_i >= 0$ para las restricciones $xi_i >= 0$:

$
  cal(L)(fvec(w), b, xi, alpha, r) = 1/2 fvec(w)^T fvec(w) + C sum_(i=1)^m xi_i - sum_(i=1)^m alpha_i [y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) - 1 + xi_i] - sum_(i=1)^m r_i xi_i.
$

Las derivadas respecto a $fvec(w)$ y $b$ son las mismas de antes, y dan otra vez $fvec(w) = sum_i alpha_i y^(\(i\)) fvec(x)^(\(i\))$ y $sum_i alpha_i y^(\(i\)) = 0$. Lo único nuevo es la derivada respecto a $xi_i$:

$
  (partial cal(L)) / (partial xi_i) = C - alpha_i - r_i = 0.
$

Como $r_i >= 0$, esto implica que $alpha_i <= C$. Al sustituir en el Lagrangiano, los términos con $xi_i$ se cancelan, y se obtiene el problema dual:

$
  max_alpha quad & W(alpha) = sum_(i=1)^m alpha_i - 1/2 sum_(i,j=1)^m y^(\(i\)) y^(\(j\)) alpha_i alpha_j chevron.l fvec(x)^(\(i\)), fvec(x)^(\(j\)) chevron.r \
  "sujeto a" quad & 0 <= alpha_i <= C, quad i = 1, ..., m \
  & sum_(i=1)^m alpha_i y^(\(i\)) = 0.
$

Sorprendentemente, el único cambio respecto al margen duro es que ahora los $alpha_i$ están acotados por $C$. Todo lo demás sigue igual: la predicción usa la misma fórmula, solo hay productos internos, y por lo tanto podemos usar kernels.

Las condiciones de complementariedad dan una clasificación de los ejemplos:

$
  alpha_i = 0 & quad => quad y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) >= 1 \
  0 < alpha_i < C & quad => quad y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) = 1 \
  alpha_i = C & quad => quad y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) <= 1.
$

Los ejemplos con $alpha_i = 0$ están fuera del margen y no influyen en la frontera. Los que tienen $0 < alpha_i < C$ están exactamente sobre el margen, como los vectores de soporte del margen duro. Los que tienen $alpha_i = C$ están dentro del margen o mal clasificados. Los vectores de soporte son ahora todos los ejemplos con $alpha_i > 0$. Para calcular $b$ se usan los que tienen $0 < alpha_i < C$, que son los que están exactamente sobre el margen.

== Aplicaciones

Las SVM con kernels fueron durante muchos años uno de los mejores algoritmos de clasificación "listos para usar". Dos ejemplos:

- En el reconocimiento de dígitos escritos a mano, con imágenes de $16 times 16$ píxeles, una SVM con kernel polinomial o gaussiano aplicada directamente a las 256 intensidades de los píxeles obtuvo resultados excelentes, sin ningún conocimiento sobre imágenes, ni siquiera sobre qué píxeles son vecinos.
- Los kernels permiten clasificar objetos que no son vectores. Por ejemplo, para clasificar proteínas, que son secuencias de aminoácidos de longitud variable, se puede definir $phi(fvec(x))$ como el número de veces que aparece cada secuencia de 4 aminoácidos. Hay 20 aminoácidos, así que $phi(fvec(x))$ tiene $20^4 = 160 000$ componentes. Con algoritmos de búsqueda de cadenas es posible calcular $K(fvec(x), fvec(z)) = phi(fvec(x))^T phi(fvec(z))$ de forma eficiente sin construir $phi$.

En scikit-learn, la clase `SVC` implementa la SVM de margen suave. Sus parámetros principales son `C`, el `kernel` (`"linear"`, `"poly"`, `"rbf"`) y, para el kernel gaussiano, `gamma`, que corresponde a $1 slash (2 sigma^2)$: un `gamma` grande es un $sigma$ pequeño. Como el kernel gaussiano usa distancias entre puntos, es importante estandarizar las features antes de entrenar. Después del entrenamiento, el atributo `support_vectors_` contiene los vectores de soporte y `dual_coef_` contiene los productos $alpha_i y^(\(i\))$.

= Resumen

- El margen funcional $y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b)$ es positivo cuando un ejemplo está bien clasificado. El margen geométrico es el margen funcional dividido entre $norm(fvec(w))$, es la distancia del ejemplo a la frontera y no depende de la escala de $fvec(w)$ y $b$.
- El clasificador de margen óptimo busca la frontera que maximiza el margen geométrico del peor ejemplo. Escogiendo la escala para que el margen funcional sea 1, el problema es minimizar $1/2 norm(fvec(w))^2$ sujeto a $y^(\(i\)) (fvec(w)^T fvec(x)^(\(i\)) + b) >= 1$.
- Con dualidad de Lagrange se obtiene el problema dual, en el que $fvec(w) = sum_i alpha_i y^(\(i\)) fvec(x)^(\(i\))$ y los datos solo aparecen en productos internos. Por las condiciones KKT, solo los vectores de soporte tienen $alpha_i > 0$.
- Un kernel es una función $K(fvec(x), fvec(z)) = phi(fvec(x))^T phi(fvec(z))$ que se calcula sin construir $phi$. Reemplazando los productos internos por $K$, la SVM trabaja en espacios de dimensión muy grande o infinita. Por el teorema de Mercer, $K$ es válido si y solo si sus matrices kernel son simétricas y semidefinidas positivas.
- La SVM de margen suave agrega holguras $xi_i$ con un costo $C sum xi_i$. En el dual, el único cambio es $0 <= alpha_i <= C$. El parámetro $C$ hace el papel de $1 slash lambda$, y junto con el kernel y sus parámetros se escoge con validación.
