// OUTLINE:
// 1. Clasificación lineal
// 1.1. Regresión lineal sobre etiquetas 0/1
// 1.2. Score, regla de decisión y frontera
// 1.3. Geometría de la frontera
// 2. Regresión logística
// 2.1. La función sigmoide
// 2.2. Un modelo de probabilidad para y
// 3. Máxima verosimilitud
// 3.1. Verosimilitud y log-verosimilitud
// 3.2. Lo que aporta cada ejemplo
// 3.3. Comparación con regresión lineal
// 4. Entrenamiento
// 4.1. El gradiente
// 4.2. Ascenso por gradiente
// 4.3. Datos separables y regularización
// 5. Transformaciones de features
// 5.1. Fronteras no lineales con un modelo lineal
// 5.2. El costo de muchas features
// 5.3. Hacia SVM y kernels
//
// Los datos de las figuras se generan con `python generar_figuras.py`
// y se leen desde la carpeta figs/.

#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *

#let fvec(x) = math.bold(math.upright(x))
// En Typst, `phi` es la phi curva (\varphi de LaTeX); usamos la recta, como en CS229
#let phi = sym.phi.alt

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
  title: [Clasificación lineal y regresión logística],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

// Datos de las figuras
#let d-reg = json("figs/regresion_falla.json")
#let d-front = json("figs/frontera.json")
#let d-log = json("figs/logistica.json")
#let d-desc = json("figs/ascenso.json")
#let d-par = json("figs/parabola.json")

#let num(x, digits: 2) = str(calc.round(x, digits: digits))

// Estilo de las clases (mismo en todas las figuras)
#let c0 = rgb("#1f77b4")
#let c1 = rgb("#d62728")

#let puntos-clase(datos, clase, tam: 0.13) = plot.add(
  datos,
  mark: if clase == 0 { "o" } else { "triangle" },
  mark-size: tam,
  mark-style: (stroke: none, fill: if clase == 0 { c0 } else { c1 }),
  style: (stroke: none),
)

// Recta θ_0 + θ_1 x_1 + θ_2 x_2 = c en el plano (x_1, x_2)
#let recta(theta, c: 0, dominio: (0, 5), estilo: (paint: black, thickness: 1.5pt)) = plot.add(
  domain: dominio,
  x1 => (c - theta.at(0) - theta.at(1) * x1) / theta.at(2),
  style: (stroke: estilo),
)

#let sigmoide(z) = 1 / (1 + calc.exp(-z))

#let leyenda-clases = {
  set text(size: 9pt)
  grid(
    columns: 2,
    column-gutter: 16pt,
    box[#text(fill: c0)[●] $y = 0$],
    box[#text(fill: c1)[▲] $y = 1$],
  )
}

Hasta ahora hemos trabajado con problemas de regresión, donde la etiqueta $y$ es un número real. En estas notas pasamos a problemas de clasificación binaria, donde $y$ solo puede tomar dos valores, que representamos como $y in {0, 1}$. Por ejemplo, $y = 1$ si un correo es spam y $y = 0$ si no lo es. A la clase $y = 1$ se le llama clase positiva y a $y = 0$ clase negativa.

La idea central es que podemos reutilizar casi todo lo que ya sabemos de regresión lineal: el modelo sigue usando $theta^T fvec(x)$, los parámetros se escogen por máxima verosimilitud, y se entrenan con un método de gradiente. Lo que cambia es la forma en que el modelo describe la probabilidad de $y$.

Seguimos la notación de regresión lineal: $m$ ejemplos de entrenamiento $(fvec(x)^(\(i\)), y^(\(i\)))$, parámetros $theta_0, ..., theta_(n-1)$, y la convención $x_0 = 1$, de forma que $theta^T fvec(x) = theta_0 + theta_1 x_1 + dots + theta_(n-1) x_(n-1)$.

Estas notas siguen los capítulos 2 y 5 de las #link("https://cs229.stanford.edu/main_notes.pdf")[notas de Stanford CS229] y la sección 4.3 de #link("https://www.statlearning.com")[ISLP].

= Parte 1: Clasificación lineal

== ¿Por qué no usar regresión lineal?

La primera idea sería ignorar que $y$ solo toma los valores 0 y 1, ajustar una regresión lineal como siempre, y predecir $y = 1$ cuando $h_theta (fvec(x)) >= 0.5$. En algunos casos esto funciona, pero es fácil construir ejemplos donde falla.

En la @fig-regresion-falla, $x$ es el tamaño de un tumor y $y = 1$ indica que es maligno. A la izquierda, la recta de mínimos cuadrados cruza 0.5 en $x approx #num(d-reg.a.umbral)$, que separa bien las dos clases. A la derecha agregamos tres tumores muy grandes, todos malignos. Estos ejemplos no deberían cambiar nada: son casos fáciles. Pero la recta se inclina para acercarse a ellos, el umbral se mueve a $x approx #num(d-reg.b.umbral)$, y ahora #d-reg.b.errores tumores malignos quedan mal clasificados.

#let panel-regresion(d, x-max) = custom-plot(
  x-min: 0,
  x-max: x-max,
  y-min: -0.3,
  y-max: 1.5,
  x-label: [Tamaño del tumor],
  y-label: $y$,
  x-tick-step: if x-max > 15 { 5 } else { 2 },
  y-tick-step: 0.5,
  size: (6, 4.5),
  {
    let (t0, t1) = d.theta
    plot.add-hline(0.5, style: (stroke: (paint: gray, thickness: 0.6pt, dash: "dotted")))
    plot.add-vline(d.umbral, style: (stroke: (paint: black, thickness: 0.8pt, dash: "dashed")))
    plot.add(
      domain: (0, x-max),
      x => t0 + t1 * x,
      style: (stroke: (paint: green.darken(20%), thickness: 1.6pt)),
    )
    puntos-clase(d.puntos.filter(p => p.at(1) == 0), 0)
    puntos-clase(d.puntos.filter(p => p.at(1) == 1), 1)
  },
)

#figure(
  caption: [Regresión lineal aplicada a etiquetas 0/1. La recta verde es el ajuste de mínimos cuadrados y la línea punteada vertical es el punto donde la recta vale 0.5. Al agregar tres ejemplos fáciles a la derecha, la recta cambia de pendiente y el umbral se desplaza, clasificando mal ejemplos que antes estaban bien.],
  grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    panel-regresion(d-reg.a, 10), panel-regresion(d-reg.b, 30),
  ),
) <fig-regresion-falla>

Hay dos problemas de fondo. El primero es que el error cuadrático castiga a la recta por predecir valores lejanos a 1 para los tumores grandes, aunque esas predicciones estén del lado correcto. El segundo es que $h_theta (fvec(x))$ puede tomar cualquier valor real, mientras que sabemos que $y in {0, 1}$. Una predicción de 3.7 o de $-2$ no tiene una interpretación clara.

Vamos a resolver ambos problemas, pero primero conviene entender qué significa clasificar con un modelo lineal.

== Score, regla de decisión y frontera

Llamamos score al valor $z = theta^T fvec(x)$. Un clasificador lineal predice usando el signo del score:

$
  hat(y) = cases(
    1 quad & "si" theta^T fvec(x) >= 0,
    0 quad & "si" theta^T fvec(x) < 0.
  )
$

Los puntos donde el score vale exactamente cero forman la frontera de decisión:

$
  {fvec(x) : theta^T fvec(x) = 0}.
$

Con dos variables de entrada, la frontera es $theta_0 + theta_1 x_1 + theta_2 x_2 = 0$, que es una recta. Con tres variables es un plano, y en general es un hiperplano. La frontera divide el espacio en dos mitades: de un lado el modelo predice la clase 1 y del otro la clase 0. Por eso se dice que el clasificador es lineal: no porque el modelo sea una recta, sino porque la frontera lo es.

Si existe algún $theta$ cuya frontera deja todos los ejemplos de entrenamiento del lado correcto, decimos que los datos son linealmente separables.

== Geometría de la frontera

Tomemos el caso de dos variables. El vector $(theta_1, theta_2)$ es perpendicular a la frontera. Para verlo, tomemos dos puntos $fvec(a)$ y $fvec(b)$ sobre la frontera. Restando sus ecuaciones:

$
  (theta_0 + theta_1 a_1 + theta_2 a_2) - (theta_0 + theta_1 b_1 + theta_2 b_2) = theta_1 (a_1 - b_1) + theta_2 (a_2 - b_2) = 0.
$

Es decir, $(theta_1, theta_2)$ es perpendicular a $fvec(a) - fvec(b)$, que es un vector en la dirección de la frontera. A $(theta_1, theta_2)$ se le llama vector normal. Además, apunta hacia el lado donde el score es positivo, es decir, hacia la región que el modelo clasifica como 1. El parámetro $theta_0$ no cambia la orientación de la frontera: solo la desplaza.

El score también dice qué tan lejos está un punto de la frontera. Se puede demostrar que la distancia (con signo) de un punto $fvec(x)$ a la frontera es

$
  (theta_0 + theta_1 x_1 + theta_2 x_2) / sqrt(theta_1^2 + theta_2^2).
$

Así, un score grande y positivo significa que el punto está lejos de la frontera, del lado de la clase 1. Esto lo vamos a interpretar como que el modelo está muy seguro de su predicción.

Un detalle importante: si multiplicamos todos los parámetros por una constante $c > 0$, la frontera no cambia, porque $c theta^T fvec(x) = 0$ si y solo si $theta^T fvec(x) = 0$. Pero los scores sí cambian: se multiplican por $c$. Volveremos a esto al entrenar.

#let panel-frontera(otras: false) = custom-plot(
  x-min: 0,
  x-max: 5,
  y-min: 0,
  y-max: 5,
  x-label: $x_1$,
  y-label: $x_2$,
  x-tick-step: 1,
  y-tick-step: 1,
  size: (5.5, 5.5),
  {
    import cetz.draw: *
    let t = d-front.theta
    if otras {
      let colores = (green.darken(20%), orange.darken(10%), purple)
      for (k, o) in d-front.otras.enumerate() {
        recta(o, estilo: (paint: colores.at(k), thickness: 1.3pt, dash: "dashed"))
      }
      recta(t)
    } else {
      recta(t)
      // Vector normal desde un punto de la frontera
      let p = (3.3, 1.3)
      vector(p, (p.at(0) + t.at(1) * 0.8, p.at(1) + t.at(2) * 0.8), label: $(theta_1, theta_2)$, label-position: "head")
      plot.annotate({
        content((4.1, 4.6), text(size: 9pt)[$theta^T fvec(x) > 0$])
        content((0.9, 0.3), text(size: 9pt)[$theta^T fvec(x) < 0$])
      })
    }
    puntos-clase(d-front.clase0, 0)
    puntos-clase(d-front.clase1, 1)
  },
)

#figure(
  caption: [A la izquierda, una frontera lineal $theta_0 + theta_1 x_1 + theta_2 x_2 = 0$ con su vector normal $(theta_1, theta_2)$, que apunta hacia la región donde el modelo predice $y = 1$. A la derecha, los mismos datos con otras tres fronteras que también separan perfectamente las dos clases.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      panel-frontera(), panel-frontera(otras: true),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-frontera>

La @fig-frontera deja abierta una pregunta: cuando los datos son separables, hay infinitas fronteras que clasifican perfectamente el conjunto de entrenamiento. ¿Cuál deberíamos escoger? Intuitivamente, una frontera que pasa muy cerca de algunos puntos parece más frágil que una que deja espacio a ambos lados. Esta idea dará origen a las máquinas de soporte vectorial (SVM). Por ahora, vamos a escoger $theta$ con un criterio probabilístico.

= Parte 2: Regresión logística

== La función sigmoide

En lugar de predecir directamente 0 o 1, nos gustaría que el modelo diera una probabilidad: un número entre 0 y 1 que indique qué tan probable es que $y = 1$. Para eso tomamos el score $theta^T fvec(x)$, que puede ser cualquier número real, y lo pasamos por una función que lo comprima al intervalo $(0, 1)$:

$
  h_theta (fvec(x)) = g(theta^T fvec(x)) = 1 / (1 + e^(-theta^T fvec(x))),
$

donde

$
  g(z) = 1 / (1 + e^(-z))
$

se llama función logística o función sigmoide. A este modelo se le llama regresión logística. A pesar del nombre, es un método de clasificación: la palabra "regresión" viene de que el modelo predice un número (la probabilidad).

#figure(
  caption: [La función sigmoide $g(z) = 1 slash (1 + e^(-z))$. Cuando el score $z$ es muy positivo, $g(z)$ se acerca a 1; cuando es muy negativo, se acerca a 0. Cuando $z = 0$, $g(z) = 0.5$.],
  custom-plot(
    x-min: -6,
    x-max: 6,
    y-min: 0,
    y-max: 1,
    x-label: $z$,
    y-label: $g(z)$,
    x-tick-step: 2,
    y-tick-step: 0.25,
    size: (8, 4.5),
    {
      plot.add-hline(0.5, style: (stroke: (paint: gray, thickness: 0.6pt, dash: "dotted")))
      plot.add-vline(0, style: (stroke: (paint: gray, thickness: 0.6pt, dash: "dotted")))
      plot.add(
        domain: (-6, 6),
        samples: 100,
        sigmoide,
        style: (stroke: (paint: c0, thickness: 1.8pt)),
      )
    },
  ),
) <fig-sigmoide>

Algunas propiedades de la sigmoide que usaremos:

- $g(z) -> 1$ cuando $z -> infinity$, y $g(z) -> 0$ cuando $z -> -infinity$.
- $g(0) = 0.5$.
- $g(-z) = 1 - g(z)$. Es decir, la curva es simétrica alrededor del punto $(0, 0.5)$.
- Su derivada se puede escribir en términos de la misma función:

$
  g'(z) = d/(d z) 1 / (1 + e^(-z))
  = e^(-z) / (1 + e^(-z))^2
  = 1 / (1 + e^(-z)) dot (1 - 1 / (1 + e^(-z)))
  = g(z) (1 - g(z)).
$

En el tercer paso usamos que $e^(-z) slash (1 + e^(-z)) = 1 - 1 slash (1 + e^(-z))$.

La sigmoide no es la única función que va suavemente de 0 a 1, pero es la más usada, y tiene propiedades que simplifican mucho las cuentas.

== Un modelo de probabilidad para $y$

Interpretamos la salida del modelo como la probabilidad de que $y = 1$ dado $fvec(x)$:

$
  P(y = 1 | fvec(x); theta) = h_theta (fvec(x)), quad quad P(y = 0 | fvec(x); theta) = 1 - h_theta (fvec(x)).
$

Las dos ecuaciones se pueden escribir en una sola:

$
  p(y | fvec(x); theta) = (h_theta (fvec(x)))^y (1 - h_theta (fvec(x)))^(1 - y).
$

Basta sustituir $y = 1$ y $y = 0$ para comprobarlo: en cada caso, uno de los dos factores se vuelve 1.

Para clasificar, predecimos $y = 1$ cuando $h_theta (fvec(x)) >= 0.5$. Como $g(z) >= 0.5$ exactamente cuando $z >= 0$, esto es lo mismo que predecir $y = 1$ cuando $theta^T fvec(x) >= 0$. Es decir, la regresión logística es un clasificador lineal: su frontera de decisión es el mismo hiperplano $theta^T fvec(x) = 0$ de la Parte 1. Lo que agrega es una probabilidad para cada punto, que depende de qué tan lejos está de la frontera.

Más en general, los puntos donde el modelo asigna una probabilidad fija $p$ cumplen $g(theta^T fvec(x)) = p$, es decir, $theta^T fvec(x) = log(p slash (1 - p))$. Son rectas paralelas a la frontera, como se ve en la @fig-logistica.

#figure(
  caption: [Regresión logística ajustada a #d-log.m ejemplos cuyas clases se solapan. La recta continua es la frontera de decisión ($h_theta (fvec(x)) = 0.5$), y las rectas punteadas son los puntos donde el modelo asigna probabilidad 0.1 y 0.9 a la clase 1. Todas son paralelas. El modelo clasifica bien el #calc.round(d-log.exactitud * 100)% de los ejemplos de entrenamiento.],
  [
    #custom-plot(
      x-min: 0,
      x-max: 5,
      y-min: 0,
      y-max: 5,
      x-label: $x_1$,
      y-label: $x_2$,
      x-tick-step: 1,
      y-tick-step: 1,
      size: (6.5, 6),
      {
        import cetz.draw: *
        let t = d-log.theta
        for p in (0.1, 0.9) {
          recta(t, c: calc.ln(p / (1 - p)), estilo: (paint: gray.darken(30%), thickness: 1pt, dash: "dashed"))
        }
        recta(t)
        puntos-clase(d-log.clase0, 0)
        puntos-clase(d-log.clase1, 1)
        plot.annotate({
          content((0.2, 0.2), anchor: "south-west", text(size: 9pt)[$h approx 0.1$])
          content((4.8, 4.8), anchor: "north-east", text(size: 9pt)[$h approx 0.9$])
        })
      },
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-logistica>

Hay otra forma útil de leer el modelo. Si despejamos el score:

$
  (h_theta (fvec(x))) / (1 - h_theta (fvec(x))) = e^(theta^T fvec(x)) quad "y por lo tanto" quad log (h_theta (fvec(x))) / (1 - h_theta (fvec(x))) = theta^T fvec(x).
$

Al cociente $p slash (1 - p)$ se le llama odds (por ejemplo, $p = 0.75$ corresponde a odds de 3 a 1), y a su logaritmo se le llama log-odds o logit. Entonces la regresión logística supone que el log-odds de la clase 1 es una función lineal de $fvec(x)$. Esto da una interpretación de los coeficientes: si $x_j$ aumenta en una unidad y lo demás queda igual, el log-odds aumenta en $theta_j$, o sea, el odds se multiplica por $e^(theta_j)$.

= Parte 3: Máxima verosimilitud

Ya tenemos el modelo. Falta escoger $theta$. Para la regresión lineal vimos que, si suponemos $y = theta^T fvec(x) + epsilon$ con $epsilon tilde cal(N)(0, sigma^2)$, maximizar la verosimilitud es lo mismo que minimizar el error cuadrático. Vamos a seguir la misma receta: escribir la probabilidad que el modelo le asigna a los datos y escoger el $theta$ que la hace más grande.

== Verosimilitud y log-verosimilitud

Suponemos que los $m$ ejemplos de entrenamiento se generaron de forma independiente. Entonces la verosimilitud de los parámetros es el producto de las probabilidades de cada etiqueta:

$
  L(theta) = p(fvec(y) | fvec(X); theta)
  = product_(i=1)^m p(y^(\(i\)) | fvec(x)^(\(i\)); theta)
  = product_(i=1)^m (h_theta (fvec(x)^(\(i\))))^(y^(\(i\))) (1 - h_theta (fvec(x)^(\(i\))))^(1 - y^(\(i\))).
$

$L(theta)$ es la probabilidad que el modelo le asigna a las etiquetas que efectivamente observamos. El principio de máxima verosimilitud dice que debemos escoger el $theta$ que la hace más grande.

Igual que en regresión lineal, es más cómodo maximizar el logaritmo, que convierte el producto en una suma:

$
  ell(theta) = log L(theta) = sum_(i=1)^m y^(\(i\)) log h_theta (fvec(x)^(\(i\))) + (1 - y^(\(i\))) log (1 - h_theta (fvec(x)^(\(i\)))).
$

Como el logaritmo es una función creciente, el $theta$ que maximiza $ell(theta)$ es el mismo que maximiza $L(theta)$. Además del motivo algebraico, hay un motivo numérico: un producto de muchas probabilidades menores que 1 se vuelve rápidamente un número tan pequeño que la computadora lo redondea a 0.

== Lo que aporta cada ejemplo

Como $y$ vale 0 o 1, en cada término de $ell(theta)$ solo uno de los dos logaritmos está activo:

$
  "aporte del ejemplo" = cases(
    log h_theta (fvec(x)) quad & "si" y = 1,
    log (1 - h_theta (fvec(x))) quad & "si" y = 0.
  )
$

Estos valores siempre son menores o iguales a cero, porque son logaritmos de probabilidades. Así que $ell(theta) <= 0$, y el mejor valor posible sería $ell(theta) = 0$, que corresponde a asignarle probabilidad 1 a todas las etiquetas observadas.

#figure(
  caption: [Aporte de un ejemplo a la log-verosimilitud en función de la probabilidad $h$ que el modelo le asigna a la clase 1. Si $y = 1$ (rojo), el aporte es $log h$: vale cero cuando $h = 1$ y se va a $-infinity$ cuando $h -> 0$. Si $y = 0$ (azul), el aporte es $log(1 - h)$, con el comportamiento opuesto.],
  [
    #custom-plot(
      x-min: 0,
      x-max: 1,
      y-min: -5,
      y-max: 0,
      x-label: $h_theta (fvec(x))$,
      y-label: [Aporte a $ell(theta)$],
      x-tick-step: 0.25,
      y-tick-step: 1,
      size: (7, 4.5),
      {
        plot.add(
          domain: (0.006, 1),
          samples: 150,
          h => calc.ln(h),
          style: (stroke: (paint: c1, thickness: 1.8pt)),
        )
        plot.add(
          domain: (0, 0.994),
          samples: 150,
          h => calc.ln(1 - h),
          style: (stroke: (paint: c0, thickness: 1.8pt, dash: "dashed")),
        )
      },
    )
    #v(0.2cm)
    #set text(size: 9pt)
    #grid(
      columns: 2,
      column-gutter: 16pt,
      box[#box(width: 0.6cm, height: 0pt, baseline: -0.3em, stroke: (top: (paint: c1, thickness: 1.6pt))) $y = 1$: $log h$],
      box[#box(width: 0.6cm, height: 0pt, baseline: -0.3em, stroke: (top: (paint: c0, thickness: 1.6pt, dash: "dashed"))) $y = 0$: $log(1 - h)$],
    )
  ],
) <fig-aporte>

La @fig-aporte muestra el comportamiento. Si el ejemplo es de la clase 1 y el modelo le asigna probabilidad 0.9, su aporte es $log 0.9 approx -0.11$. Si le asigna 0.1, su aporte es $log 0.1 approx -2.3$. Y si le asigna una probabilidad cercana a cero, el aporte es enorme y negativo. Para maximizar $ell(theta)$, el modelo tiene que evitar sobre todo los errores hechos con mucha seguridad.

Esto también resuelve el problema de la @fig-regresion-falla. Un tumor muy grande y maligno tiene un score muy positivo, $h_theta approx 1$, y su aporte es prácticamente cero, que es lo máximo posible. No tira de la frontera como lo hacía el error cuadrático.

== Comparación con regresión lineal

Ambos modelos salen de la misma receta: suponer cómo se distribuye $y$ dado $fvec(x)$, escribir la log-verosimilitud y maximizarla.

#align(center, block(breakable: false, table(
    columns: (2.8cm, 4.6cm, 6.6cm),
    align: (left, center, center),
    [], [Regresión lineal], [Regresión logística],
    [Suposición], [$y tilde cal(N)(theta^T fvec(x), sigma^2)$], [$P(y = 1 | fvec(x); theta) = h_theta (fvec(x))$],
    [$h_theta (fvec(x))$], [$theta^T fvec(x)$], [$g(theta^T fvec(x))$],
    [Maximizar $ell$], [$min 1/2 sum (h_theta - y)^2$], [$max sum y log h_theta + (1 - y) log(1 - h_theta)$],
    [Solución cerrada], [Sí], [No],
  )))

¿Por qué no simplemente minimizar el error cuadrático con la sigmoide, $1/2 sum (g(theta^T fvec(x)^(\(i\))) - y^(\(i\)))^2$? Se puede, pero tiene dos problemas. El primero es que esa función puede tener mínimos locales. El segundo es que su gradiente incluye el factor $g'(z) = g(z)(1 - g(z))$, que es casi cero cuando el modelo está muy seguro. Si el modelo está muy seguro y equivocado, el gradiente es casi cero y el modelo aprende muy lento justo cuando más lo necesita. Como veremos ahora, en el gradiente de $ell(theta)$ ese factor se cancela.

= Parte 4: Entrenamiento

== El gradiente

Como queremos maximizar $ell(theta)$, vamos a usar ascenso por gradiente: en lugar de restar el gradiente, lo sumamos, porque el gradiente apunta en la dirección en la que la función crece más rápido:

$
  theta := theta + alpha gradient_theta ell(theta).
$

Calculemos la derivada para un solo ejemplo $(fvec(x), y)$. Escribimos $h = g(theta^T fvec(x))$ y usamos la regla de la cadena junto con $g'(z) = g(z)(1 - g(z))$:

$
  partial / (partial theta_j) [y log h + (1 - y) log(1 - h)]
  &= (y / h - (1 - y) / (1 - h)) partial / (partial theta_j) g(theta^T fvec(x)) \
  &= (y / h - (1 - y) / (1 - h)) h (1 - h) x_j \
  &= (y (1 - h) - (1 - y) h) x_j \
  &= (y - h) x_j.
$

En el segundo paso usamos que $partial (theta^T fvec(x)) slash partial theta_j = x_j$. Sumando sobre todos los ejemplos:

$
  (partial ell(theta)) / (partial theta_j) = sum_(i=1)^m (y^(\(i\)) - h_theta (fvec(x)^(\(i\)))) x_j^(\(i\)).
$

En forma matricial, si $fvec(h)$ es el vector con las predicciones $h_theta (fvec(x)^(\(i\)))$ de todos los ejemplos,

$
  gradient_theta ell(theta) = fvec(X)^T (fvec(y) - fvec(h)).
$

Cada ejemplo empuja a $theta$ en la dirección de $fvec(x)^(\(i\))$, con una fuerza proporcional a su error $y^(\(i\)) - h_theta (fvec(x)^(\(i\)))$. Un ejemplo bien clasificado con mucha seguridad tiene un error cercano a cero y casi no influye.

== Ascenso por gradiente

Sustituyendo en la regla de actualización:

$
  theta_j := theta_j + alpha sum_(i=1)^m (y^(\(i\)) - h_theta (fvec(x)^(\(i\)))) x_j^(\(i\)), quad j = 0, ..., n-1.
$

Y la versión estocástica, que usa un solo ejemplo a la vez:

$
  theta_j := theta_j + alpha (y^(\(i\)) - h_theta (fvec(x)^(\(i\)))) x_j^(\(i\)).
$

Si comparamos con la regla LMS de regresión lineal, $theta_j := theta_j + alpha (y^(\(i\)) - h_theta (fvec(x)^(\(i\)))) x_j^(\(i\))$, vemos que es idéntica. Pero no es el mismo algoritmo: en regresión lineal $h_theta (fvec(x)) = theta^T fvec(x)$, y aquí $h_theta (fvec(x)) = g(theta^T fvec(x))$. Es un resultado algo sorprendente, ya que llegamos a la misma regla partiendo de suposiciones muy distintas.

Todo lo que vimos sobre descenso por gradiente se aplica igual al ascenso: la versión batch usa todos los ejemplos en cada paso, SGD usa uno a la vez, y mini-batch usa pequeños grupos. También ayuda estandarizar las features.

A diferencia de la regresión lineal, ahora no hay solución cerrada. Al igualar el gradiente a cero, $theta$ queda adentro de la sigmoide y no se puede despejar. Por eso siempre se usa un método iterativo.

La buena noticia es que $ell(theta)$ es cóncava (la versión "al revés" de convexa), así que no tiene máximos locales: si el ascenso por gradiente converge, llega al máximo global. Para verlo, se puede calcular la matriz de segundas derivadas:

$
  (partial^2 ell(theta)) / (partial theta_j partial theta_k) = -sum_(i=1)^m h_theta (fvec(x)^(\(i\))) (1 - h_theta (fvec(x)^(\(i\)))) x_j^(\(i\)) x_k^(\(i\)).
$

Para cualquier vector $fvec(v)$, $sum_(j,k) v_j v_k (partial^2 ell) slash (partial theta_j partial theta_k) = -sum_i h_theta (fvec(x)^(\(i\))) (1 - h_theta (fvec(x)^(\(i\)))) (fvec(v)^T fvec(x)^(\(i\)))^2 <= 0$, porque cada término de la suma es un producto de números no negativos. Una función cuya matriz de segundas derivadas cumple esto es cóncava.

== Datos separables y regularización

Hay un caso donde el ascenso por gradiente no converge. Supongamos que los datos son linealmente separables y que $theta$ los separa. Como vimos en la Parte 1, si multiplicamos $theta$ por $c > 1$ la frontera no cambia, pero todos los scores se multiplican por $c$. Los ejemplos de la clase 1 tienen scores aún más positivos y los de la clase 0 aún más negativos, así que todas las probabilidades se acercan más a la etiqueta correcta y $ell$ aumenta. Esto pasa para cualquier $c$, así que el máximo no existe: $ell$ se acerca a 0 a medida que $norm(theta)$ crece sin límite, pero nunca llega.

En la práctica esto significa que, con datos separables, el ascenso por gradiente hace que los parámetros crezcan indefinidamente, y el modelo termina asignando probabilidades de casi exactamente 0 o 1. Eso es una forma de overfitting: el modelo está demasiado seguro.

La solución es la misma que en regresión lineal: penalizar el tamaño de los parámetros. Con regularización $L_2$ maximizamos

$
  ell(theta) - lambda / 2 sum_(j=1)^(n-1) theta_j^2,
$

y la regla de actualización gana el término $-alpha lambda theta_j$ para $j >= 1$. Ahora el crecimiento de $theta$ tiene un costo, y el máximo existe siempre. También se puede usar regularización $L_1$ para obtener modelos sparse. Igual que antes, se escoge $lambda$ con validación y se estandarizan las features.

#let colores-desc = (c0, c1, green.darken(20%))
#let estilos-desc = ((none), "dashed", "dotted")

#figure(
  caption: [Norma de los parámetros $sqrt(theta_1^2 + theta_2^2)$ durante el ascenso por gradiente, en escala logarítmica para las iteraciones. Con datos que se solapan (azul), la norma se estabiliza. Con datos separables (rojo), sigue creciendo sin detenerse. Con los mismos datos separables y regularización $L_2$ con $lambda = 1$ (verde), vuelve a estabilizarse.],
  [
    #custom-plot(
      x-min: 1,
      x-max: d-desc.pasos,
      y-min: 0,
      y-max: 9,
      x-label: [Iteración],
      y-label: $norm(theta)$,
      x-tick-step: 1,
      y-tick-step: 2,
      x-mode: "log",
      x-format: v => {
        let k = int(calc.round(calc.log(v)))
        $10^#k$
      },
      size: (8, 4.5),
      {
        let series = (d-desc.norma_solapan, d-desc.norma_separa, d-desc.norma_separa_reg)
        for (k, s) in series.enumerate() {
          plot.add(
            s,
            style: (stroke: (paint: colores-desc.at(k), thickness: 1.8pt, dash: estilos-desc.at(k))),
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
        #box(width: 0.6cm, height: 0pt, baseline: -0.3em, stroke: (top: (paint: colores-desc.at(k), thickness: 1.6pt, dash: estilos-desc.at(k))))
        #("Se solapan", "Separables", "Separables, " + sym.lambda + " = 1").at(k)
      ])
    )
  ],
) <fig-ascenso>

Una advertencia para el laboratorio: `LogisticRegression` de scikit-learn regulariza por defecto, y no usa $lambda$ sino un parámetro `C` que es su inverso: con la convención de estas notas, `C` $= 1 slash lambda$. Un `C` pequeño significa mucha regularización. Además, scikit-learn, como la mayoría de las librerías, plantea el problema como una minimización de $-ell(theta)$; maximizar $ell$ y minimizar $-ell$ es lo mismo.

= Parte 5: Transformaciones de features

== Fronteras no lineales con un modelo lineal

Hasta ahora, la frontera de decisión siempre es un hiperplano. Pero hay problemas donde ninguna recta sirve. En la @fig-parabola, los ejemplos de la clase 1 están por encima de una parábola y los de la clase 0 por debajo, a ambos lados. Si entrenamos una regresión logística con $x_1$ y $x_2$, la mejor recta acierta en el #calc.round(d-par.exactitud_lin * 100)% de los ejemplos: no puede dejar de un mismo lado los dos grupos de la clase 0.

La solución es la misma que usamos para la regresión polinomial: construir nuevas features a partir de las originales. Definimos una transformación, o feature map,

$
  phi : RR^d -> RR^p,
$

que toma las $d$ variables originales y produce $p$ features nuevas. El modelo es ahora

$
  h_theta (fvec(x)) = g(theta^T phi(fvec(x))),
$

con $theta in RR^p$. Para el ejemplo de la figura usamos

$
  phi(fvec(x)) = mat(1; x_1^2; x_2).
$

La frontera de decisión $theta_0 + theta_1 x_1^2 + theta_2 x_2 = 0$ es una recta en el espacio de las features $(x_1^2, x_2)$, pero en el espacio original es la parábola $x_2 = -(theta_0 + theta_1 x_1^2) slash theta_2$. Con esta transformación el modelo acierta en el #calc.round(d-par.exactitud_cuad * 100)% de los ejemplos.

#figure(
  caption: [A la izquierda, datos en el espacio original. La recta punteada es la frontera de la regresión logística con $(x_1, x_2)$; la curva continua es la frontera del modelo entrenado con $phi(fvec(x)) = (1, x_1^2, x_2)$. A la derecha, los mismos datos en el espacio transformado $(x_1^2, x_2)$, donde esa misma frontera es una recta.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      custom-plot(
        x-min: -2,
        x-max: 2,
        y-min: -1,
        y-max: 3,
        x-label: $x_1$,
        y-label: $x_2$,
        x-tick-step: 1,
        y-tick-step: 1,
        size: (5.5, 5.5),
        {
          recta(d-par.theta_lin, dominio: (-2, 2), estilo: (paint: gray.darken(30%), thickness: 1.2pt, dash: "dashed"))
          plot.add(d-par.parabola.filter(p => p.at(1) <= 3), style: (stroke: (paint: black, thickness: 1.5pt)))
          puntos-clase(d-par.clase0, 0, tam: 0.1)
          puntos-clase(d-par.clase1, 1, tam: 0.1)
        },
      ),
      custom-plot(
        x-min: 0,
        x-max: 4,
        y-min: -1,
        y-max: 3,
        x-label: $x_1^2$,
        y-label: $x_2$,
        x-tick-step: 1,
        y-tick-step: 1,
        size: (5.5, 5.5),
        {
          recta(d-par.theta_cuad, dominio: (0, 4))
          puntos-clase(d-par.phi0, 0, tam: 0.1)
          puntos-clase(d-par.phi1, 1, tam: 0.1)
        },
      ),
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-parabola>

Nada del entrenamiento cambia: basta reemplazar $fvec(x)^(\(i\))$ por $phi(fvec(x)^(\(i\)))$ en todas las fórmulas. Por ejemplo, la regla de ascenso por gradiente queda

$
  theta := theta + alpha sum_(i=1)^m (y^(\(i\)) - g(theta^T phi(fvec(x)^(\(i\))))) phi(fvec(x)^(\(i\))).
$

El modelo sigue siendo lineal en $theta$, y por eso sigue siendo fácil de entrenar. Lo no lineal está en $phi$, que escogemos nosotros.

Las mismas advertencias de la regresión polinomial aplican aquí. Con un $phi$ muy rico (por ejemplo, todos los monomios hasta grado 10), la frontera puede volverse tan flexible que rodee cada ejemplo de entrenamiento: overfitting. La complejidad se controla igual que antes, escogiendo $phi$ con validación y usando regularización.

== El costo de muchas features

En el ejemplo anterior sabíamos qué transformación usar. Muchas veces no lo sabemos, y la opción natural es incluir todos los términos cuadráticos. Si $fvec(x)$ tiene $d$ variables:

$
  phi(fvec(x)) = (1, x_1, ..., x_d, x_1 x_1, x_1 x_2, ..., x_1 x_d, x_2 x_1, ..., x_d x_d).
$

Si contamos cada producto en sus dos órdenes (por ejemplo, $x_1 x_2$ y $x_2 x_1$ por separado), hay $1 + d + d^2$ features. Con $d = 10 000$, eso es alrededor de $10^8$ features, y cada paso del ascenso por gradiente requiere calcular y guardar vectores de ese tamaño. Con términos de grado 3 el número crece como $d^3$. Parece que transformar las features nos obliga a pagar un costo enorme.

== Hacia SVM y kernels

Hay una observación que cambia el panorama. Supongamos que solo necesitamos productos internos entre features, $phi(fvec(x))^T phi(fvec(z))$. Para el $phi$ cuadrático:

$
  phi(fvec(x))^T phi(fvec(z))
  = 1 + sum_(i=1)^d x_i z_i + sum_(i=1)^d sum_(j=1)^d x_i x_j z_i z_j.
$

El último término se puede reorganizar:

$
  sum_(i=1)^d sum_(j=1)^d x_i x_j z_i z_j = (sum_(i=1)^d x_i z_i) (sum_(j=1)^d x_j z_j) = (fvec(x)^T fvec(z))^2.
$

Entonces

$
  phi(fvec(x))^T phi(fvec(z)) = 1 + fvec(x)^T fvec(z) + (fvec(x)^T fvec(z))^2.
$

El lado izquierdo es un producto interno entre vectores de $10^8$ componentes. El lado derecho solo requiere calcular $fvec(x)^T fvec(z)$, que tiene $d = 10 000$ términos, y luego unas pocas operaciones. A la función $K(fvec(x), fvec(z)) = phi(fvec(x))^T phi(fvec(z))$ se le llama kernel.

¿Y cuándo necesitamos solo productos internos? Miremos la regla de ascenso por gradiente con features. Si empezamos con $theta = 0$, cada paso le suma a $theta$ una combinación de los vectores $phi(fvec(x)^(\(i\)))$. Por lo tanto, en cualquier momento $theta = sum_i beta_i phi(fvec(x)^(\(i\)))$ para ciertos números $beta_i$, y el score de un punto nuevo es

$
  theta^T phi(fvec(x)) = sum_(i=1)^m beta_i phi(fvec(x)^(\(i\)))^T phi(fvec(x)) = sum_(i=1)^m beta_i K(fvec(x)^(\(i\)), fvec(x)).
$

Es decir, podríamos entrenar y predecir sin construir nunca $phi$: basta con guardar los $m$ números $beta_i$ y saber calcular $K$. Esta idea se conoce como el kernel trick, y la desarrollaremos en detalle más adelante.

Para cerrar, dos preguntas que quedan abiertas y que responden las máquinas de soporte vectorial:

+ Cuando los datos son separables, ¿cuál de todas las fronteras posibles es la mejor? La respuesta será la frontera con mayor margen, es decir, la que queda lo más lejos posible de los puntos más cercanos de cada clase.
+ ¿Se puede escribir ese problema de forma que solo dependa de productos internos, para usar kernels y obtener fronteras no lineales sin calcular $phi$?

En ese contexto es costumbre separar el intercepto del resto de los parámetros. Si llamamos $b = theta_0$ y $fvec(w) = (theta_1, ..., theta_(n-1))$, el score se escribe

$
  theta^T fvec(x) = fvec(w)^T fvec(x) + b,
$

donde ahora $fvec(x)$ no incluye el 1 inicial. El vector $fvec(w)$ es el vector normal a la frontera, y la distancia de un punto a la frontera es $(fvec(w)^T fvec(x) + b) slash norm(fvec(w))$. Esta es la notación que usaremos con SVM.

= Resumen

- Un clasificador lineal predice con el signo del score $theta^T fvec(x)$. Su frontera de decisión es el hiperplano $theta^T fvec(x) = 0$, con vector normal $(theta_1, ..., theta_(n-1))$.
- La regresión logística pasa el score por la sigmoide, $h_theta (fvec(x)) = g(theta^T fvec(x))$, y lo interpreta como la probabilidad de que $y = 1$.
- Los parámetros se escogen por máxima verosimilitud, maximizando $ell(theta) = sum y log h_theta + (1 - y) log(1 - h_theta)$.
- El gradiente es $sum (y^(\(i\)) - h_theta (fvec(x)^(\(i\)))) fvec(x)^(\(i\))$, y la regla de ascenso por gradiente tiene la misma forma que la regla LMS. No hay solución cerrada, pero $ell$ es cóncava y el ascenso por gradiente encuentra el máximo global.
- Con datos separables, los parámetros crecen sin límite; la regularización lo evita.
- Con un feature map $phi$ obtenemos fronteras no lineales sin dejar de tener un modelo lineal en $theta$. Cuando $phi$ tiene muchas componentes, los kernels permiten calcular productos internos $phi(fvec(x))^T phi(fvec(z))$ sin construir $phi$.
