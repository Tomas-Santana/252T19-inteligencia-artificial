// OUTLINE (3 horas, en tres bloques con un descanso)
// 1. Una neurona
// 1.1. De la regresión logística a una neurona
// 1.2. Funciones de activación
// 1.3. El límite de una neurona: XOR
// 2. Redes neuronales
// 2.1. Apilar neuronas (Ejercicios 1 y 2, en papel)
// 2.2. La capa oculta transforma los datos
// 2.3. Redes totalmente conectadas
// 2.4. Notación vectorial
// 2.5. Redes con más capas
// 2.6. ¿Por qué no usar la identidad como activación?
// 2.7. La red aprende las features
// 2.8. Varias clases: softmax
// 2.9. (Opcional) Vectorización sobre varios ejemplos
// 3. Cómo aprende una red
// 3.1. La pérdida
// 3.2. Descenso por gradiente (Ejercicio 3, en papel)
// 3.3. Backpropagation: el error viaja hacia atrás (Ejercicio 4, en papel)
// 3.4. Entrenar una red en PyTorch
// 3.5. (Opcional) Backpropagation en general
//
// Hay dos versiones de estas notas, generadas desde este mismo archivo:
//   pública:      typst compile --root . notes/07-redes-neuronales.typ
//   del profesor: typst compile --root . --input version=profesor notes/07-redes-neuronales.typ notes/07-redes-neuronales-profesor.pdf
// La versión del profesor muestra, en rojo y negrita, las indicaciones para la clase
// y las soluciones de los ejercicios. No subir ese PDF al repositorio público.
//
// Los datos de las figuras se generan con `python notes/scripts/07-redes-neuronales-figuras.py`
// y se leen desde la carpeta figs/07.

#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *

#let fvec(x) = math.bold(math.upright(x))
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

// Versión del profesor: comentarios en rojo y negrita
#let profesor = sys.inputs.at("version", default: "publica") == "profesor"
#let prof(cuerpo) = if profesor {
  block(
    width: 100%,
    inset: (left: 8pt, y: 4pt),
    stroke: (left: 2pt + red),
    text(fill: red, weight: "bold")[#cuerpo],
  )
}

#show: minimal-note.with(
  title: if profesor [Redes neuronales (versión del profesor)] else [Redes neuronales],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

// Datos de las figuras
#let d-lunas = json("figs/07/lunas.json")
#let d-perdida = json("figs/07/perdida.json")

#let num(x, digits: 2) = str(calc.round(x, digits: digits))
#let pct(x) = str(calc.round(x * 100, digits: 1)) + "%"

// Estilo de las clases (el mismo de las notas anteriores)
#let c-neg = rgb("#1f77b4")
#let c-pos = rgb("#d62728")
#let c-3 = green.darken(25%)

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
  set text(size: 9pt)
  grid(
    columns: 2,
    column-gutter: 16pt,
    box[#text(fill: c-neg)[●] clase 0], box[#text(fill: c-pos)[▲] clase 1],
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

#let ejercicio(titulo, cuerpo) = block(
  width: 100%,
  fill: luma(237),
  inset: 12pt,
  radius: 4pt,
  breakable: false,
)[
  *#titulo*

  #cuerpo
]

// Diagrama de una red: `capas` es la lista de neuronas por capa.
// `etiquetas` asigna contenido a la neurona i de la capa l con la clave "l-i".
// `pesos` asigna una etiqueta a la conexión de (l, i) a (l+1, j) con la clave "l-i-j".
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

// Panel cuadrado para los datos de XOR
#let panel-xor(cuerpo, x-label: $x_1$, y-label: $x_2$, x-min: -0.5, x-max: 1.5, y-min: -0.5, y-max: 1.5) = custom-plot(
  x-min: x-min,
  x-max: x-max,
  y-min: y-min,
  y-max: y-max,
  x-label: x-label,
  y-label: y-label,
  x-tick-step: 0.5,
  y-tick-step: 0.5,
  size: (4.6, 4.6),
  cuerpo,
)

#let xor-neg = ((0, 0), (1, 1))
#let xor-pos = ((0, 1), (1, 0))

#if profesor [
  #text(fill: red, weight: "bold", size: 12pt)[Plan de la clase (3 horas)]

  #prof[
    - *0:00–0:55 · Bloque 1.* Parte 1 y secciones 2.1 a 2.3. Ejercicios 1 y 2 en papel, en parejas (unos 25 min en total).
    - *0:55–1:05 · Descanso.*
    - *1:05–1:55 · Bloque 2.* Secciones 2.4 a 2.8 (unos 20 min; la 2.9 solo si van bien de tiempo). Demo del Playground (unos 10 min). Lab, Parte 1 (ejercicios 1 a 3 del notebook, unos 20 min).
    - *1:55–2:50 · Bloque 3.* Parte 3. Ejercicios 3 y 4 en papel. Lab, Parte 2 (ejercicios 4 y 5 del notebook).
    - *2:50–3:00 · Cierre.* Las tres ideas del día (Resumen) y presentar la Tarea 4.
    - Si se acaba el tiempo: la sección 2.9, la 3.5 y el ejercicio 6 del notebook son lo primero que se corta. No cortar el ejercicio 4 del notebook (`loss.backward()` da −32): es el momento clave del Bloque 3.
    - Recordar: el Playground no funciona sin internet en el laboratorio. Abrirlo antes de clase desde tu máquina.
  ]
]

= Parte 1: Una neurona

#prof[Bloque 1 (0:00). Arrancar con la pregunta: "¿Qué hicimos en la clase 4 cuando la frontera no era una recta?" Que respondan ellos: inventamos features, como $x_1^2$. Esa es la motivación de toda la clase: hoy la red las inventa sola.]

== De la regresión logística a una neurona

Recordemos el modelo de regresión logística de la clase 4. Dado un ejemplo $fvec(x) in RR^d$, el modelo calcula un score lineal $theta^T fvec(x)$ y lo pasa por la función sigmoide:

$
  h_theta (fvec(x)) = g(theta^T fvec(x)), quad quad g(z) = 1 / (1 + e^(-z)).
$

En redes neuronales es costumbre escribir el intercepto por separado, en lugar de agregar una feature $x_0 = 1$. Escribimos los parámetros como un vector de pesos $fvec(w) in RR^d$ y un número $b in RR$, que llamamos el sesgo (bias). El score es entonces

$
  z = fvec(w)^T fvec(x) + b = w_1 x_1 + w_2 x_2 + dots.c + w_d x_d + b,
$

que es lo mismo que $theta^T fvec(x)$ con $theta_0 = b$ y $theta_j = w_j$ para $j = 1, ..., d$. La predicción es

$
  a = g(z).
$

A una unidad que calcula una suma ponderada de sus entradas, le suma un sesgo y aplica una función no lineal al resultado, la llamamos una *neurona*. La función $g$ se llama *función de activación*, y el número $a$ es la *activación* de la neurona. Con esta terminología, la regresión logística es una red neuronal con una sola neurona (@fig-neurona).

#figure(
  caption: [La regresión logística dibujada como una neurona. Cada entrada $x_j$ se multiplica por su peso $w_j$, se suman los resultados y el sesgo $b$, y la suma $z$ pasa por la activación $g$.],
  red-diagrama(
    (3, 1),
    etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "0-2": $x_3$, "1-0": $a$),
    pesos: ("0-0-0": $w_1$, "0-1-0": $w_2$, "0-2-0": $w_3$),
    pos-peso: 0.5,
    nombres: ([entrada], [$a = g(fvec(w)^T fvec(x) + b)$]),
    dx: 4,
  ),
) <fig-neurona>

El nombre viene de las neuronas del cerebro: una neurona biológica recibe señales de otras, y se activa si la señal total es suficientemente fuerte. La analogía es bastante libre, y no hay que tomarla muy en serio. Para nosotros, una neurona es simplemente una función de $fvec(x)$ con parámetros $fvec(w)$ y $b$.

== Funciones de activación

La sigmoide no es la única función de activación posible. Supongamos, por ejemplo, que queremos predecir el precio de una casa a partir de su tamaño. Una recta $w x + b$ puede dar precios negativos para casas muy pequeñas. Una forma de evitarlo es poner el precio en cero cuando la recta es negativa:

$
  hat(y) = max(w x + b, 0).
$

La función $max(z, 0)$ se llama *ReLU* (rectified linear unit):

$
  "ReLU"(z) = max(z, 0).
$

Si el score es negativo, la neurona da 0 (decimos que está "apagada"); si es positivo, deja pasar el valor sin cambiarlo (@fig-activaciones). Hoy, la ReLU es la activación más usada en las capas internas de una red. Parece demasiado simple para servir de algo, pero, como veremos, lo único que se necesita es que la activación no sea lineal. Además, las redes con ReLU son mucho más fáciles de entrenar que las redes con sigmoide cuando tienen muchas capas.

#figure(
  caption: [Las dos activaciones que usaremos. La sigmoide comprime cualquier número al intervalo $(0, 1)$, así que es útil en la salida cuando queremos una probabilidad. La ReLU es cero para $z < 0$ y la identidad para $z >= 0$.],
  grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [Sigmoide $g(z)$ #v(0.1cm) #custom-plot(
        x-min: -5,
        x-max: 5,
        y-min: -0.2,
        y-max: 1.2,
        x-label: $z$,
        x-tick-step: 2,
        y-tick-step: 0.5,
        size: (5, 3.4),
        plot.add(domain: (-5, 5), samples: 100, sigmoide, style: (stroke: (paint: c-neg, thickness: 1.8pt))),
      )],
    [ReLU $max(z, 0)$ #v(0.1cm) #custom-plot(
        x-min: -3,
        x-max: 3,
        y-min: -0.5,
        y-max: 3,
        x-label: $z$,
        x-tick-step: 1,
        y-tick-step: 1,
        size: (5, 3.4),
        plot.add(domain: (-3, 3), samples: 61, relu, style: (stroke: (paint: c-pos, thickness: 1.8pt))),
      )],
  ),
) <fig-activaciones>

== El límite de una neurona: XOR

Una sola neurona tiene la misma limitación que la regresión logística: su frontera de decisión, $fvec(w)^T fvec(x) + b = 0$, es una recta (o, con más features, un hiperplano). El ejemplo más simple de un problema que no se puede resolver con una recta es la función *XOR* ("o exclusivo"), en la que la etiqueta es 1 si exactamente una de las dos entradas es 1.

#grid(
  columns: (1fr, 1fr),
  align: horizon,
  gutter: 0.5cm,
  table(
    columns: 3,
    align: center,
    table.header($x_1$, $x_2$, $y$),
    $0$, $0$, $0$,
    $0$, $1$, $1$,
    $1$, $0$, $1$,
    $1$, $1$, $0$,
  ),
  figure(
    caption: [Los cuatro ejemplos de XOR. Ninguna recta deja los triángulos de un lado y los círculos del otro.],
    [
      #panel-xor({
        puntos(xor-neg, 0, tam: 0.22)
        puntos(xor-pos, 1, tam: 0.22)
      })
      #leyenda-clases
    ],
  ),
)

Ya vimos estos datos en la clase de árboles, donde un árbol los resuelve con dos preguntas. En la clase 4 resolvimos problemas parecidos con un feature map $phi(fvec(x))$ diseñado a mano: para XOR serviría, por ejemplo, agregar la feature $x_1 x_2$. Pero diseñar features a mano requiere conocer bien el problema, y no escala. Con imágenes de miles de píxeles, nadie sabe qué features escribir.

#prof[Preguntar a la clase antes de mostrar la figura: "¿Alguna recta separa estos cuatro puntos?" Dejar que lo intenten en el aire. Recordar la figura de XOR de la clase de árboles.]

= Parte 2: Redes neuronales

== Apilar neuronas

Una red neuronal más compleja se construye tomando varias neuronas y "apilándolas": la salida de unas neuronas es la entrada de otras.

Veamos cómo resolver XOR de esta forma. Supongamos por un momento que sabemos algo del problema: la etiqueta es 1 cuando al menos una de las entradas es 1, excepto cuando las dos lo son. Esto sugiere calcular primero dos cantidades intermedias: cuántas entradas están en 1, y si las dos lo están. Cada una se puede calcular con una neurona con activación ReLU:

$
  a_1 & = "ReLU"(x_1 + x_2) quad && "(cuántas entradas están en 1)", \
  a_2 & = "ReLU"(x_1 + x_2 - 1) quad && "(vale 1 si las dos están en 1, y 0 si no)".
$

A partir de $a_1$ y $a_2$, la predicción es otra neurona, esta vez sin activación:

$
  hat(y) = a_1 - 2 a_2.
$

Es decir, contamos las entradas que están en 1, y si son las dos, restamos 2. Esta es una red neuronal con tres neuronas (@fig-red-xor). A las cantidades intermedias $a_1$ y $a_2$ se les llama *unidades ocultas* o neuronas ocultas, porque no aparecen en los datos: no son entradas ni etiquetas.

#figure(
  caption: [La red que resuelve XOR. Los números sobre las flechas son los pesos y los números dentro de cada neurona son los sesgos. Las neuronas ocultas usan ReLU y la de salida no tiene activación.],
  red-diagrama(
    (2, 2, 1),
    etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "1-0": $0$, "1-1": $-1$, "2-0": $0$),
    pesos: ("0-0-0": $1$, "0-0-1": $1$, "0-1-0": $1$, "0-1-1": $1$, "1-0-0": $1$, "1-1-0": $-2$),
    nombres: ([entrada], [oculta (ReLU)], [salida]),
    dx: 4,
    pos-peso: 0.4,
  ),
) <fig-red-xor>

#ejercicio("Ejercicio 1 ")[
  Calcula la predicción de la red de la @fig-red-xor para los cuatro ejemplos de XOR y completa la tabla. ¿La red acierta en todos?

  #align(center, table(
    columns: 6,
    align: center + horizon,
    rows: (auto, 0.75cm),
    table.header($x_1$, $x_2$, $x_1 + x_2$, $a_1$, $a_2$, $hat(y) = a_1 - 2 a_2$),
    $0$, $0$, [], [], [], [],
    $0$, $1$, [], [], [], [],
    $1$, $0$, [], [], [], [],
    $1$, $1$, [], [], [], [],
  ))
]

#prof[Ejercicio 1 en parejas, unos 12 min. Solución: $x_1 + x_2 = 0, 1, 1, 2$; $a_1 = 0, 1, 1, 2$; $a_2 = 0, 0, 0, 1$; $hat(y) = 0, 1, 1, 0$. La red acierta en los cuatro. Señalar el caso (1, 1): $a_2$ se "enciende" y su peso $-2$ cancela a $a_1$.]

A calcular $hat(y)$ a partir de $fvec(x)$, neurona por neurona y de izquierda a derecha, se le llama *propagación hacia adelante* (forward pass).

La ReLU parece un detalle. Veamos qué pasa si la quitamos.

#ejercicio("Ejercicio 2 ")[
  Repite el Ejercicio 1 sin la ReLU: $a_1 = x_1 + x_2$, $a_2 = x_1 + x_2 - 1$ y $hat(y) = a_1 - 2 a_2$.
  + Calcula $hat(y)$ para los cuatro ejemplos. ¿Sigue resolviendo XOR?
  + Sustituye $a_1$ y $a_2$ en la fórmula de $hat(y)$ y simplifica. ¿Qué tipo de función de $x_1$ y $x_2$ queda?
]

#prof[Solución: $hat(y) = 2, 1, 1, 0$, ya no resuelve XOR. Al simplificar: $hat(y) = (x_1 + x_2) - 2(x_1 + x_2 - 1) = 2 - x_1 - x_2$, una función lineal. Que lo descubran ellos: esta es la idea 2 de la clase. En la sección 2.6 se generaliza.]

== La capa oculta transforma los datos

¿Qué hacen exactamente las neuronas ocultas? Miremos los cuatro ejemplos a través de sus activaciones $(a_1, a_2)$ en lugar de sus entradas $(x_1, x_2)$. Los ejemplos $(0, 1)$ y $(1, 0)$ van al mismo punto, $(1, 0)$; el $(0, 0)$ va al $(0, 0)$, y el $(1, 1)$ va al $(2, 1)$. En este espacio nuevo, una recta sí separa las dos clases: es la frontera $a_1 - 2 a_2 = 0.5$ de la neurona de salida (@fig-xor-espacios).

#figure(
  caption: [A la izquierda, la región donde la red predice $hat(y) > 0.5$ (sombreada), en el espacio de las entradas: es la franja entre las rectas $x_1 + x_2 = 0.5$ y $x_1 + x_2 = 1.5$. A la derecha, los mismos ejemplos en el espacio de las unidades ocultas $(a_1, a_2)$. Los dos positivos caen en el mismo punto, $(1, 0)$, y la recta $a_1 - 2 a_2 = 0.5$ de la neurona de salida separa las clases.],
  grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [Espacio de las entradas #v(0.1cm) #panel-xor({
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
      })],
    [Espacio de las unidades ocultas #v(0.1cm) #panel-xor(
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
            content((1.75, -0.15), text(size: 7.5pt)[$(0,1)$ y $(1,0)$])
            content((0.1, 0.32), text(size: 7.5pt)[$(0,0)$])
            content((2, 1.32), text(size: 7.5pt)[$(1,1)$])
          })
        },
      )],
  ),
) <fig-xor-espacios>

Es decir, las neuronas ocultas hacen lo mismo que hacíamos a mano en la clase 4: transforman los datos, $phi(fvec(x)) = (a_1, a_2)$, de forma que un modelo lineal (la neurona de salida) pueda separarlos con una recta.

== Redes totalmente conectadas

Construimos la red de XOR usando lo que sabíamos del problema: decidimos de antemano qué debía calcular cada neurona oculta. En la mayoría de las aplicaciones no tenemos ese conocimiento. Sería más flexible y más general dejar que cada unidad oculta dependa de *todas* las entradas, con pesos que no fijamos nosotros:

$
  a_1 & = "ReLU"(fvec(w)_1^T fvec(x) + b_1), quad && "donde" fvec(w)_1 in RR^2 "y" b_1 in RR, \
  a_2 & = "ReLU"(fvec(w)_2^T fvec(x) + b_2), quad && "donde" fvec(w)_2 in RR^2 "y" b_2 in RR.
$

La red de XOR es un caso particular, con $fvec(w)_1 = fvec(w)_2 = (1, 1)$, $b_1 = 0$ y $b_2 = -1$. A una red así se le llama *totalmente conectada* (fully connected), porque cada unidad oculta está conectada con todas las entradas (@fig-red-221).

La red de XOR tiene dos *capas* de neuronas: la capa oculta, con $a_1$ y $a_2$, y la capa de salida, con $hat(y)$. Generalicemos esta misma estructura a $d$ entradas y $m$ unidades ocultas. Una red totalmente conectada *de dos capas* con entradas $fvec(x) in RR^d$ y $m$ unidades ocultas se define así. Para cada unidad oculta $j = 1, ..., m$,

$
  z_j = fvec(w)_j^(\(1\)T) fvec(x) + b_j^(\(1\)), quad quad a_j = "ReLU"(z_j), quad quad "donde" fvec(w)_j^(\(1\)) in RR^d "y" b_j^(\(1\)) in RR.
$

Con las activaciones agrupadas en un vector $fvec(a) = (a_1, ..., a_m) in RR^m$, la salida es

$
  hat(y) = fvec(w)^(\(2\)T) fvec(a) + b^(\(2\)), quad quad "donde" fvec(w)^(\(2\)) in RR^m "y" b^(\(2\)) in RR.
$

Los superíndices $(1)$ y $(2)$ distinguen los parámetros de las dos capas: los vectores $fvec(w)_j^(\(1\))$ (uno por unidad oculta, cada uno en $RR^d$) y el vector $fvec(w)^(\(2\))$ (en $RR^m$). En las clases anteriores usábamos el superíndice $(i)$ para numerar los ejemplos de entrenamiento. En estas notas casi siempre trabajamos con un solo ejemplo $fvec(x)$, así que el superíndice indicará la capa. Por ahora tenemos una sola capa oculta; en la sección 2.5 veremos que se pueden apilar tantas como queramos.

#figure(
  caption: [Una red totalmente conectada de dos capas, con 2 entradas, $m = 2$ unidades ocultas y una salida. Cada flecha tiene un peso, y cada unidad oculta y la salida tienen un sesgo.],
  red-diagrama(
    (2, 2, 1),
    etiquetas: ("0-0": $x_1$, "0-1": $x_2$, "1-0": $a_1$, "1-1": $a_2$, "2-0": $hat(y)$),
    nombres: ([entrada], [capa oculta], [salida]),
  ),
) <fig-red-221>

La salida no tiene activación ReLU, porque en general queremos que pueda ser cualquier número. Si queremos una probabilidad, como en la regresión logística, le aplicamos la sigmoide: $hat(y) = g(fvec(w)^(\(2\)T) fvec(a) + b^(\(2\)))$.

Igual que en regresión lineal y logística, los parámetros no los escogemos nosotros: los encuentra el entrenamiento, minimizando una función de pérdida. Lo veremos en la Parte 3.

#prof[Insistir: cada unidad oculta es una regresión logística más (con ReLU). No hay nada nuevo excepto conectarlas. Fin del Bloque 1 (aprox. 0:55). Descanso de 10 minutos.]

== Notación vectorial

#prof[Bloque 2 (1:05).]

Antes de agregar más capas, vamos a escribir la red de dos capas de la sección anterior de forma más compacta, usando matrices y vectores. Hay además una razón práctica: la forma más directa de implementar la red es con un ciclo `for` sobre las unidades ocultas, pero con miles de unidades eso es muy lento. Las librerías de álgebra lineal (y sobre todo las GPU) multiplican matrices muy rápido, así que conviene escribir todo como multiplicaciones de matrices.

Definimos la matriz de pesos $fvec(W)^(\(1\)) in RR^(m times d)$ poniendo los vectores $fvec(w)_j^(\(1\))$ como filas:

$
  fvec(W)^(\(1\)) = mat(
    "—" fvec(w)_1^(\(1\)T) "—";
    "—" fvec(w)_2^(\(1\)T) "—";
    dots.v;
    "—" fvec(w)_m^(\(1\)T) "—"
  ) in RR^(m times d).
$

Por la definición del producto de una matriz por un vector, el vector $fvec(z) = (z_1, ..., z_m)$ se puede escribir como

$
  underbrace(mat(z_1; z_2; dots.v; z_m), fvec(z) in RR^(m times 1)) = underbrace(mat(
    "—" fvec(w)_1^(\(1\)T) "—";
    "—" fvec(w)_2^(\(1\)T) "—";
    dots.v;
    "—" fvec(w)_m^(\(1\)T) "—"
  ), fvec(W)^(\(1\)) in RR^(m times d)) underbrace(mat(x_1; x_2; dots.v; x_d), fvec(x) in RR^(d times 1)) + underbrace(mat(b_1^(\(1\)); b_2^(\(1\)); dots.v; b_m^(\(1\))), fvec(b)^(\(1\)) in RR^(m times 1)),
$

o, de forma más breve,

$
  fvec(z) = fvec(W)^(\(1\)) fvec(x) + fvec(b)^(\(1\)).
$

Para obtener las activaciones aplicamos la ReLU a cada componente de $fvec(z)$ por separado. Si extendemos la notación de forma que, para un vector $fvec(z)$, $"ReLU"(fvec(z))$ sea el vector con componentes $"ReLU"(z_j)$, tenemos

$
  fvec(a) = "ReLU"(fvec(z)).
$

Definiendo de la misma forma $fvec(W)^(\(2\)) = fvec(w)^(\(2\)T) in RR^(1 times m)$, toda la red se resume en dos líneas:

$
  fvec(a) & = "ReLU"(fvec(W)^(\(1\)) fvec(x) + fvec(b)^(\(1\))), \
  hat(y) & = fvec(W)^(\(2\)) fvec(a) + b^(\(2\)).
$

A $fvec(W)^(\(1\))$ y $fvec(W)^(\(2\))$ se les llama matrices de pesos, y a $fvec(b)^(\(1\))$ y $b^(\(2\))$, sesgos. El par $fvec(W)^(\(1\)), fvec(b)^(\(1\))$ forma la primera *capa* de la red y $fvec(W)^(\(2\)), b^(\(2\))$ la segunda. Al vector de activaciones $fvec(a)$ se le llama la *capa oculta*. Cada capa hace lo mismo: multiplica su entrada por una matriz de pesos, suma un vector de sesgos y aplica la activación. En la siguiente sección usaremos esta observación para apilar más capas.

Por ejemplo, la red de XOR de la @fig-red-xor es

$
  fvec(W)^(\(1\)) = mat(1, 1; 1, 1), quad fvec(b)^(\(1\)) = mat(0; -1), quad fvec(W)^(\(2\)) = mat(1, -2), quad b^(\(2\)) = 0.
$

Para $fvec(x) = (1, 1)$, el forward pass es

$
  fvec(z) = mat(1, 1; 1, 1) mat(1; 1) + mat(0; -1) = mat(2; 1), quad fvec(a) = "ReLU"(fvec(z)) = mat(2; 1), quad hat(y) = mat(1, -2) mat(2; 1) + 0 = 0,
$

que es lo mismo que calculamos neurona por neurona en el Ejercicio 1.

== Redes con más capas

Con esta notación es fácil apilar más capas. Sea $r$ el número de capas, y sean $fvec(W)^(\(1\)), ..., fvec(W)^(\(r\))$ y $fvec(b)^(\(1\)), ..., fvec(b)^(\(r\))$ las matrices de pesos y los sesgos de cada capa. Una red totalmente conectada con $r$ capas es

$
  fvec(a)^(\(1\)) & = "ReLU"(fvec(W)^(\(1\)) fvec(x) + fvec(b)^(\(1\))), \
  fvec(a)^(\(2\)) & = "ReLU"(fvec(W)^(\(2\)) fvec(a)^(\(1\)) + fvec(b)^(\(2\))), \
  & dots.v \
  fvec(a)^(\(r-1\)) & = "ReLU"(fvec(W)^(\(r-1\)) fvec(a)^(\(r-2\)) + fvec(b)^(\(r-1\))), \
  hat(y) & = fvec(W)^(\(r\)) fvec(a)^(\(r-1\)) + fvec(b)^(\(r\)).
$

Las dimensiones tienen que ser compatibles. Si la capa $k$ tiene $m_k$ unidades, es decir, $fvec(a)^(\(k\)) in RR^(m_k)$, entonces $fvec(W)^(\(k\))$ es de $m_k times m_(k-1)$ y $fvec(b)^(\(k\)) in RR^(m_k)$. Si escribimos además $fvec(a)^(\(0\)) = fvec(x)$ (con $m_0 = d$), cada capa oculta se calcula a partir de la anterior con la misma regla:

$
  fvec(a)^(\(k\)) = "ReLU"(fvec(W)^(\(k\)) fvec(a)^(\(k-1\)) + fvec(b)^(\(k\))), quad k = 1, ..., r - 1.
$

La última capa normalmente no lleva ReLU, para que la salida pueda ser negativa (o se le aplica la sigmoide o softmax, como veremos, si queremos probabilidades). A estas redes se les llama también *perceptrones multicapa* (MLP).

*Conteo de parámetros.* La capa $k$ tiene $m_k m_(k-1)$ pesos y $m_k$ sesgos, es decir, $(m_(k-1) + 1) m_k$ parámetros. Por ejemplo, consideremos una red para reconocer dígitos escritos a mano. Cada imagen tiene $28 times 28 = 784$ píxeles, así que $d = 784$. Supongamos que la red tiene dos capas ocultas de 16 unidades y una salida con 10 unidades, una por dígito:

#align(center, table(
  columns: 4,
  align: (left, center, center, center),
  table.header([Capa], [Pesos], [Sesgos], [Total]),
  [1 ($784 -> 16$)], $784 dot 16 = 12544$, $16$, $12560$,
  [2 ($16 -> 16$)], $16 dot 16 = 256$, $16$, $272$,
  [3 ($16 -> 10$)], $16 dot 10 = 160$, $10$, $170$,
  [*Total*], [], [], [*13002*],
))

Entrenar esta red significa encontrar valores para esos 13002 números. Las redes que se usan hoy tienen millones o miles de millones de parámetros, pero la idea es la misma.

#prof[Hacer el conteo en la pizarra con ellos, capa por capa (es el ejercicio 3 del notebook).]

¿Para qué sirven varias capas? La intuición es que cada capa construye features a partir de las de la capa anterior. En imágenes, por ejemplo, las neuronas de la primera capa podrían detectar bordes pequeños, las de la segunda podrían combinar bordes en trazos, y las de la tercera combinar trazos en partes de un dígito, como un círculo o una línea vertical. Es una intuición útil, pero hay que tomarla con cuidado: en redes entrenadas de verdad, las neuronas casi nunca aprenden patrones tan limpios ni tan fáciles de interpretar.

== ¿Por qué no usar la identidad como activación?

En el Ejercicio 2 vimos que, sin la ReLU, la red de XOR se convierte en una función lineal. Esto es cierto para cualquier red. Supongamos que la activación es la identidad, $"ReLU"(z)$ reemplazada por $z$, y, para simplificar, que los sesgos son cero. Entonces, para una red de dos capas,

$
  hat(y) & = fvec(W)^(\(2\)) fvec(a)^(\(1\)) \
  & = fvec(W)^(\(2\)) fvec(z)^(\(1\)) quad && "porque la activación es la identidad" \
  & = fvec(W)^(\(2\)) fvec(W)^(\(1\)) fvec(x) \
  & = tilde(fvec(W)) fvec(x), quad && "donde" tilde(fvec(W)) = fvec(W)^(\(2\)) fvec(W)^(\(1\)).
$

Las dos matrices se colapsan en una sola: aplicar una función lineal a otra función lineal da una función lineal de la entrada original. Con más capas pasa lo mismo, y con sesgos el resultado es $tilde(fvec(W)) fvec(x) + tilde(fvec(b))$. Una red de cien capas sin activaciones es equivalente a una sola neurona, y su frontera sigue siendo una recta. *La activación no lineal es lo que le da a la red su capacidad de representar funciones no lineales.*

== La red aprende las features

En la clase 4 vimos que, con un feature map $phi(fvec(x))$, un modelo lineal $theta^T phi(fvec(x))$ puede representar fronteras no lineales. El problema es escoger $phi$: el desempeño depende mucho de esa elección, y normalmente se usa conocimiento del problema para diseñarla. A este proceso se le llama *ingeniería de features* (feature engineering).

Las redes neuronales se pueden ver como una forma de *aprender* el feature map. Llamemos $beta$ a todos los parámetros de la red excepto los de la última capa. La última capa oculta, $fvec(a)^(\(r-1\))$, es una función de $fvec(x)$ que depende de $beta$; escribámosla $phi_beta (fvec(x))$. Entonces el modelo es

$
  hat(y) = fvec(W)^(\(r\)) phi_beta (fvec(x)) + fvec(b)^(\(r\)).
$

Si $beta$ estuviera fijo, esto sería un modelo lineal sobre las features $phi_beta (fvec(x))$, exactamente como en la clase 4. Pero al entrenar la red se optimizan a la vez $beta$ y la última capa: no solo aprendemos un modelo lineal sobre las features, sino también las features mismas. Esto es lo que vimos en la @fig-xor-espacios, ahora sin que nadie escoja los pesos. Por eso las redes neuronales dependen menos del conocimiento del problema. La contraparte es que las features que encuentra la red suelen ser difíciles de interpretar para una persona, y por eso a veces se dice que las redes son "cajas negras".

La @fig-lunas muestra el efecto con datos más realistas. La regresión logística solo puede trazar una recta. Una red con una capa oculta de 3 unidades ReLU ya dobla la frontera, y con 32 unidades la frontera sigue la forma de los datos.

#let panel-lunas(p, y-label: none) = custom-plot(
  x-min: d-lunas.limites.at(0),
  x-max: d-lunas.limites.at(1),
  y-min: d-lunas.limites.at(2),
  y-max: d-lunas.limites.at(3),
  x-label: $x_1$,
  y-label: y-label,
  x-tick-step: 1,
  y-tick-step: 1,
  size: (4.4, 3.4),
  {
    puntos-clases(d-lunas, tam: 0.07)
    for s in p.frontera { curva(s, grosor: 1.4pt) }
  },
)

#figure(
  caption: [Fronteras de decisión ($hat(y) = 0.5$) aprendidas sobre 200 ejemplos. A la izquierda, regresión logística (una neurona). En el centro y a la derecha, redes con una capa oculta de 3 y de 32 unidades ReLU (#d-lunas.paneles.h3.parametros y #d-lunas.paneles.h32.parametros parámetros). La exactitud sobre datos de prueba es #pct(d-lunas.paneles.logistica.exactitud), #pct(d-lunas.paneles.h3.exactitud) y #pct(d-lunas.paneles.h32.exactitud).],
  [
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 0.3cm,
      [Regresión logística #panel-lunas(d-lunas.paneles.logistica, y-label: $x_2$)],
      [3 unidades ocultas #panel-lunas(d-lunas.paneles.h3)],
      [32 unidades ocultas #panel-lunas(d-lunas.paneles.h32)],
    )
    #v(0.2cm)
    #leyenda-clases
  ],
) <fig-lunas>

== Varias clases: softmax

Con dos clases basta una unidad de salida con sigmoide. Si hay $K$ clases, como los diez dígitos, la última capa tiene $K$ unidades, una por clase, con scores $z_1, ..., z_K$. Para convertir los scores en probabilidades que sumen 1 se usa la función *softmax*:

$
  hat(y)_k = e^(z_k) / (sum_(j=1)^K e^(z_j)), quad quad k = 1, ..., K.
$

La exponencial hace que todos los valores sean positivos, y dividir entre la suma hace que sumen 1. La clase predicha es la de mayor probabilidad, que es también la de mayor score. Por ejemplo, con tres clases y scores $fvec(z) = (2, 1, -1)$, las probabilidades son aproximadamente $(0.71, 0.26, 0.04)$. Con dos clases, softmax da lo mismo que la sigmoide aplicada a la diferencia de los scores, así que softmax es la generalización de la sigmoide a más de dos clases.

#prof[5 minutos como máximo. Basta con que sepan que en MNIST la salida tiene 10 unidades y que softmax las convierte en probabilidades.]

== (Opcional) Vectorización sobre varios ejemplos

#prof[OPCIONAL. Dar solo si el grupo entendió bien la notación vectorial y van a tiempo (unos 10 min). Si no, saltar a la demo del Playground y al lab. Lo importante para el lab es la última oración de la sección: PyTorch espera un ejemplo por fila.]

En la sección anterior vectorizamos sobre las unidades de una capa, pero seguimos procesando un ejemplo a la vez. En la práctica se procesan muchos ejemplos al mismo tiempo.

Pongamos $n$ ejemplos como filas de una matriz $fvec(X) in RR^(n times d)$, como en las clases anteriores. Para un ejemplo $fvec(x)$, la primera capa calcula $fvec(z) = fvec(W)^(\(1\)) fvec(x) + fvec(b)^(\(1\))$. Transponiendo, $fvec(z)^T = fvec(x)^T fvec(W)^(\(1\)T) + fvec(b)^(\(1\)T)$, y como las filas de $fvec(X)$ son los $fvec(x)^T$, los scores de todos los ejemplos se obtienen con una sola multiplicación:

$
  underbrace(fvec(Z)^(\(1\)), n times m_1) = underbrace(fvec(X), n times d) quad underbrace(fvec(W)^(\(1\)T), d times m_1) + underbrace(fvec(b)^(\(1\)T), 1 times m_1), quad quad fvec(A)^(\(1\)) = "ReLU"(fvec(Z)^(\(1\))).
$

La fila $i$ de $fvec(Z)^(\(1\))$ contiene los scores de la primera capa para el ejemplo $i$. Las demás capas son iguales, reemplazando $fvec(X)$ por $fvec(A)^(\(k-1\))$.

La suma del final tiene un detalle: $fvec(X) fvec(W)^(\(1\)T)$ es una matriz de $n times m_1$, pero $fvec(b)^(\(1\)T)$ es una sola fila. Lo que queremos es sumarle esa fila a cada una de las $n$ filas, porque todos los ejemplos usan los mismos sesgos. NumPy y PyTorch lo hacen automáticamente, copiando la fila $n$ veces. A esto se le llama *broadcasting*. Los sesgos no se multiplican: siguen siendo $m_1$ parámetros.

En las notas de CS229 los ejemplos se ponen como columnas y la fórmula queda $fvec(Z) = fvec(W) fvec(X) + fvec(b)$; es lo mismo, transpuesto. Nosotros usamos filas porque es la convención de scikit-learn y de PyTorch: en PyTorch, `nn.Linear(d, m)` guarda una matriz de pesos de $m times d$ y calcula exactamente `X @ W.T + b`.

#prof[Demo del Playground (unos 10 min). Antes de cada cambio, que la clase vote qué va a pasar: (1) dataset XOR, 2 neuronas ocultas, entrenar; (2) bajar a 1 neurona; (3) activación "Linear": la frontera vuelve a ser una recta (lo mismo del Ejercicio 2); (4) dataset espiral: ¿qué necesitamos? más capas y neuronas; (5) learning rate 3 o 10: la pérdida salta y no baja. Después: abrir labs/05/lab.ipynb, Parte 1 (ejercicios 1 a 3), unos 20 min.]

= Parte 3: Cómo aprende una red

#prof[Bloque 3 (1:55). Mensaje central: no hay nada nuevo. Es el descenso por gradiente de la clase 2 aplicado a muchos parámetros a la vez.]

Hasta ahora escogimos los parámetros a mano. Con 13000 parámetros eso es imposible. Igual que en regresión lineal y logística, los vamos a encontrar minimizando una función de pérdida con descenso por gradiente.

== La pérdida

La *pérdida* $L(hat(y), y)$ mide qué tan lejos está la predicción de la etiqueta en un ejemplo, y el *costo* $J$ es el promedio de la pérdida sobre los $n$ ejemplos de entrenamiento:

$
  J = 1 / n sum_(i=1)^n L(hat(y)^(\(i\)), y^(\(i\))).
$

(Aquí, excepcionalmente, el superíndice $(i)$ numera los ejemplos.) La pérdida depende del problema, y ya las conocemos todas:

- *Regresión:* el error cuadrático, $L = (hat(y) - y)^2$, como en regresión lineal. En PyTorch, `nn.MSELoss`.
- *Clasificación binaria:* la *entropía cruzada binaria*,
  $
    L = -[y log hat(y) + (1 - y) log(1 - hat(y))],
  $
  que es exactamente el negativo de la log-verosimilitud de un ejemplo en la regresión logística (clase 4). Allí maximizábamos la log-verosimilitud con ascenso por gradiente; aquí minimizamos su negativo con descenso por gradiente, que es lo mismo. En PyTorch, `nn.BCEWithLogitsLoss`, que aplica la sigmoide por nosotros.
- *Varias clases:* la *entropía cruzada*, $L = -log hat(y)_c$, donde $c$ es la clase correcta: queremos que la probabilidad que la red le asigna a la clase correcta sea alta. En PyTorch, `nn.CrossEntropyLoss`, que aplica softmax por nosotros.

== Descenso por gradiente

Agrupemos todos los pesos y sesgos de la red en un vector $theta$. El descenso por gradiente repite

$
  theta := theta - alpha nabla_theta J,
$

donde $alpha$ es la tasa de aprendizaje. Cada componente del gradiente dice cuánto cambia el costo si movemos un poco ese parámetro, y en qué dirección. Restar el gradiente mueve cada parámetro en la dirección en la que el costo baja. En la práctica se usa la versión con mini-batches: en cada paso, el gradiente se calcula con un grupo pequeño de ejemplos escogidos al azar.

Antes de ver cómo se calcula el gradiente de una red, hagamos algunos pasos a mano con la red más pequeña posible: una neurona sin activación, sin sesgo y con un solo peso, $hat(y) = w x$, un solo ejemplo $x = 2$, $y = 10$, y pérdida $L = (hat(y) - y)^2$ (@fig-red-un-peso).

#figure(
  caption: [La red más pequeña posible: una neurona sin activación, sin sesgo y con un solo peso $w$.],
  red-diagrama(
    (1, 1),
    etiquetas: ("0-0": $x$, "1-0": $hat(y)$),
    pesos: ("0-0-0": $w$),
    pos-peso: 0.5,
    nombres: ([$x = 2$], [$hat(y) = w x$]),
    dx: 4,
  ),
) <fig-red-un-peso>

Para encontrar la derivada de $L$ respecto a $w$, notemos que $L$ no depende de $w$ directamente, sino a través de $hat(y)$: el peso determina la predicción, y la predicción determina la pérdida. En estos casos se usa la *regla de la cadena*: si $L$ depende de $hat(y)$ y $hat(y)$ depende de $w$, entonces

$
  (d L) / (d w) = (d L) / (d hat(y)) dot (d hat(y)) / (d w).
$

Es decir, cuánto cambia $L$ al mover $w$ es cuánto cambia $L$ al mover $hat(y)$, multiplicado por cuánto cambia $hat(y)$ al mover $w$. Calculemos cada factor por separado. Como $L = (hat(y) - y)^2$ y $hat(y) = w x$,

$
  (d L) / (d hat(y)) = 2 (hat(y) - y), quad quad (d hat(y)) / (d w) = x.
$

Multiplicando,

$
  (d L) / (d w) = 2 (hat(y) - y) x.
$

En nuestro ejemplo, con $x = 2$, $(d L) / (d w) = 4 (hat(y) - 10)$. Vamos a volver a usar esta misma idea en la siguiente sección: backpropagation no es más que la regla de la cadena aplicada a muchos pasos seguidos.

#ejercicio("Ejercicio 3 ")[
  Empezando con $w = 1$ y $alpha = 0.05$, completa tres pasos de descenso por gradiente.

  #align(center, table(
    columns: (auto, 1.6cm, 2cm, 2.6cm, 3.4cm),
    align: center + horizon,
    rows: (auto, 0.75cm),
    table.header([Paso], $w$, $hat(y) = 2 w$, $L = (hat(y) - 10)^2$, $(d L) / (d w) = 4(hat(y) - 10)$),
    $0$, $1$, [], [], [],
    $1$, [], [], [], [],
    $2$, [], [], [], [],
    $3$, [], [], [], [],
  ))

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

#prof[Solución: $w = 1, 2.6, 3.56, 4.136$; $hat(y) = 2, 5.2, 7.12, 8.272$; $L = 64, 23.04, 8.29, 2.99$; gradiente $= -32, -19.2, -11.52, -6.91$. Se acerca a $w = 5$, porque $2 dot 5 = 10$. El primer gradiente, −32, lo vamos a volver a ver en el lab: PyTorch lo calcula solo.]

La @fig-descenso muestra cómo evoluciona $w$ con dos tasas de aprendizaje. Con $alpha = 0.05$, $w$ se acerca a 5 poco a poco. Con $alpha = 0.3$, cada paso se pasa del mínimo, cada vez más lejos, y el descenso diverge. Es lo mismo que vimos en la clase 2, y es el error más común al entrenar una red: si la pérdida sube o se vuelve `nan`, lo primero es bajar la tasa de aprendizaje.

#figure(
  caption: [El peso $w$ en los primeros pasos de descenso por gradiente, con $alpha = 0.05$ (converge a 5) y con $alpha = 0.3$ (oscila alrededor de 5, cada vez más lejos).],
  [
    #custom-plot(
      x-min: 0,
      x-max: 10,
      y-min: -4,
      y-max: 17,
      x-label: [paso],
      y-label: $w$,
      x-tick-step: 2,
      y-tick-step: 5,
      size: (7, 3.8),
      {
        plot.add-hline(5, style: (stroke: (paint: gray, thickness: 0.6pt, dash: "dotted")))
        plot.add(pasos-w(0.05, 10), mark: "o", mark-size: 0.1, style: (stroke: (paint: c-neg, thickness: 1.4pt)), mark-style: (fill: c-neg, stroke: none))
        plot.add(pasos-w(0.3, 3), mark: "o", mark-size: 0.1, style: (stroke: (paint: c-pos, thickness: 1.4pt, dash: "dashed")), mark-style: (fill: c-pos, stroke: none))
      },
    )
    #v(0.2cm)
    #leyenda-lineas(linea-leyenda(c-neg, $alpha = 0.05$), linea-leyenda(c-pos, $alpha = 0.3$, punteada: true))
  ],
) <fig-descenso>

== Backpropagation: el error viaja hacia atrás

Con una neurona, la derivada sale directo. En una red, en cambio, un peso de la primera capa afecta a la pérdida a través de todas las capas que vienen después. *Backpropagation* (propagación hacia atrás) es el algoritmo que calcula todas esas derivadas de forma eficiente. No es un método de aprendizaje distinto: el aprendizaje sigue siendo descenso por gradiente, y backpropagation es solo la forma de calcular el gradiente.

Veamos la idea con una red de dos pesos: una unidad oculta con ReLU y una salida sin activación,

$
  a = "ReLU"(w_1 x), quad quad hat(y) = w_2 a, quad quad L = (hat(y) - y)^2,
$

con $x = 2$, $y = 10$, $w_1 = 1$ y $w_2 = 1$ (@fig-red-dos-pesos). El forward pass da $a = 2$, $hat(y) = 2$ y $L = 64$.

#figure(
  caption: [Una red de dos pesos: una unidad oculta con ReLU, $a = "ReLU"(w_1 x)$, y una salida sin activación, $hat(y) = w_2 a$.],
  red-diagrama(
    (1, 1, 1),
    etiquetas: ("0-0": $x$, "1-0": $a$, "2-0": $hat(y)$),
    pesos: ("0-0-0": $w_1$, "1-0-0": $w_2$),
    pos-peso: 0.5,
    nombres: ([entrada], [oculta (ReLU)], [salida]),
    dx: 3,
  ),
) <fig-red-dos-pesos>

Ahora vamos de derecha a izquierda, preguntándonos en cada paso cuánto cambia la pérdida si cambia ese valor:

+ *En la salida.* $(partial L) / (partial hat(y)) = 2 (hat(y) - y) = -16$. Esta es la "señal de error". Es negativa porque la predicción es demasiado pequeña: si $hat(y)$ aumenta, la pérdida baja.
+ *El peso $w_2$.* Como $hat(y) = w_2 a$, mover $w_2$ mueve $hat(y)$ en proporción a $a$. Entonces $(partial L) / (partial w_2) = -16 dot a = -32$.
+ *La unidad oculta.* Mover $a$ mueve $hat(y)$ en proporción a $w_2$. La señal de error llega a $a$ multiplicada por el peso de la conexión: $(partial L) / (partial a) = -16 dot w_2 = -16$.
+ *La ReLU.* Como $w_1 x = 2 > 0$, la ReLU deja pasar los cambios tal cual (su pendiente es 1). Si la neurona estuviera apagada ($w_1 x < 0$), la pendiente sería 0 y la señal se detendría aquí.
+ *El peso $w_1$.* Igual que en el paso 2: $(partial L) / (partial w_1) = (partial L) / (partial a) dot 1 dot x = -16 dot 2 = -32$.

Ese es todo el algoritmo. El error se calcula en la salida y viaja hacia atrás, capa por capa. Al cruzar cada conexión se multiplica por su peso, y al cruzar cada activación se multiplica por su pendiente. Cuando la señal llega a un peso, la derivada de ese peso es la señal multiplicada por la entrada que recibe ese peso. Es la regla de la cadena, aplicada de forma ordenada y reutilizando los cálculos que ya hicimos.

Hay una forma intuitiva de leer esto. Si la salida debería ser más grande, cada conexión "pide" un cambio en la capa anterior, en la dirección que más ayudaría a la salida, y esos pedidos se pasan hacia atrás, capa por capa. Como la derivada de un peso es proporcional a la entrada que recibe, las conexiones que vienen de neuronas muy activas son las que más influyen en la salida, y por eso son las que más cambian. Las que vienen de neuronas apagadas no cambian.

#ejercicio("Ejercicio 4 ")[
  Para la red de dos pesos de la @fig-red-dos-pesos ($x = 2$, $y = 10$), calcula el forward pass y las dos derivadas $(partial L) / (partial w_1)$ y $(partial L) / (partial w_2)$ si empezamos con $w_1 = -1$ y $w_2 = 1$. ¿Qué pasa con $w_1$ si damos un paso de descenso por gradiente? ¿Por qué?
]

#prof[Solución: $w_1 x = -2$, así que $a = 0$, $hat(y) = 0$, $L = 100$. La señal de error es $2(0 - 10) = -20$. $(partial L) / (partial w_2) = -20 dot a = 0$ y $(partial L) / (partial w_1) = -20 dot w_2 dot 0 dot x = 0$, porque la ReLU está apagada. Ningún peso cambia: la neurona está "muerta" para este ejemplo. Es una buena excusa para explicar por qué en el lab usamos más neuronas de las estrictamente necesarias.]

En la práctica nadie calcula estas derivadas a mano. Las librerías como PyTorch registran las operaciones del forward pass y, con una sola instrucción, recorren el camino hacia atrás y calculan el gradiente de la pérdida respecto a cada parámetro. A esto se le llama *diferenciación automática* (autograd).

== Entrenar una red en PyTorch

Un entrenamiento en PyTorch tiene siempre la misma estructura, con cinco líneas que se repiten en cada paso:

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

La @fig-perdida muestra la pérdida durante el entrenamiento de una red con una capa oculta de 32 unidades sobre los datos de la @fig-lunas. Baja rápido al principio y luego cada vez más despacio, como en regresión lineal. Igual que con los árboles, que la pérdida de entrenamiento baje no garantiza que la red generalice: una red grande también puede memorizar los datos. Eso lo veremos la próxima clase.

#figure(
  caption: [Entropía cruzada de entrenamiento de una red con 32 unidades ocultas sobre los datos de la @fig-lunas, en función del número de pasos de descenso por gradiente ($alpha = 0.1$). La exactitud final sobre datos de prueba es #pct(d-perdida.exactitud).],
  custom-plot(
    x-min: 0,
    x-max: 1000,
    y-min: 0,
    y-max: 0.8,
    x-label: [paso],
    y-label: [pérdida],
    x-tick-step: 200,
    y-tick-step: 0.2,
    size: (7, 3.6),
    curva(d-perdida.perdida, color: c-pos, grosor: 1.5pt),
  ),
) <fig-perdida>

#prof[Ejercicio 4 en papel (aprox. 10 min) y después el lab, Parte 2 (ejercicios 4 y 5 del notebook, unos 25 min). El momento clave: `w.grad` da −32, lo mismo que calcularon a mano en el Ejercicio 3. Hacerlo notar en voz alta.]

== (Opcional) Backpropagation en general

#prof[OPCIONAL, para leer en casa. No darlo en clase.]

Para una red con $r$ capas, llamemos $fvec(delta)^(\(k\)) = partial L slash partial fvec(z)^(\(k\))$ a la señal de error que llega a los scores de la capa $k$, donde $fvec(z)^(\(k\)) = fvec(W)^(\(k\)) fvec(a)^(\(k-1\)) + fvec(b)^(\(k\))$. Backpropagation calcula, para un ejemplo:

+ En la salida, con sigmoide y entropía cruzada binaria (o con softmax y entropía cruzada): $fvec(delta)^(\(r\)) = hat(fvec(y)) - fvec(y)$. Es el mismo $h - y$ que apareció en el gradiente de la regresión logística en la clase 4.
+ Para cada capa, de la última a la primera:
  $
    (partial L) / (partial fvec(W)^(\(k\))) = fvec(delta)^(\(k\)) fvec(a)^(\(k-1\)T), quad quad (partial L) / (partial fvec(b)^(\(k\))) = fvec(delta)^(\(k\)), quad quad fvec(delta)^(\(k-1\)) = (fvec(W)^(\(k\)T) fvec(delta)^(\(k\))) dot.o g'(fvec(z)^(\(k-1\))),
  $
  donde $dot.o$ es el producto elemento a elemento y $g'$ es la derivada de la activación.

Son las mismas tres reglas del ejemplo de dos pesos, escritas para capas completas: la derivada de un peso es la señal que llega multiplicada por la entrada que recibe; al cruzar las conexiones, la señal se multiplica por los pesos ($fvec(W)^T$); y al cruzar la activación, se multiplica por su pendiente ($g'$). Para la ReLU, $g'(z)$ es 1 si $z > 0$ y 0 si no.

= Resumen

Las tres ideas de la clase:

- *Una neurona es una regresión logística:* una suma ponderada de las entradas más un sesgo, $z = fvec(w)^T fvec(x) + b$, seguida de una activación, $a = g(z)$. Las activaciones más usadas son la sigmoide y la ReLU, $max(z, 0)$.
- *Apilar neuronas con activaciones no lineales produce fronteras no lineales.* Las capas ocultas transforman los datos en features nuevas, y la última capa es un modelo lineal sobre esas features: la red aprende el $phi(fvec(x))$ que en la clase 4 diseñábamos a mano. Sin activaciones no lineales, cualquier red, por grande que sea, es equivalente a un modelo lineal.
- *Una red aprende con descenso por gradiente, y backpropagation es la forma de calcular el gradiente.* El error se calcula en la salida y viaja hacia atrás capa por capa, multiplicándose por los pesos y por las pendientes de las activaciones. En PyTorch lo hace `loss.backward()`.

Además:

- Con notación vectorial, cada capa calcula $fvec(a)^(\(k\)) = "ReLU"(fvec(W)^(\(k\)) fvec(a)^(\(k-1\)) + fvec(b)^(\(k\)))$, donde la fila $j$ de $fvec(W)^(\(k\))$ son los pesos de la unidad $j$. Una capa con $m_(k-1)$ entradas y $m_k$ unidades tiene $(m_(k-1) + 1) m_k$ parámetros.
- Con varias clases, la salida tiene una unidad por clase y softmax convierte los scores en probabilidades.
- La pérdida es el error cuadrático en regresión y la entropía cruzada en clasificación (el negativo de la log-verosimilitud).

#prof[Cierre (2:50): leer las tres ideas en voz alta y pedir a dos o tres alumnos que las expliquen con sus palabras. Presentar la Tarea 4.]
