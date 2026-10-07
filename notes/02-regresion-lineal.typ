// OUTLINE DE ESTA CLASE:
// 1. Bias Variance tradeoff
// 2. Descripcion rapido de un problema de machine learning: funcion de perdida, optimizacion
// 3. Primer algoritmo: regresion lineal
// 3.1. Descripcion del problema: Ejemplo con casas
// 3.2. Descripcion de la funcion objetivo: SSE
// 3.3. Derivada de la funcion objetivo: Gradiente
// 3.4. Algoritmo de optimizacion: Descenso por gradiente
// 3.5. Descripcion: BGD, SGD, MBGD
// 3.6. Descripcion rapido de Matrix Calculus: Gradiente, Jacobiano, denominator layout, identidades basicas para OLS
// 3.7 Solucion analitica: OLS
// 4. Matrix calculus: revisitado
// 5. Laboratorio: Numpy, Matplotlib, Pandas, Scikit-learn

#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *
#import "@preview/suiji:0.5.1": *

#let fvec(x) = math.bold(math.upright(x))
#let rng = gen-rng(42)

#set text(lang: "es")

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
  title: [Regresión lineal e introducción al aprendizaje supervisado],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

= Parte 1: Estructura de un problema de Aprendizaje Supervisado

En el dominio del aprendizaje supervisado, se busca aprender una función $fvec(x) mapsto y$ a partir de un conjunto de entrenamiento $italic(S) = {(fvec(x)^(\(i\)), y^(\(i\)))}_(i=1)^m$. Es decir, cada observación $i$ del conjunto de entrenamiento está compuesta por un vector de características $fvec(x)^(\(i\))$ y una etiqueta $y^(\(i\))$. El objetivo es encontrar una función que pueda predecir la etiqueta $y$ para nuevas observaciones basadas en sus características $fvec(x)$.

== Regresión vs Clasificación

Toda variable puede ser de tipo cuantitativa o cualitativa (también llamada categórica).

Ejemplos de variables cuantitativas: precio de una casa, temperatura, altura de una persona, etc.

Ejemplos de variables cualitativas: el estado civil de una persona, el color de un auto, si una persona incumple el pago de un crédito, etc.

A los problemas supervisados asociados a predecir variables cuantitativas se les llama problemas de *regresión*, mientras que a los problemas supervisados asociados a predecir variables cualitativas se les llama problemas de *clasificación*.

Los métodos a utilizar dependen del tipo de variable que se desea predecir. Por ejemplo, para problemas de regresión se pueden utilizar la regresión lineal, mientras que para problemas de clasificación se pueden utilizar árboles de decisión, o regresión logística

Por otro lado, el tipo de las variables predictoras no tiene tanta influencia en la elección del método a utilizar.

== Midiendo la calidad de un modelo

Para poder evaluar el desempeño de un modelo, se debe definir una métrica que mida la calidad de sus predicciones (cuantificando que tan cerca de la predicción está el valor real). Por ejemplo, para problemas de regresión la métrica más utilizada es el error cuadrático medio (MSE).

$
  "MSE" = frac(1, m) sum_(i=1)^m (y^(\(i\)) - hat(y)^(\(i\)))^2
$

Cuando se computa el MSE sobre el conjunto de entrenamiento, se le llama *error de entrenamiento*, mientras que cuando se computa sobre un conjunto de prueba, se le llama *error de prueba*.

En general, el error de entrenamiento no nos interesa tanto como el error de prueba. Es decir, queremos saber *la precisión de las predicciones ante datos no vistos*.

- Para un modelo del mercado de valores, queremos saber que tan bien predice el precio de una acción en el futuro, no que tan bien predice el precio de una acción en el pasado.

De forma precisa, queremos el modelo que nos de el menor MSE de prueba, incluso si eso significa que el MSE de entrenamiento es mayor. Veamos un ejemplo en la siguiente figura:

#figure(
  caption: [A la izquierda, se muestran varios ejemplos de funciones que podrían modelar los datos de entrenamiento. La función azul es un ejemplo de *underfitting*, la función verde es un ejemplo de *ajuste intermedio* y la función roja es un ejemplo de *overfitting*. A la derecha, se muestra el error cuadrático medio de entrenamiento (línea gris punteada) y prueba (línea roja) en función de la flexibilidad del modelo.],
  grid(
    columns: (1fr, 1fr),
    gutter: 1.2cm,
    align: top + center,

    [
      #custom-plot(
        x-min: 0,
        x-max: 5,
        y-min: 0,
        y-max: 5,
        x-label: $x$,
        y-label: $y$,
        x-tick-step: 1,
        y-tick-step: 1,
        size: (6, 5),
        {
          // Función de sobreajuste
          let f-overfit(x) = 0.8 + 0.75 * x + 0.55 * calc.sin(x * 4.2)
          let f-true(x) = 0.8 + 0.65 * x + 0.05 * calc.pow(x, 2)

          // Puntos evaluados exactamente en la función anterior
          let x-coords = (0.7, 1.3, 1.8, 2.3, 2.8, 3.4, 3.9, 4.3)
          let exact-data = x-coords.map(x => (x, f-overfit(x)))
          let test-data = (
            (0.9, f-true(0.9) + 0.35),
            (1.5, f-true(1.5) - 0.40),
            (2.0, f-true(2.0) + 0.45),
            (2.5, f-true(2.5) - 0.30),
            (3.1, f-true(3.1) + 0.50),
            (3.6, f-true(3.6) - 0.35),
            (4.1, f-true(4.1) + 0.25),
          )

          // 1. Data Points
          plot.add(
            exact-data,
            mark: "o",
            mark-size: 0.16,
            style: (stroke: none),
            mark-style: (stroke: none, fill: black),
          )

          // 1.1. Función real puntos
          plot.add(
            test-data,
            mark: "o",
            mark-size: 0.16,
            style: (stroke: none),
            mark-style: (stroke: none, fill: gray.darken(40%)),
          )

          // 2. Modelo 1: Underfitting (Línea recta)
          plot.add(
            domain: (0.0, 5),
            x => 0.6 + 0.65 * x,
            style: (stroke: (paint: blue, thickness: 1.5pt, dash: "dashed")),
          )

          // 3. Modelo 2: Ajuste intermedio / suave
          plot.add(
            domain: (0.0, 5),
            x => 0.8 + 0.65 * x + 0.05 * calc.pow(x, 2),
            style: (stroke: (paint: green.darken(20%), thickness: 1.8pt)),
          )

          // 4. Modelo 3: Overfitting (Pasa exactamente por todos los puntos)
          plot.add(
            domain: (0.0, 5),
            samples: 150,
            f-overfit,
            style: (stroke: (paint: red, thickness: 1.3pt)),
          )
        },
      )
    ],

    [
      #custom-plot(
        x-min: 0,
        x-max: 5,
        y-min: 0,
        y-max: 5,
        x-label: [Flexibilidad],
        y-label: [MSE],
        x-tick-step: 1,
        y-tick-step: 1,
        size: (6, 5),
        {
          // Training MSE: Monótonamente decreciente conforme aumenta la flexibilidad
          plot.add(
            domain: (0.0, 5),
            x => 2 * calc.exp(-0.7 * x) + 0.15,
            style: (stroke: (paint: gray.darken(40%), thickness: 1.5pt, dash: "dashed")),
          )

          // Test MSE: Curva en 'U' clásica del error cuadrático medio de prueba
          plot.add(
            domain: (0.0, 5),
            x => 1.8 * calc.exp(-0.85 * x) + 0.08 * calc.exp(0.85 * x) + 0.3,
            style: (stroke: (paint: red.darken(20%), thickness: 2pt)),
          )
        },
      )
    ],
  ),
)

En la figura anterior, se puede observar que el modelo azul (underfitting) tiene un MSE de entrenamiento relativamente alto y un MSE de prueba también alto. El modelo verde (ajuste intermedio) tiene un MSE de entrenamiento más bajo y un MSE de prueba más bajo que el modelo azul. Finalmente, el modelo rojo (overfitting) tiene un MSE de entrenamiento muy bajo, pero su MSE de prueba es más alto que el del modelo verde.

Podemos ver que inicialmente, a medida que aumentamos la flexibilidad del modelo, el MSE de prueba disminuye, pero después de cierto punto, el MSE de prueba comienza a aumentar nuevamente.

Esto ilustra el *dilema sesgo-varianza* (bias-variance tradeoff), donde un modelo demasiado simple (alto sesgo) no captura la complejidad de los datos, mientras que un modelo demasiado complejo (alta varianza) se ajusta demasiado a los datos de entrenamiento y no generaliza bien a nuevos datos.

El sesgo es el error introducido por aproximar un problema real, que puede ser complejo, por un modelo más simple. La varianza corresponde a la sensibilidad del modelo a pequeñas fluctuaciones en el conjunto de entrenamiento. Más detalles sobre el dilema sesgo-varianza se puede encontrar en la #link("https://www.statlearning.com", "sección 2.2.2  de ISLP").

== Definiendo un problema de aprendizaje supervisado

Suponiendo que tenemos un conjunto de datos sobre casas en Maracaibo, su área en pies cuadrados y su precio en dólares.

#align(
  center,
  table(
    columns: (5cm, 5cm),
    align: (center, center),
    [*Área (pies²)*], [*Precio (USD)*],
    [630], [173875],
    [966], [229699],
    [974], [234262],
    [1269], [335333],
    [1360], [262383],
    [1684], [440726],
    [2185], [504177],
    [2891], [635097],
    [...], [...],
  ),
)

Podemos graficar estos datos:

#figure(
  caption: [Área y precio de un conjunto de casas en Maracaibo.],
  align(center)[
    #custom-plot(
      x-min: 0,
      x-max: 5000,
      y-min: 0,
      y-max: 1300000,
      x-label: [Área (pies²)],
      y-label: [Precio (USD)],
      x-tick-step: 1000,
      y-tick-step: 200000,
      size: (10, 6),
      {
        let house-data = (
          (630, 173875),
          (966, 229699),
          (974, 234262),
          (1269, 335333),
          (1360, 262383),
          (1684, 440726),
          (2185, 504177),
          (2891, 635097),
          (2933, 701134),
          (3404, 735851),
          (3419, 703413),
          (3592, 777977),
          (3671, 814428),
          (3885, 838719),
          (3944, 879797),
          (4272, 985261),
          (4617, 1019193),
          (4926, 1041741),
          (740, 248292),
          (999, 265480),
          (1195, 317367),
          (1397, 350166),
          (1575, 394890),
          (1788, 402213),
          (1994, 419835),
          (2271, 548233),
          (2463, 444772),
          (2685, 648542),
          (2843, 670717),
          (3132, 656624),
          (3334, 754106),
          (3498, 715799),
          (3709, 742468),
          (3922, 946101),
          (4145, 927821),
          (4376, 1148715),
          (4581, 1263131),
          (4783, 1115818),
        )

        plot.add(
          house-data,
          mark: "x",
          mark-size: 0.18,
          style: (stroke: none),
          mark-style: (stroke: blue.darken(20%), fill: blue.darken(20%)),
        )
      },
    )
  ],
)

Notación: $x^(\(i\))$ denota un vector de características (o features), y $y^(\(i\))$ denota la etiqueta (o label). El par $(x^(\(i\)), y^(\(i\)))$ representa la observación $i$-ésima del conjunto de entrenamiento. Nótese que el superíndice $(i)$ no tiene relación con la potencia, y es simplemente otra forma de indexar.

Para representar el espacio de valores de entrada se usa la letra $cal(X)$, mientras que para representar el espacio de valores de salida se usa la letra $cal(Y)$. En este caso, $cal(X) = cal(Y) = RR$.

Nuestro objetivo es encontrar una función $h: cal(X) arrow.r cal(Y)$ tal que $h(x^(\(i\))) approx y^(\(i\))$ para cada observación $(x^(\(i\)), y^(\(i\)))$ del conjunto de entrenamiento. A esta función se le llama *hipótesis*.

#figure(
  align(center)[
    #set text(font: "Liberation Serif")

    #let caja(w, h, contenido) = rect(
      stroke: 1pt + black,
      radius: 4pt,
      width: w,
      height: h,
      align(center + horizon, contenido),
    )

    #let flecha-abajo = text(1.5em)[$arrow.b$]
    #let flecha-derecha = text(1.5em)[$arrow.r$]

    #stack(
      spacing: 5pt,
      caja(130pt, 55pt)[*Conjunto de*\ *entrenamiento*],
      flecha-abajo,
      caja(130pt, 55pt)[*Algoritmo de*\ *aprendizaje*],
      flecha-abajo,
      grid(
        columns: (auto, auto, auto, auto, auto),
        gutter: 8pt,
        align: horizon,
        align(right)[
          #text(1.2em)[$x$] \
          #text(0.72em)[(área habitable\ de la casa)]
        ],
        flecha-derecha,
        caja(32pt, 26pt)[$h$],
        flecha-derecha,
        align(left)[
          #text(1.2em)[$y$ *predicha*] \
          #text(0.72em)[(precio predicho\ de la casa)]
        ],
      ),
    )
  ],
  caption: [Esquema del proceso de aprendizaje supervisado. Extraído de las notas de #link("https://cs229.stanford.edu/main_notes.pdf")[Stanford CS229].],
) <fig-aprendizaje-supervisado>

El proceso de aprendizaje, como se observó, implica un objetivo de minimización de una función de pérdida que cuantifica el error entre las predicciones del modelo y los valores reales.

= Parte 2: Regresión Lineal

La regresión lineal es uno de los modelos más sencillos de aprendizaje
supervisado. Su objetivo es predecir una variable cuantitativa $y$ a partir
de una o más variables de entrada $fvec(x)$.

Continuemos con el ejemplo de las casas. Queremos utilizar información de
una casa, como su área, para predecir su precio.

== Notación

Supongamos que tenemos $m$ ejemplos de entrenamiento:

$
  S = { (fvec(x)^(\(i\)), y^(\(i\))) }_(i=1)^m
$

donde:

- $m$ es el número de ejemplos de entrenamiento.
- $fvec(x)^(\(i\))$ representa las variables de entrada del ejemplo $i$.
- $y^(\(i\))$ representa el valor que queremos predecir para el ejemplo $i$.
- $theta$ representa los parámetros del modelo.
- $n$ es el número de parámetros del modelo.

Por ejemplo, en nuestro problema de casas, si solamente utilizamos el área
de la casa como entrada:

$
  x_1^(\(i\)) = "área de la casa i"
$

y

$
  y^(\(i\)) = "precio de la casa i".
$

== El modelo lineal

Ahora debemos decidir qué tipo de función utilizaremos para realizar las
predicciones.

La regresión lineal supone que podemos aproximar $y$ mediante una función
lineal de las variables de entrada.

En el caso más sencillo, utilizando solamente el área de una casa:

$
  h_(theta)(fvec(x)) = theta_0 + theta_1 x_1
$


== Más de una variable de entrada

Podemos utilizar más información para realizar la predicción.

Por ejemplo, podríamos utilizar:

$
  x_1 = "área de la casa"
$

$
  x_2 = "número de habitaciones"
$

En ese caso, nuestro modelo sería:

$
  h_(theta)(fvec(x))
  = theta_0 + theta_1 x_1 + theta_2 x_2
$

En general, un modelo lineal puede escribirse como:

$
  h_(theta)(fvec(x))
  = theta_0
  + theta_1 x_1
  + theta_2 x_2
  + dots
  + theta_(n - 1) x_(n - 1)
$

Aquí tenemos $n$ parámetros:

$
  theta_0, theta_1, ..., theta_(n - 1).
$

== Una notación más conveniente

Para simplificar las ecuaciones, introducimos una variable adicional:

$
  x_0 = 1
$

De esta manera podemos escribir:

$
  h_(theta)(fvec(x))
  = sum_(j=0)^(n - 1) theta_j x_j
$

Y si representamos los parámetros y las variables de entrada como vectores:

$
  theta =
  mat(
    theta_0;
    theta_1;
    dots.v;
    theta_(n - 1)
  )
$

y

$
  fvec(x) =
  mat(
    1;
    x_1;
    dots.v;
    x_(n - 1)
  ),
$

entonces podemos escribir todo el modelo simplemente como:

$
  h_(theta)(fvec(x)) = theta^T fvec(x).
$

== Función de costo

Para encontrar los parámetros del modelo correctos, necesitamos definir una función que nos diga cuánto error comete el modelo en general. Una forma sencilla de medir el error consiste en elevar al cuadrado la
diferencia entre cada predicción y su valor real:

$
  (h_(theta)(fvec(x)^(\(i\))) - y^(\(i\)))^2.
$

Luego sumamos los errores de todos los ejemplos:

$
  J(theta)
  = 1/2
  sum_(i=1)^m
  (h_(theta)(fvec(x)^(\(i\))) - y^(\(i\)))^2.
$

A esta función $J(theta)$ se le llama *función de costo*.

¿Por qué elevar al cuadrado la diferencia? Nos da dos beneficios: el primero es que errores positivos y negativos no se cancelan entre sí, y el segundo es que penaliza más los errores grandes que los pequeños.

Por otro lado, el factor $1/2$ no cambia qué valores de $theta$ minimizan la función.
Simplemente hará más convenientes algunas derivadas que veremos más
adelante. Nuestro problema de aprendizaje puede entonces escribirse como:

$
  min_theta J(theta).
$

¿Cómo encontramos los valores de $theta$ que minimizan $J(theta)$?

== Algoritmo LMS

Queremos escoger los parámetros $theta$ de manera que minimicen la función de costo $J(theta)$. Una forma de hacer esto es comenzar con algún valor inicial de $theta$ e ir modificándolo poco a poco, buscando que $J(theta)$ sea cada vez menor. Para esto utilizaremos el algoritmo de *descenso por gradiente*.

El descenso por gradiente actualiza cada parámetro $theta_j$ utilizando:

$
  theta_j := theta_j - alpha frac(partial J(theta), partial theta_j)
$

para cada $j = 0, ..., n - 1$.

Aquí, $alpha$ se conoce como la *tasa de aprendizaje*. La idea es que en cada paso modificamos los parámetros en la dirección que hace que $J(theta)$ disminuya más rápidamente (es decir, en la dirección del gradiente negativo).

El símbolo $:=$ indica una actualización. Es decir, el nuevo valor de $theta_j$ reemplaza al valor anterior.

=== Derivada para un ejemplo

Para poder aplicar el algoritmo necesitamos calcular:

$
  frac(partial J(theta), partial theta_j)
$
  
Comencemos con el caso más sencillo: supongamos que tenemos un único ejemplo de entrenamiento $(fvec(x), y)$.

En ese caso, la función de costo es:

$
  J(theta) = 1/2 (h_theta(fvec(x)) - y)^2.
$

Queremos calcular su derivada con respecto a un parámetro $theta_j$:

$
  (partial J(theta)) / (partial theta_j)
  =
  partial / (partial theta_j) (1/2 (h_theta(fvec(x)) - y)^2)
$

Aplicando la regla de la cadena:

$
  (partial J(theta)) / (partial theta_j)
  =
  (h_theta(fvec(x)) - y) partial / (partial theta_j) (h_theta(fvec(x)) - y)
$

Como:

$
  h_theta(fvec(x))
  =
  sum_(k=0)^(n - 1) theta_k x_k,
$

tenemos:

$
  (partial h_theta(fvec(x)) / (partial theta_j)) = x_j.
$

Por lo tanto:

$
  (partial J(theta)) / (partial theta_j)
  =
  (h_theta(fvec(x)) - y) x_j.
$

Si sustituimos este resultado en la regla de descenso por gradiente:

$
  theta_j
  :=
  theta_j
  - alpha (h_theta(fvec(x)) - y) x_j.
$

Esta expresión se conoce como la *regla de actualización LMS*. También puede escribirse como:

$
  theta_j
  :=
  theta_j
  + alpha (y - h_theta(fvec(x))) x_j.
$

Interpretación: si la predicción $h_theta(fvec(x))$ está muy cerca del valor real $y$, el error será pequeño y los parámetros cambiarán poco. Por el contrario, si la predicción está muy lejos del valor real, el error será mayor y el cambio realizado sobre los parámetros también será mayor. Cada parámetro se modifica de acuerdo con:

$
  "cambio"
  =
  "tasa de aprendizaje"
  times
  "error"
  times
  "entrada".
$

=== Más de un ejemplo de entrenamiento

Hasta ahora derivamos la regla LMS suponiendo que teníamos un solo ejemplo. Sin embargo, normalmente tenemos un conjunto con $m$ ejemplos.

Existen diferentes formas de utilizar estos ejemplos para actualizar los parámetros. Una primera opción consiste en utilizar todos los ejemplos de entrenamiento antes de realizar una actualización.

Recordemos que nuestra función de costo es:

$
  J(theta)
  =
  1/2
  sum_(i=1)^m
  (
    h_theta(fvec(x)^(\(i\)))
    - y^(\(i\))
  )^2.
$

La derivada con respecto a $theta_j$ es:

$
  (partial J(theta)) / (partial theta_j)
  =
  sum_(i=1)^m
  (
    h_theta(fvec(x)^(\(i\)))
    - y^(\(i\))
  )
  x_j^(\(i\)).
$

Por lo tanto, podemos utilizar el siguiente algoritmo de descenso por gradiente para actualizar los parámetros:

#align(center,
block(
  inset: 10pt,
  stroke: 0.5pt,
)[
  *Repetir hasta convergencia:*

  $
    theta_j
    :=
    theta_j
    -
    alpha
    sum_(i=1)^m
    (
      h_theta(fvec(x)^(\(i\)))
      - y^(\(i\))
    )
    x_j^(\(i\))
  $

  para cada $j = 0, ..., n - 1$.
]
)

Agrupando cada actualización de los parámetros en un solo vector, podemos escribir:

$
  theta
  :=
  theta
  -
  alpha
  sum_(i=1)^m
  (
    h_theta(fvec(x)^(\(i\)))
    - y^(\(i\))
  )
  fvec(x)^(\(i\)).
$

Este algoritmo se conoce como batch gradient descent. Nótese que el algoritmo de descenso de gradiente es susceptible a quedar atrapado en mínimos locales. Sin embargo, para el caso de regresión lineal, la función de costo es convexa, por lo que no hay mínimos locales y el algoritmo siempre encontrará el mínimo global (siempre que $alpha$ no sea demasiado grande). La @fig-contour-descent muestra las actualizaciones de los parámetros del algoritmo de descenso por gradiente sobre las curvas de nivel de la función de costo.

#figure(
  custom-plot(
    x-min: 0,
    x-max: 50,
    y-min: 0,
    y-max: 50,
    size: (8, 7),
    {
      // Centro de las curvas de nivel (mínimo local/global)
      let cx = 24.7
      let cy = 25.0
      
      // Ángulo de inclinación de las elipses (~28 grados)
      let angle = 28deg
      let cos-a = calc.cos(angle)
      let sin-a = calc.sin(angle)

      // Función generadora de elipses rotadas
      let ellipse-fn(a, b) = {
        t => {
          let u = a * calc.cos(t)
          let v = b * calc.sin(t)
          (
            cx + u * cos-a - v * sin-a,
            cy + u * sin-a + v * cos-a,
          )
        }
      }

      // 1. Curvas de nivel (de adentro hacia afuera con la paleta de la imagen)
      let contours = (
        (a: 5.5,  b: 3.5,  color: rgb("#1a237e")), // Azul marino
        (a: 11.5, b: 7.2,  color: rgb("#2979ff")), // Azul brillante
        (a: 17.5, b: 10.8, color: rgb("#00e5ff")), // Cian
        (a: 24.0, b: 14.8, color: rgb("#76ff03")), // Verde claro
        (a: 31.0, b: 19.5, color: rgb("#ffd600")), // Amarillo
        (a: 38.5, b: 24.2, color: rgb("#ff6d00")), // Naranja
        (a: 47.0, b: 29.5, color: rgb("#b71c1c")), // Rojo ladrillo
        (a: 56.0, b: 35.0, color: rgb("#792b2b")), // Marrón oscuro exterior
      )

      for c in contours {
        plot.add(
          domain: (0, 2 * calc.pi),
          samples: 120,
          ellipse-fn(c.a, c.b),
          style: (stroke: 0.9pt + c.color),
        )
      }

      // 2. Trayectoria de iteraciones del algoritmo de optimización
      let path-points = (
        (48.0, 30.0),
        (39.0, 30.7),
        (33.5, 30.0),
        (30.3, 28.9),
        (28.5, 27.8),
        (27.2, 27.1),
        (26.3, 26.4),
        (25.7, 25.8),
        (25.2, 25.4),
        (24.9, 25.2),
        (24.7, 25.0),
      )

      // Línea continua azul con marcas en "x"
      plot.add(
        path-points,
        mark: "x",
        mark-size: 0.18,
        style: (
          stroke: 1.2pt + rgb("#155dfc"),
          fill: rgb("#155dfc"),
        ),
      )
    },
  ),
  caption: [Curvas de nivel de la función de costo y pasos del algoritmo de descenso por gradiente hacia el mínimo.],
) <fig-contour-descent>

Como alternativa al batch gradient descent, consideremos este otro algoritmo:

#align(center,
block(
  inset: 10pt,
  stroke: 0.5pt,
)[
  *Repetir:*

  Por cada ejemplo de entrenamiento $(fvec(x)^(\(i\)), y^(\(i\)))$ $1,...,m$:

  $
    theta_j
    :=
    theta_j
    -
    alpha
    (
      h_theta(fvec(x)^(\(i\)))
      - y^(\(i\))
    )
    x_j^(\(i\))
  $

  para cada $j = 0, ..., n - 1$.
]
)

Es decir, en lugar de esperar a que todos los ejemplos de entrenamiento sean evaluados antes de actualizar los parámetros, actualizamos los parámetros después de evaluar cada ejemplo. Este algoritmo se conoce como *stochastic gradient descent* (SGD). Este método es más rápido que el batch gradient descent, pero el camino hacia el mínimo es más ruidoso, ya que cada actualización depende de un solo ejemplo de entrenamiento. 

Una tercera alternativa consiste en actualizar los parámetros después de evaluar un pequeño subconjunto de ejemplos de entrenamiento (por ejemplo, 32 o 64). Este algoritmo se conoce como *mini-batch gradient descent* (MBGD). Esta suele ser la opción más utilizada en la práctica, ya que combina las ventajas de BGD y SGD: no es demasiado costoso computacionalmente y su camino hacia el mínimo es más estable que el de SGD.

Para los casos de SGD y MBGD, el algoritmo nunca convergerá, pues nuevos ejemplos modificarán los parámetros ligeramente en una dirección específica. En este caso, se escoge un número de repeticiones específica, o se repite la actualización hasta que el costo no disminuya significativamente. Otra técnica es disminuir $alpha$ poco a poco hasta llegar a 0.

== Ecuaciones normales

En lugar de realizar actualizaciones usando descenso de gradiente, podemos obtener (en el caso de regresión lineal), el valor de los parámetros de forma explícita. Para derivar estas ecuaciones de forma simple, debemos introducir notación sobre como tomar derivadas de matrices y vectores.

=== Cálculo Matricial

Para una función $f: RR^(n times d) arrow.r RR$, es decir, que mapea una matriz de dimensión $n times d$ a los números reales, se define la derivada (o gradiente) de $f$ respecto a una matriz $fvec(A)$ siguiendo el *denominator layout* (la derivada preserva las dimensiones de la variable):

$
  gradient_fvec(A) f(fvec(A)) = mat(
    (partial f) / (partial fvec(A)_(11)), dots, (partial f) / (partial fvec(A)_(1d));
    dots.v, dots.down, dots.v;
    (partial f) / (partial fvec(A)_(n 1)), dots, (partial f) / (partial fvec(A)_(n d))
  )
$

Es decir, el gradiente $gradient_fvec(A) f(fvec(A))$ es una matriz de tamaño $n times d$, donde la entrada $(i, j)$ es $(partial f) / (partial fvec(A)_(i j))$. 

Por ejemplo, si $fvec(A) = mat(A_(11), A_(12); A_(21), A_(22))$ y $f(fvec(A)) = 3/2 fvec(A)_(11) + 5 fvec(A)_(12)^2 + fvec(A)_(21) fvec(A)_(22)$, el gradiente respecto a $fvec(A)$ es:

$
  gradient_fvec(A) f(fvec(A)) = mat(
    3/2, 10 A_(12);
    A_(22), A_(21)
  )
$

==== Gradiente de un Escalar respecto a un Vector

Como caso particular, para un vector columna $fvec(x) in RR^(n times 1)$ y una función escalar $f(fvec(x)) in RR$, el gradiente adopta la forma del denominador (vector columna de $n times 1$):

$
  (partial f) / (partial fvec(x)) = gradient_fvec(x) f(fvec(x)) = mat(
    (partial f) / (partial x_1);
    dots.v;
    (partial f) / (partial x_n)
  ) in RR^(n times 1)
$

==== Derivada de Vector respecto a Vector (*Denominator Layout*)

Si tenemos una función vectorial $fvec(y): RR^n arrow.r RR^m$, dada por $fvec(y)(fvec(x)) = [y_1(fvec(x)), dots, y_m(fvec(x))]^T$, la derivada de $fvec(y)$ respecto a $fvec(x)$ en esta convención es una matriz de tamaño $n times m$ (la transpuesta de la Jacobiana usual):

$
  (partial fvec(y)) / (partial fvec(x)) = mat(
    (partial y_1) / (partial x_1), (partial y_2) / (partial x_1), dots, (partial y_m) / (partial x_1);
    (partial y_1) / (partial x_2), (partial y_2) / (partial x_2), dots, (partial y_m) / (partial x_2);
    dots.v, dots.v, dots.down, dots.v;
    (partial y_1) / (partial x_n), (partial y_2) / (partial x_n), dots, (partial y_m) / (partial x_n)
  ) in RR^(n times m)
$

Donde la fila $j$ contiene las derivadas respecto a $x_j$ y la columna $i$ corresponde a las variaciones de la componente $y_i$. Es decir:
$
  [(partial fvec(y)) / (partial fvec(x))]_(j, i) = (partial y_i) / (partial x_j)
$

*Nota útil:* Si $fvec(y) = fvec(x)$, entonces $(partial fvec(x)) / (partial fvec(x)) = fvec(I)_n$ (la matriz identidad $n times n$).

==== Regla de la Cadena para Funciones Escalares

Consideremos una función escalar compuesta $g(fvec(x)) = f(fvec(y)(fvec(x)))$, donde $fvec(y) in RR^m$ y $f: RR^m arrow.r RR$. Aplicando la regla de la cadena en *denominator layout*:

$
  (partial g) / (partial fvec(x)) = (partial fvec(y)) / (partial fvec(x)) (partial f) / (partial fvec(y))
$

*Verificación dimensional:*
- $(partial fvec(y)) / (partial fvec(x)) in RR^(n times m)$
- $(partial f) / (partial fvec(y)) in RR^(m times 1)$
- El producto resultante es de tamaño $(n times m) times (m times 1) = n times 1$, preservando la forma de $fvec(x)$.

==== Propiedades y Reglas Operativas Fundamentales

Sean $fvec(a) in RR^m$ un vector constante, y $fvec(u)(fvec(x)), fvec(v)(fvec(x)) in RR^m$ funciones vectoriales diferenciables:

+ *Forma lineal con vector constante ($fvec(a)^T fvec(y)$):*
  Como $(partial (fvec(a)^T fvec(y))) / (partial fvec(y)) = fvec(a)$, por la regla de la cadena:
  $
    (partial (fvec(a)^T fvec(y))) / (partial fvec(x)) = (partial fvec(y)) / (partial fvec(x)) fvec(a)
  $

+ *Regla del producto para producto punto ($fvec(u)^T fvec(v)$):*
  Cuando ambos vectores dependen de $fvec(x)$:
  $
    (partial (fvec(u)^T fvec(v))) / (partial fvec(x)) = (partial fvec(u)) / (partial fvec(x)) fvec(v) + (partial fvec(v)) / (partial fvec(x)) fvec(u)
  $
  En el caso particular de $fvec(x)^T fvec(y)$ (donde $fvec(u) = fvec(x)$ y por tanto $(partial fvec(x)) / (partial fvec(x)) = fvec(I)$):
  $
    (partial (fvec(x)^T fvec(y))) / (partial fvec(x)) = fvec(I) fvec(y) + (partial fvec(y)) / (partial fvec(x)) fvec(x) = fvec(y) + (partial fvec(y)) / (partial fvec(x)) fvec(x)
  $

+ *Norma al cuadrado / Producto consigo mismo ($fvec(y)^T fvec(y)$):*
  Dado que $(partial (fvec(y)^T fvec(y))) / (partial fvec(y)) = 2 fvec(y)$, aplicando la regla de la cadena:
  $
    (partial (fvec(y)^T fvec(y))) / (partial fvec(x)) = 2 (partial fvec(y)) / (partial fvec(x)) fvec(y)
  $

=== Solución analítica de mínimos cuadrados

Para utilizar las herramientas del cálculo matricial para solucionar el problema de mínimos cuadrados, debemos definir el problema en términos de vectores y matrices. En primer lugar definimos la matriz de diseño $fvec(X)$ que contiene los ejemplos de entrenamiento como filas.

$
  fvec(X) = mat(- (fvec(x)^(\(1\)))^T -; - (fvec(x)^(\(2\)))^T -; dots.v; - (fvec(x)^(\(m\)))^T - ) 
$

Definamos $fvec(y)$ como un vector columna que contiene todas las etiquetas:

$
  fvec(y) = mat(y^(\(1\)); y^(\(2\)); dots.v; y^(\(m\)))
$

Sabemos que $h_theta (fvec(x)^(\(i\))) = (fvec(x)^(\(i\)))^T theta$. Por lo tanto, podemos escribir el problema como:

$
  fvec(X) theta - fvec(y)
$

Usando el hecho que para un vector $fvec(u)$, $fvec(u)^T fvec(u) = sum_i fvec(u)_i^2$, la función de costo:

$
  1/2
  (fvec(X) theta - fvec(y))^T
  (fvec(X) theta - fvec(y))
  =
  1/2
  sum_(i=1)^m
  (h_theta (fvec(x)^(\(i\))) - y^(\(i\)))^2
  =
  J(theta)
$

Para minimizar esta función de costo, debemos derivar con respecto a $theta$ y establecer la derivada igual a cero:

$
  nabla_theta J(theta)
  &= nabla_theta 1/2 (fvec(X) theta - fvec(y))^T
    (fvec(X) theta - fvec(y)) \
  &= 1/2 nabla_theta (
    (fvec(X) theta)^T fvec(X) theta
    - (fvec(X) theta)^T fvec(y)
    - fvec(y)^T fvec(X) theta
    + fvec(y)^T fvec(y)
  ) \
  &= 1/2 nabla_theta (
    theta^T (fvec(X)^T fvec(X)) theta
    - 2 (fvec(X)^T fvec(y))^T theta
  ) \
  &= 1/2 (
    2 fvec(X)^T fvec(X) theta
    - 2 fvec(X)^T fvec(y)
  ) \
  &= fvec(X)^T fvec(X) theta - fvec(X)^T fvec(y)
$

En el paso 3, utilizamos el hecho que $fvec(a)^T fvec(b) = fvec(b)^T fvec(a)$. En el paso 4, utilizamos las siguientes reglas de derivación:

- $gradient_fvec(x) fvec(b)^T fvec(x) = fvec(b)$
- $gradient_fvec(x) fvec(x)^T fvec(A) fvec(x) = 2 fvec(A) fvec(x)$ para una matriz simétrica $fvec(A)$.

Si hacemos la derivada igual a cero, obtenemos el punto crítico:

$ fvec(X)^T fvec(X) theta = fvec(X)^T fvec(y) $

Y si $fvec(X)^T fvec(X)$ es invertible, podemos despejar $theta$:

$
  theta = (fvec(X)^T fvec(X))^(-1) fvec(X)^T fvec(y)
$
