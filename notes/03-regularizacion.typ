// OUTLINE DE ESTA CLASE:
// 1. Selección de modelos
// 1.1. Regresión polinomial: más features, más complejidad
// 1.2. Por qué no escoger con el error de entrenamiento
// 1.3. Validación hold-out y conjunto de prueba
// 1.4. Validación cruzada k-fold y leave-one-out
// 1.5. Baseline y data leakage
// 2. Regularización
// 2.1. Idea general: J + lambda R
// 2.2. Ridge: centrar y escalar, solución cerrada, invertibilidad, weight decay
// 2.3. Sparsity y Lasso: geometría y caminos de coeficientes
// 2.4. Elección de lambda, comparación
// 3. Interpretación probabilística
// 3.1. Mínimos cuadrados como máxima verosimilitud
// 3.2. Visión bayesiana: MAP, Ridge y Lasso (opcional)
// 4. Laboratorio
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
  title: [Generalización, selección de modelos y regularización],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

// Datos de las figuras
#let d-poli = json("figs/polinomios.json")
#let d-ridge = json("figs/ridge_polinomio.json")
#let d-caminos = json("figs/caminos.json")
#let d-geo = json("figs/geometria.json")
#let d-gauss = json("figs/gaussiana.json")

// Formato de ticks para ejes logarítmicos: 10^k
#let fmt-pot(v) = {
  let k = int(calc.round(calc.log(v)))
  $10^#k$
}

// Redondeo para mostrar números en el texto
#let num(x, digits: 2) = str(calc.round(x, digits: digits))

En la clase anterior vimos que lo que nos interesa de un modelo es su error sobre datos que no ha visto (error de prueba), y no su error sobre los datos con los que fue entrenado. También vimos con la Figura 1 de la clase anterior que, a medida que un modelo se vuelve más flexible, el error de entrenamiento siempre baja, pero el error de prueba primero baja y luego sube.

En esta clase vamos a responder tres preguntas que salen de esa figura:

+ Si no podemos usar el error de entrenamiento para escoger la flexibilidad de un modelo, ¿qué usamos? (Parte 1)
+ ¿Hay alguna forma de controlar la flexibilidad de un modelo sin cambiar el número de parámetros? (Parte 2)
+ ¿Por qué usamos el error cuadrático y no otra función de costo, y qué relación tiene eso con la regularización? (Parte 3)

Seguimos con la notación de la clase anterior: tenemos $m$ ejemplos de entrenamiento $(fvec(x)^(\(i\)), y^(\(i\)))$, el modelo tiene $n$ parámetros $theta_0, ..., theta_(n-1)$, y la función de costo es

$
  J(theta) = 1/2 sum_(i=1)^m (h_theta (fvec(x)^(\(i\))) - y^(\(i\)))^2.
$

Esta clase sigue de cerca las secciones 1.3 y 9.1 a 9.4 de las #link("https://cs229.stanford.edu/main_notes.pdf")[notas de Stanford CS229], y los capítulos 5 y 6 de #link("https://www.statlearning.com")[ISLP].

= Parte 1: Selección de modelos

== Más features, más complejidad: regresión polinomial

Antes de hablar de cómo escoger un modelo, necesitamos una forma concreta de hacer que un modelo lineal sea más o menos flexible. La más sencilla es agregar features.

Supongamos que tenemos una sola variable de entrada $x$. En lugar de usar el modelo $h_theta (x) = theta_0 + theta_1 x$, podemos usar un polinomio de grado $d$:

$
  h_theta (x) = theta_0 + theta_1 x + theta_2 x^2 + dots + theta_d x^d.
$

A primera vista esto no parece regresión lineal, porque $h_theta$ no es una recta. Pero si definimos nuevas features

$
  x_1 = x, quad x_2 = x^2, quad dots, quad x_d = x^d,
$

el modelo queda $h_theta (fvec(x)) = theta^T fvec(x)$, exactamente igual que en la clase anterior, con $n = d + 1$ parámetros. La palabra "lineal" en regresión lineal se refiere a que el modelo es lineal en los parámetros $theta$, no en la variable $x$. Por eso todo lo que vimos (descenso por gradiente, ecuaciones normales) funciona sin cambios: solo cambia la matriz de diseño, que ahora tiene una columna por cada potencia:

$
  fvec(X) = mat(
    1, x^(\(1\)), (x^(\(1\)))^2, dots, (x^(\(1\)))^d;
    1, x^(\(2\)), (x^(\(2\)))^2, dots, (x^(\(2\)))^d;
    dots.v, dots.v, dots.v, dots.down, dots.v;
    1, x^(\(m\)), (x^(\(m\)))^2, dots, (x^(\(m\)))^d;
  )
$

Un detalle práctico: si $x$ está entre 0 y 5, entonces $x^10$ puede llegar a casi diez millones, mientras que la primera columna vale 1. Con columnas de tamaños tan distintos, la matriz $fvec(X)^T fvec(X)$ es muy difícil de invertir numéricamente, y el descenso por gradiente converge muy lento. Por eso, antes de construir las potencias, es usual estandarizar $x$ (restarle su media y dividir entre su desviación estándar). Volveremos a esto cuando veamos regularización.

La @fig-polinomios muestra qué pasa al ajustar polinomios de distintos grados a 15 ejemplos generados a partir de una función conocida más ruido.

#figure(
  caption: [Regresión polinomial sobre 15 ejemplos de entrenamiento. A la izquierda, ajustes de grado 1 (azul), 4 (verde) y 10 (rojo); la curva gris es la función real. A la derecha, MSE de entrenamiento (gris punteado) y de validación (rojo) en función del grado, en escala logarítmica. La línea horizontal es la varianza del ruido, $sigma^2$: ningún modelo puede tener un error esperado de validación menor.],
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
          let estilos = (
            "1": (paint: blue, thickness: 1.5pt, dash: "dashed"),
            "4": (paint: green.darken(20%), thickness: 1.8pt),
            "10": (paint: red, thickness: 1.3pt),
          )

          // Función real
          plot.add(
            d-poli.real,
            style: (stroke: (paint: gray.lighten(20%), thickness: 3pt)),
          )

          // Ajustes polinomiales
          for (grado, estilo) in estilos {
            plot.add(
              d-poli.ajustes.at(grado).curva,
              style: (stroke: estilo),
            )
          }

          // Datos de entrenamiento
          plot.add(
            d-poli.train,
            mark: "o",
            mark-size: 0.16,
            mark-style: (stroke: none, fill: black),
            style: (stroke: none),
          )
        },
      )
    ],

    [
      #custom-plot(
        x-min: 1,
        x-max: 10,
        y-min: 0.001,
        y-max: 1000,
        x-label: [Grado del polinomio],
        y-label: [MSE],
        x-tick-step: 1,
        y-tick-step: 1,
        y-mode: "log",
        y-format: fmt-pot,
        size: (6, 5),
        {
          let train = d-poli.curva_error.map(r => (r.grado, r.mse_train))
          let val = d-poli.curva_error.map(r => (r.grado, r.mse_val))

          // Varianza del ruido (error irreducible)
          plot.add-hline(
            d-poli.ruido_var,
            style: (stroke: (paint: black, thickness: 0.6pt, dash: "dotted")),
          )

          plot.add(
            train,
            mark: "o",
            mark-size: 0.1,
            mark-style: (stroke: none, fill: gray.darken(40%)),
            style: (stroke: (paint: gray.darken(40%), thickness: 1.5pt, dash: "dashed")),
          )

          plot.add(
            val,
            mark: "o",
            mark-size: 0.1,
            mark-style: (stroke: none, fill: red.darken(20%)),
            style: (stroke: (paint: red.darken(20%), thickness: 2pt)),
          )
        },
      )
    ],
  ),
) <fig-polinomios>

La figura es la versión "real" de la Figura 1 de la clase anterior, ahora con ajustes calculados de verdad. Se observan tres cosas:

- El error de entrenamiento baja siempre que aumentamos el grado. Esto no es casualidad: un polinomio de grado $d+1$ puede imitar a cualquier polinomio de grado $d$ (basta poner $theta_(d+1) = 0$), así que su mejor ajuste nunca puede ser peor.
- El error de validación baja hasta grado 3 o 4, y luego sube. Con grado 10 el error de validación es más de mil veces mayor que con grado 4.
- El polinomio de grado 10 pasa muy cerca de los puntos de entrenamiento, pero entre ellos hace oscilaciones enormes. Sus coeficientes también son grandes: el mayor tiene magnitud #num(d-poli.ajustes.at("10").max_abs_theta, digits: 1), mientras que el mayor del polinomio de grado 4 tiene magnitud #num(d-poli.ajustes.at("4").max_abs_theta, digits: 1). Esta observación será el punto de partida de la Parte 2.

El grado $d$ es un ejemplo de hiperparámetro: un valor que define al modelo pero que no se aprende con el algoritmo de entrenamiento, sino que lo escogemos nosotros. La pregunta de esta parte es cómo escogerlo.

== Por qué no podemos escoger con el error de entrenamiento

Supongamos que tenemos varios modelos candidatos $M_1, ..., M_k$, por ejemplo, polinomios de grado $1, ..., 10$. La primera idea que se nos podría ocurrir es:

+ Entrenar cada modelo $M_j$ sobre el conjunto de entrenamiento $S$.
+ Escoger el modelo con el menor error de entrenamiento.

Este procedimiento no funciona. Como acabamos de ver, el error de entrenamiento baja siempre que el modelo es más flexible, así que este método escogería siempre el polinomio de mayor grado, que es justamente el que peor generaliza.

El problema de fondo es que estamos usando los mismos datos para dos cosas: para ajustar los parámetros y para medir qué tan bueno es el ajuste. Necesitamos medir el error con datos que el modelo no haya usado para entrenarse.

== Validación hold-out

La solución más sencilla es separar una parte de los datos y no usarla para entrenar. Este procedimiento se conoce como validación hold-out:

+ Dividir aleatoriamente $S$ en dos partes: $S_"train"$ (por ejemplo, el 70% de los datos) y $S_"val"$ (el 30% restante). A $S_"val"$ se le llama conjunto de validación.
+ Entrenar cada modelo $M_j$ usando solamente $S_"train"$.
+ Escoger el modelo con el menor error sobre $S_"val"$ (el error de validación).

Como los modelos no vieron los ejemplos de $S_"val"$ durante el entrenamiento, el error de validación es una estimación razonable del error que tendrán con datos nuevos. Esto es exactamente lo que hicimos en la @fig-polinomios: el grado 4 tiene el menor error de validación, así que es el que escogeríamos.

Usualmente se separa entre un cuarto y un tercio de los datos para validación. Cuando hay muchos datos, la fracción puede ser mucho menor, siempre que el número de ejemplos de validación sea suficiente. Por ejemplo, en ImageNet, con más de un millón de imágenes, es común usar unas 50 mil para validación, es decir, alrededor del 5%.

Una vez escogido el modelo, es buena idea volver a entrenarlo usando todos los datos ($S_"train"$ y $S_"val"$ juntos), ya que con más datos el ajuste suele ser mejor.

=== El conjunto de prueba

Hay un detalle sutil. Si probamos muchos modelos y escogemos el que tiene menor error de validación, ese error de validación tiende a ser optimista: escogimos ese modelo precisamente porque le fue bien en esos datos, en parte por suerte. Por eso, si queremos reportar qué tan bueno es el modelo final, necesitamos un tercer conjunto que no se use para nada durante el desarrollo: el conjunto de prueba.

#let caja-datos(color, contenido) = rect(
  width: 100%,
  height: 0.9cm,
  fill: color,
  stroke: 0.8pt + black,
  align(center + horizon, contenido),
)

#let c-train = rgb("#BBD5F2")
#let c-val = rgb("#F6C99A")
#let c-test = rgb("#D0D0D0")

#figure(
  caption: [División de los datos en entrenamiento, validación y prueba. El conjunto de entrenamiento se usa para ajustar los parámetros, el de validación para escoger hiperparámetros (como el grado del polinomio o $lambda$), y el de prueba solamente para medir el desempeño del modelo final.],
  block(width: 100%)[
    #set text(size: 9pt)
    #grid(
      columns: (6fr, 2fr, 2fr),
      gutter: 0pt,
      caja-datos(c-train)[Entrenamiento (60%)],
      caja-datos(c-val)[Validación (20%)],
      caja-datos(c-test)[Prueba (20%)],
    )
  ],
) <fig-split>

La regla es: el conjunto de prueba se usa una sola vez, al final. Si después de ver el error de prueba volvemos a cambiar el modelo, el conjunto de prueba se convirtió en otro conjunto de validación, y su error deja de ser una estimación honesta.

== Validación cruzada

La desventaja de la validación hold-out es que desperdicia datos: los modelos se entrenan con el 70% de los datos, no con el 100%. Si tenemos muchos datos esto no importa, pero si tenemos pocos (por ejemplo, $m = 20$), perder el 30% puede cambiar mucho el resultado. Además, el error de validación depende de cuáles ejemplos quedaron en $S_"val"$ por azar.

La validación cruzada k-fold resuelve esto usando todos los datos para validar, por turnos:

+ Dividir aleatoriamente $S$ en $k$ partes disjuntas $S_1, ..., S_k$, de $m slash k$ ejemplos cada una.
+ Para cada modelo $M_j$:
  - Para cada parte $S_l$, con $l = 1, ..., k$: entrenar $M_j$ con todas las partes excepto $S_l$, y calcular su error sobre $S_l$.
  - Estimar el error de $M_j$ como el promedio de esos $k$ errores.
+ Escoger el modelo con el menor error estimado, y volver a entrenarlo con todo $S$.

#figure(
  caption: [Validación cruzada con $k = 5$. En cada iteración, una parte distinta (naranja) se usa para validar y las demás (azul) para entrenar. El error estimado es el promedio de los 5 errores de validación. El conjunto de prueba (gris) queda fuera de todo el proceso.],
  block(width: 85%)[
    #grid(
      columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 0.4cm, 1.3fr),
      row-gutter: 4pt,
      column-gutter: 0pt,
      align: horizon,
      ..for j in range(5) {
        (
          box(inset: (right: 8pt))[Iteración #(j + 1)],
          ..range(5).map(l => caja-datos(if l == j { c-val } else { c-train })[
            #if l == j [Val.] else [Ent.]
          ]),
          [],
          caja-datos(c-test)[Prueba],
        )
      }
    )
  ],
) <fig-kfold>

Un valor típico es $k = 10$. Ahora en cada iteración solo se deja fuera $1 slash k$ de los datos, mucho menos que en hold-out. El costo es que cada modelo se entrena $k$ veces.

Cuando hay muy pocos datos, se puede usar el caso extremo $k = m$: se entrena con todos los ejemplos menos uno, se mide el error en el ejemplo que quedó fuera, y se repite para cada ejemplo. Esto se conoce como leave-one-out cross validation (LOOCV).

La validación cruzada no solo sirve para escoger entre modelos. También se puede usar simplemente para estimar qué tan bueno es un único modelo cuando no tenemos suficientes datos para separar un conjunto de validación.

== Baseline y data leakage

Antes de entrenar cualquier modelo, conviene tener un punto de comparación sencillo, llamado baseline. Para regresión, el baseline más simple es predecir siempre la media de las etiquetas de entrenamiento, $hat(y) = overline(y)$. Nótese que esto es exactamente regresión lineal con un solo parámetro, $h_theta (fvec(x)) = theta_0$: si minimizamos $J(theta)$ respecto a $theta_0$, obtenemos $theta_0 = overline(y)$. Si un modelo más complicado no supera al baseline en validación, algo anda mal.

El otro error común en este proceso es el data leakage (fuga de datos): que información de los conjuntos de validación o prueba se filtre al entrenamiento. Cuando eso pasa, el error de validación o prueba sale mejor de lo que realmente es. Algunos ejemplos:

- Estandarizar las features usando la media y desviación estándar de todo el conjunto de datos, antes de separar entrenamiento y prueba. La media y la desviación estándar también se "aprenden" de los datos, así que deben calcularse solo con el conjunto de entrenamiento, y luego aplicarse tal cual a validación y prueba.
- Tener ejemplos duplicados que terminan uno en entrenamiento y otro en prueba.
- Usar features que no estarán disponibles al momento de predecir. Por ejemplo, para predecir el precio de una casa, usar la comisión que cobró el agente por venderla.
- Con datos que dependen del tiempo (como precios de acciones), separar los datos aleatoriamente. Si el modelo se entrena con datos de 2026 y se valida con datos de 2024, está usando el futuro para predecir el pasado. En estos casos se separa por fecha.

= Parte 2: Regularización

== Idea general

En la Parte 1, controlamos la complejidad del modelo cambiando el número de parámetros (el grado del polinomio). Pero la complejidad de un modelo no depende solo de cuántos parámetros tiene, sino también de qué tan grandes son. En la @fig-polinomios, el polinomio de grado 10 tiene coeficientes grandes, y eso es lo que le permite oscilar tanto entre los puntos de entrenamiento.

La regularización consiste en agregar a la función de costo un término $R(theta)$, llamado regularizador, que mide la complejidad del modelo:

$
  J_lambda (theta) = J(theta) + lambda R(theta).
$

El parámetro $lambda >= 0$ se llama parámetro de regularización, y controla el balance entre dos objetivos: ajustarse bien a los datos (que $J(theta)$ sea pequeño) y tener un modelo sencillo (que $R(theta)$ sea pequeño).

- Si $lambda = 0$, recuperamos la función de costo original.
- Si $lambda$ es pequeño, el regularizador funciona casi como un criterio de desempate: entre modelos que se ajustan igual de bien a los datos, prefiere el más sencillo.
- Si $lambda$ es muy grande, el ajuste a los datos casi no importa, y el modelo termina siendo demasiado simple (alto sesgo).

Igual que el grado del polinomio, $lambda$ es un hiperparámetro, y se escoge con validación.

== Regularización $L_2$: Ridge

El regularizador más usado es la suma de los cuadrados de los parámetros:

$
  R(theta) = 1/2 sum_(j=1)^(n-1) theta_j^2.
$

A la regresión lineal con este regularizador se le llama Ridge regression. Su función de costo es

$
  J_lambda (theta) = 1/2 sum_(i=1)^m (h_theta (fvec(x)^(\(i\))) - y^(\(i\)))^2 + lambda/2 sum_(j=1)^(n-1) theta_j^2.
$

Nótese que la suma empieza en $j = 1$: el intercepto $theta_0$ no se penaliza. La razón es que $theta_0$ no hace al modelo más flexible, solo lo sube o lo baja. Si lo penalizáramos, estaríamos empujando las predicciones hacia 0, y el resultado cambiaría según las unidades de $y$ (no es lo mismo medir precios en dólares que en miles de dólares). Enseguida veremos que, preparando bien los datos, podemos sacar a $theta_0$ del problema.

=== Centrar y escalar las features

Antes de resolver Ridge, necesitamos preparar los datos. Hay dos razones.

La primera es que la penalización trata a todos los coeficientes por igual: $theta_1^2$ cuesta lo mismo que $theta_2^2$. Pero el tamaño de un coeficiente depende de las unidades de su feature. Si el área de una casa se mide en pies cuadrados en lugar de metros cuadrados, la feature se multiplica por 10.76 y su coeficiente se divide entre 10.76 para dar la misma predicción. Con un coeficiente más pequeño, la penalización sobre el área sería mucho menor, y el modelo regularizado cambiaría solo por haber cambiado de unidades.

Por eso, antes de regularizar, se estandariza cada feature:

$
  x_j^(\(i\)) := (x_j^(\(i\)) - mu_j) / s_j, quad j = 1, ..., n-1,
$

donde $mu_j$ y $s_j$ son la media y la desviación estándar de la feature $j$. Después de esto, todas las features están en la misma escala y la penalización las trata de forma justa.

La segunda razón tiene que ver con $theta_0$. Después de estandarizar, cada feature tiene media cero en el conjunto de entrenamiento, es decir, $sum_(i=1)^m x_j^(\(i\)) = 0$ para cada $j >= 1$. Veamos qué pasa con la derivada de $J_lambda$ respecto a $theta_0$ (que no aparece en la penalización):

$
  (partial J_lambda (theta)) / (partial theta_0)
  = sum_(i=1)^m (theta_0 + sum_(j=1)^(n-1) theta_j x_j^(\(i\)) - y^(\(i\)))
  = m theta_0 + sum_(j=1)^(n-1) theta_j underbrace(sum_(i=1)^m x_j^(\(i\)), = 0) - sum_(i=1)^m y^(\(i\)).
$

Igualando a cero obtenemos

$
  theta_0 = 1/m sum_(i=1)^m y^(\(i\)) = overline(y).
$

Es decir, con las features centradas, el intercepto siempre es la media de las etiquetas, sin importar el valor de $lambda$ ni de los demás parámetros. Podemos fijarlo desde el principio y olvidarnos de él: basta con restarle $overline(y)$ a todas las etiquetas y quitar la columna de unos de la matriz de diseño.

Así que, de aquí en adelante, en esta parte supondremos que:

- Las features están estandarizadas, y la matriz de diseño $fvec(X) in RR^(m times (n-1))$ no tiene la columna de unos.
- Las etiquetas están centradas: $fvec(y)$ contiene los valores $y^(\(i\)) - overline(y)$.
- $theta = (theta_1, ..., theta_(n-1))^T$ contiene solo los parámetros que se penalizan.

Para predecir sobre un ejemplo nuevo $fvec(x)$, lo estandarizamos con las mismas $mu_j$ y $s_j$, y calculamos $h_theta (fvec(x)) = overline(y) + theta^T fvec(x)$.

Es importante que $mu_j$, $s_j$ y $overline(y)$ se calculen solo con el conjunto de entrenamiento, y luego se apliquen tal cual a los datos de validación y prueba. Si los calculáramos con todos los datos, tendríamos data leakage.

El escalamiento también ayuda al descenso por gradiente, aunque no haya regularización. Cuando las features tienen escalas muy distintas, las curvas de nivel de $J(theta)$ son elipses muy alargadas, y el descenso por gradiente avanza en zigzag. Con features en la misma escala, las curvas de nivel son más parecidas a círculos y el algoritmo converge más rápido.

=== Solución cerrada

Con los datos centrados y escalados, la función de costo de Ridge en forma matricial es

$
  J_lambda (theta) = 1/2 (fvec(X) theta - fvec(y))^T (fvec(X) theta - fvec(y)) + lambda/2 theta^T theta.
$

El gradiente del primer término ya lo calculamos en la clase anterior: $fvec(X)^T fvec(X) theta - fvec(X)^T fvec(y)$. Para el segundo término, $gradient_theta theta^T theta = 2 theta$. Entonces:

$
  gradient_theta J_lambda (theta) = fvec(X)^T fvec(X) theta - fvec(X)^T fvec(y) + lambda theta.
$

Igualando a cero:

$
  (fvec(X)^T fvec(X) + lambda fvec(I)) theta = fvec(X)^T fvec(y)
$

y por lo tanto

$
  theta = (fvec(X)^T fvec(X) + lambda fvec(I))^(-1) fvec(X)^T fvec(y).
$

Si $lambda = 0$, recuperamos las ecuaciones normales de la clase anterior. La única diferencia con Ridge es que se le suma $lambda$ a la diagonal de $fvec(X)^T fvec(X)$.

=== El problema de la invertibilidad

En la clase anterior, al despejar $theta = (fvec(X)^T fvec(X))^(-1) fvec(X)^T fvec(y)$, supusimos que $fvec(X)^T fvec(X)$ es invertible. Esto falla en dos situaciones comunes:

- Cuando hay menos ejemplos que parámetros. Por ejemplo, un polinomio de grado 20 con 15 ejemplos.
- Cuando alguna feature se puede escribir como combinación lineal de otras. Por ejemplo, si tenemos el área de la casa en metros cuadrados y también en pies cuadrados, una columna de $fvec(X)$ es la otra multiplicada por 10.76.

En ambos casos hay infinitas soluciones con el mismo error de entrenamiento, y las ecuaciones normales no nos dicen cuál escoger. Con Ridge esto no pasa: para cualquier $lambda > 0$, la matriz $fvec(X)^T fvec(X) + lambda fvec(I)$ siempre es invertible. Veamos por qué.

Sea $fvec(M) = fvec(X)^T fvec(X) + lambda fvec(I)$. Para cualquier vector $fvec(v) != 0$:

$
  fvec(v)^T fvec(M) fvec(v) = fvec(v)^T fvec(X)^T fvec(X) fvec(v) + lambda fvec(v)^T fvec(v) = norm(fvec(X) fvec(v))^2 + lambda norm(fvec(v))^2 > 0,
$

porque el primer término es mayor o igual a cero y el segundo es estrictamente positivo. Esto implica que $fvec(M)$ es invertible: si no lo fuera, existiría un $fvec(v) != 0$ con $fvec(M) fvec(v) = 0$, y entonces $fvec(v)^T fvec(M) fvec(v) = 0$, lo cual es una contradicción.

Esta fue, de hecho, la motivación original de Ridge en estadística: arreglar el problema de invertibilidad. La interpretación como control de la complejidad vino después.

=== Descenso por gradiente: weight decay

También podemos minimizar $J_lambda$ con descenso por gradiente. La derivada respecto a $theta_j$ es la misma de la clase anterior más la derivada del regularizador:

$
  (partial J_lambda (theta)) / (partial theta_j) = sum_(i=1)^m (h_theta (fvec(x)^(\(i\))) - y^(\(i\))) x_j^(\(i\)) + lambda theta_j.
$

Sustituyendo en la regla de actualización:

$
  theta_j := theta_j - alpha lambda theta_j - alpha sum_(i=1)^m (h_theta (fvec(x)^(\(i\))) - y^(\(i\))) x_j^(\(i\))
$

que podemos reescribir como

$
  theta_j := (1 - alpha lambda) theta_j - alpha sum_(i=1)^m (h_theta (fvec(x)^(\(i\))) - y^(\(i\))) x_j^(\(i\)).
$

Es decir, en cada paso primero multiplicamos cada parámetro por un número un poco menor que 1, y luego aplicamos la actualización LMS de siempre. Los parámetros se van "encogiendo" en cada paso, a menos que los datos los empujen a crecer. Por esta razón, en deep learning a la regularización $L_2$ se le llama weight decay (decaimiento de los pesos).

=== Ridge sobre el polinomio de grado 10

Volvamos al ejemplo de la @fig-polinomios. Tomamos el polinomio de grado 10, que era el que peor generalizaba, y lo entrenamos con Ridge para distintos valores de $lambda$.

#figure(
  caption: [Ridge sobre el polinomio de grado 10 de la @fig-polinomios. A la izquierda, ajustes con $lambda = 10^(-6)$ (rojo), $lambda approx #num(d-ridge.mejor_lambda, digits: 3)$ (verde, el de menor error de validación) y $lambda = 10$ (azul). A la derecha, MSE de entrenamiento (gris punteado) y de validación (rojo) en función de $lambda$. La complejidad del modelo disminuye de izquierda a derecha: la figura es la de la @fig-polinomios, pero reflejada.],
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
          let estilos = (
            "grande": (paint: blue, thickness: 1.5pt, dash: "dashed"),
            "mejor": (paint: green.darken(20%), thickness: 1.8pt),
            "pequeno": (paint: red, thickness: 1.3pt),
          )

          plot.add(
            d-ridge.real,
            style: (stroke: (paint: gray.lighten(20%), thickness: 3pt)),
          )

          for (nombre, estilo) in estilos {
            plot.add(
              d-ridge.ajustes.at(nombre).curva,
              style: (stroke: estilo),
            )
          }

          plot.add(
            d-ridge.train,
            mark: "o",
            mark-size: 0.16,
            mark-style: (stroke: none, fill: black),
            style: (stroke: none),
          )
        },
      )
    ],

    [
      #custom-plot(
        x-min: 0.000001,
        x-max: 1000,
        y-min: 0.01,
        y-max: 10,
        x-label: $lambda$,
        y-label: [MSE],
        x-tick-step: 1,
        y-tick-step: 1,
        x-mode: "log",
        y-mode: "log",
        x-format: v => {
          let k = int(calc.round(calc.log(v)))
          if calc.rem(k, 3) == 0 { $10^#k$ } else { [] }
        },
        y-format: fmt-pot,
        size: (6, 5),
        {
          let train = d-ridge.curva_error.map(r => (r.lambda, r.mse_train))
          let val = d-ridge.curva_error.map(r => (r.lambda, r.mse_val))

          plot.add-hline(
            d-ridge.ruido_var,
            style: (stroke: (paint: black, thickness: 0.6pt, dash: "dotted")),
          )

          plot.add(
            train,
            style: (stroke: (paint: gray.darken(40%), thickness: 1.5pt, dash: "dashed")),
          )

          plot.add(
            val,
            style: (stroke: (paint: red.darken(20%), thickness: 2pt)),
          )
        },
      )
    ],
  ),
) <fig-ridge>

Con $lambda$ muy pequeño, el polinomio sigue oscilando entre los puntos de entrenamiento: sobreajusta. Con $lambda approx #num(d-ridge.mejor_lambda, digits: 3)$, el polinomio de grado 10 tiene un error de validación de #num(d-ridge.mejor_mse_val, digits: 3), casi igual al del polinomio de grado 4 sin regularizar. Con $lambda = 10$ el modelo es demasiado rígido y subajusta.

Lo importante es que no tuvimos que cambiar el número de parámetros: el modelo sigue teniendo 11, pero $lambda$ controla qué tan libremente se pueden usar. El grado del polinomio y $lambda$ son dos perillas distintas para lo mismo: la complejidad del modelo.

== Sparsity y regularización $L_1$: Lasso

=== Modelos sparse

La regularización también sirve para expresar algo que creemos sobre el problema. Supongamos que tenemos muchas features, pero creemos que solo unas pocas son realmente útiles para predecir $y$. En ese caso, nos gustaría un modelo donde la mayoría de los $theta_j$ sean exactamente cero, de forma que el modelo solo use unas pocas features. A un modelo así se le llama sparse (disperso).

La forma directa de pedir esto sería penalizar el número de parámetros distintos de cero:

$
  R(theta) = norm(theta)_0 = "número de" j "tales que" theta_j != 0.
$

El problema es que esta función no es continua: salta de 0 a 1 en cuanto un parámetro deja de ser cero, sin importar si se volvió 0.001 o 1000. No podemos minimizarla con descenso por gradiente. Por eso, en la práctica se reemplaza por la suma de los valores absolutos:

$
  R(theta) = norm(theta)_1 = sum_(j=1)^(n-1) abs(theta_j).
$

A la regresión lineal con este regularizador se le llama Lasso. Igual que con Ridge, las features se centran y escalan antes de ajustar el modelo, y el intercepto es $overline(y)$:

$
  J_lambda (theta) = 1/2 sum_(i=1)^m (h_theta (fvec(x)^(\(i\))) - y^(\(i\)))^2 + lambda sum_(j=1)^(n-1) abs(theta_j).
$

=== ¿Por qué $L_1$ produce ceros y $L_2$ no?

Una primera intuición: si nos restringimos a los puntos $(theta_1, theta_2)$ que están a distancia 1 del origen (el círculo unitario), los puntos con menor norma $L_1$ son $(plus.minus 1, 0)$ y $(0, plus.minus 1)$, que son justamente los que tienen un solo coeficiente distinto de cero. Es decir, minimizar la norma $L_1$ y buscar modelos sparse apuntan en la misma dirección.

Para ver el efecto sobre la regresión, es útil escribir los dos problemas de otra forma. En lugar de sumar la penalización a la función de costo, podemos poner un límite $t > 0$ al tamaño de los parámetros y minimizar $J(theta)$ solo entre los $theta$ que lo respetan. Para Ridge:

$
  min_theta J(theta) quad "sujeto a" quad sum_(j=1)^(n-1) theta_j^2 <= t,
$

y para Lasso:

$
  min_theta J(theta) quad "sujeto a" quad sum_(j=1)^(n-1) abs(theta_j) <= t.
$

Estas formulaciones son equivalentes a las anteriores: para cada $lambda$ hay un $t$ que da la misma solución. La relación va en sentido contrario: un $lambda$ grande corresponde a un $t$ pequeño, es decir, a una región permitida pequeña. Con $lambda = 0$ no hay restricción, lo que corresponde a un $t$ tan grande que el mínimo sin regularizar ya la cumple.

Con dos parámetros, $theta_1$ y $theta_2$, podemos dibujar las regiones permitidas:

- Para Ridge, $theta_1^2 + theta_2^2 <= t$ es un círculo.
- Para Lasso, $abs(theta_1) + abs(theta_2) <= t$ es un rombo.

La solución es el punto de la región permitida que toca la curva de nivel más baja de $J(theta)$. En la @fig-geometria la función $J(theta)$ es la misma en ambos paneles. El círculo no tiene esquinas, así que la curva de nivel lo toca en un punto cualquiera, donde ambos parámetros son distintos de cero. El rombo tiene esquinas sobre los ejes, y es muy común que la curva de nivel lo toque justo en una esquina, donde uno de los parámetros es exactamente cero.

#let panel-geo(datos, region, titulo) = custom-plot(
  x-min: -1.5,
  x-max: 3,
  y-min: -1.5,
  y-max: 3,
  x-label: $theta_1$,
  y-label: $theta_2$,
  x-tick-step: 1,
  y-tick-step: 1,
  size: (5.5, 5.5),
  {
    import cetz.draw: *

    // Ejes
    plot.add-hline(0, style: (stroke: 0.5pt + gray))
    plot.add-vline(0, style: (stroke: 0.5pt + gray))

    // Región permitida
    plot.annotate({
      line(
        ..region,
        close: true,
        fill: rgb("#BBD5F2").transparentize(30%),
        stroke: 1pt + blue.darken(20%),
      )
    })

    // Curvas de nivel de J
    let colores = (rgb("#1a237e"), rgb("#2979ff"), rgb("#00b8d4"), rgb("#76c403"))
    for (k, c) in datos.contornos.enumerate() {
      plot.add(
        c,
        style: (stroke: (paint: colores.at(k), thickness: if k == 2 { 1.6pt } else { 0.9pt })),
      )
    }

    // Mínimo sin regularizar y solución regularizada
    plot.add(
      (d-geo.theta_ols,),
      mark: "x",
      mark-size: 0.2,
      mark-style: (stroke: 1.5pt + black),
      style: (stroke: none),
    )
    plot.add(
      (datos.solucion,),
      mark: "o",
      mark-size: 0.26,
      mark-style: (stroke: 0.8pt + white, fill: red.darken(10%)),
      style: (stroke: none),
    )

    plot.annotate({
      content(
        (d-geo.theta_ols.at(0) + 0.15, d-geo.theta_ols.at(1)),
        anchor: "west",
        text(size: 9pt)[$hat(theta)$],
      )
      content((-1.35, 2.75), anchor: "north-west", text(size: 9pt, titulo))
    })
  },
)

#let t-geo = d-geo.t
#let circulo-geo = range(0, 121).map(k => {
  let a = 2 * calc.pi * k / 120
  (t-geo * calc.cos(a), t-geo * calc.sin(a))
})
#let rombo-geo = ((t-geo, 0), (0, t-geo), (-t-geo, 0), (0, -t-geo))

#figure(
  caption: [Curvas de nivel de una misma función $J(theta)$ y región permitida por el regularizador. La $times$ marca el mínimo sin regularizar, $hat(theta)$, y el punto rojo la solución regularizada: el punto de la región permitida donde $J$ es menor. La curva de nivel más gruesa es la que toca la región permitida. Con Ridge (izquierda) la solución tiene ambos parámetros distintos de cero. Con Lasso (derecha) la solución cae en una esquina del rombo, donde $theta_2 = 0$.],
  grid(
    columns: (1fr, 1fr),
    gutter: 1.2cm,
    align: top + center,
    panel-geo(d-geo.l2, circulo-geo, [Ridge ($L_2$)]),
    panel-geo(d-geo.l1, rombo-geo, [Lasso ($L_1$)]),
  ),
) <fig-geometria>

=== Caminos de coeficientes

Otra forma de ver la diferencia es graficar cómo cambia cada coeficiente a medida que aumentamos $lambda$. A estas curvas se les llama caminos de regularización. Para la @fig-caminos generamos #d-caminos.m casas con ocho features: área, habitaciones, baños, antigüedad, distancia al centro, y tres features de ruido que no tienen ninguna relación con el precio. El área y el número de habitaciones están correlacionadas, como en la realidad.

#let colores-caminos = (
  rgb("#1f77b4"),
  rgb("#ff7f0e"),
  rgb("#2ca02c"),
  rgb("#d62728"),
  rgb("#9467bd"),
  gray,
  gray,
  gray,
)

#let panel-camino(caminos) = custom-plot(
  x-min: 0.01,
  x-max: 10000,
  y-min: -2.5,
  y-max: 3.5,
  x-label: $lambda$,
  y-label: [Coeficiente],
  x-tick-step: 1,
  y-tick-step: 1,
  x-mode: "log",
  x-format: fmt-pot,
  size: (6, 5),
  {
    plot.add-hline(0, style: (stroke: 0.5pt + black))
    for (j, c) in caminos.enumerate() {
      plot.add(
        c,
        style: (
          stroke: (
            paint: colores-caminos.at(j),
            thickness: if j < 5 { 1.6pt } else { 1pt },
            dash: if j < 5 { none } else { "dashed" },
          ),
        ),
      )
    }
  },
)

#figure(
  caption: [Caminos de regularización de Ridge (izquierda) y Lasso (derecha) para el problema de las casas, con las features estandarizadas. Las líneas grises punteadas son las tres features de ruido. Con Ridge, todos los coeficientes se encogen poco a poco y ninguno llega a ser exactamente cero. Con Lasso, los coeficientes van llegando a cero uno por uno, y los primeros en desaparecer son los de las features de ruido.],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1.2cm,
      align: top + center,
      panel-camino(d-caminos.ridge),
      panel-camino(d-caminos.lasso),
    )
    #v(0.3cm)
    #set text(size: 9pt)
    #grid(
      columns: 6,
      column-gutter: 12pt,
      align: horizon,
      ..range(6).map(j => {
        let nombre = if j < 5 { d-caminos.nombres.at(j) } else { [Ruido] }
        box[
          #box(
            width: 0.6cm,
            height: 0pt,
            baseline: -0.3em,
            stroke: (
              top: (
                paint: colores-caminos.at(j),
                thickness: 1.6pt,
                dash: if j < 5 { none } else { "dashed" },
              ),
            ),
          )
          #nombre
        ]
      })
    )
  ],
) <fig-caminos>

Con Lasso, las tres features de ruido son las primeras en desaparecer: sus coeficientes son exactamente cero para todo $lambda$ mayor que #calc.round(calc.max(..d-caminos.lambda_cero_lasso.slice(5))). En cambio, el área (la feature más importante) es la última, y su coeficiente sobrevive hasta $lambda approx #calc.round(d-caminos.lambda_cero_lasso.at(0))$. Por eso se dice que Lasso hace selección de features de forma automática. Ridge, en cambio, mantiene todas las features en el modelo, con coeficientes cada vez más pequeños. Algo curioso pasa con las habitaciones en Ridge: su coeficiente crece un poco antes de encogerse. Como está correlacionada con el área, cuando la penalización encoge el coeficiente del área, parte de ese peso se pasa a las habitaciones.

Lasso tiene una desventaja: la función $abs(theta_j)$ no es derivable en $theta_j = 0$, así que no tiene solución cerrada ni podemos aplicar directamente el descenso por gradiente que vimos. Se usan otros algoritmos (por ejemplo, scikit-learn usa descenso por coordenadas, que optimiza un parámetro a la vez), pero no entraremos en esos detalles.

== Escogiendo $lambda$

El valor de $lambda$ se escoge con validación, igual que el grado del polinomio en la Parte 1:

+ Escoger una lista de valores candidatos. Como no sabemos ni el orden de magnitud de $lambda$, se usa una escala logarítmica, por ejemplo $10^(-4), 10^(-3), ..., 10^3$.
+ Para cada valor, estimar el error con validación hold-out o validación cruzada.
+ Escoger el $lambda$ con menor error y volver a entrenar con todos los datos de entrenamiento.

Esto es exactamente lo que muestra el panel derecho de la @fig-ridge.

== Ridge vs. Lasso

#align(
  center,
  table(
    columns: (5.8cm, 4cm, 4.2cm),
    align: (left, center, center),
    [], [*Ridge*], [*Lasso*],
    [Regularizador], [$1/2 sum_(j >= 1) theta_j^2$], [$sum_(j >= 1) abs(theta_j)$],
    [Solución cerrada], [Sí], [No],
    [Coeficientes exactamente cero], [No], [Sí],
    [Features correlacionadas], [Reparte el peso], [Tiende a escoger una],
  ),
)

En la práctica, Ridge suele funcionar mejor cuando muchas features aportan un poco cada una, y Lasso cuando creemos que solo unas pocas importan, o cuando queremos un modelo fácil de interpretar. También existe una combinación de ambos, llamada Elastic Net, que suma los dos regularizadores.

Una advertencia para el laboratorio: scikit-learn no usa exactamente la misma escala de $lambda$ que estas notas. Con nuestra definición de $J(theta)$, el parámetro `alpha` de `Ridge` coincide con $lambda$, pero el de `Lasso` corresponde a $lambda slash m$, porque `Lasso` divide el error entre el número de ejemplos. Por eso, un mismo valor de `alpha` no significa lo mismo en ambas clases.

= Parte 3: Interpretación probabilística

En la clase anterior justificamos el error cuadrático con dos argumentos intuitivos: los errores positivos y negativos no se cancelan, y los errores grandes se penalizan más. Pero hay otras funciones con esas propiedades (por ejemplo, el valor absoluto o la cuarta potencia). En esta parte veremos que, bajo ciertas suposiciones sobre cómo se generan los datos, el error cuadrático sale de forma natural. Esta idea será la base para construir la función de costo de clasificación en la próxima clase.

== Mínimos cuadrados como máxima verosimilitud

Supongamos que las etiquetas se generan así:

$
  y^(\(i\)) = theta^T fvec(x)^(\(i\)) + epsilon^(\(i\)),
$

donde $epsilon^(\(i\))$ es un término de error. Este error representa todo lo que el modelo no captura: features que no incluimos (por ejemplo, si la casa tiene vista al lago) o simplemente ruido aleatorio. Supongamos además que los $epsilon^(\(i\))$ son independientes entre sí y siguen todos la misma distribución normal con media cero y varianza $sigma^2$, lo que escribimos $epsilon^(\(i\)) tilde cal(N)(0, sigma^2)$. La densidad de cada error es

$
  p(epsilon^(\(i\))) = 1 / (sqrt(2 pi) sigma) exp(- (epsilon^(\(i\)))^2 / (2 sigma^2)).
$

Como $epsilon^(\(i\)) = y^(\(i\)) - theta^T fvec(x)^(\(i\))$, esto significa que, dado $fvec(x)^(\(i\))$, la etiqueta $y^(\(i\))$ sigue una distribución normal centrada en la predicción del modelo:

$
  p(y^(\(i\)) | fvec(x)^(\(i\)); theta) = 1 / (sqrt(2 pi) sigma) exp(- (y^(\(i\)) - theta^T fvec(x)^(\(i\)))^2 / (2 sigma^2)).
$

Puede notarse también que para la distribución de $y^(\(i\))$, la media es $theta^T fvec(x)^(\(i\))$ y la varianza es $sigma^2$, que no depende de $fvec(x)^(\(i\))$.

El punto y coma en $p(y^(\(i\)) | fvec(x)^(\(i\)); theta)$ indica que $theta$ no es una variable aleatoria, sino un valor fijo (aunque desconocido) que parametriza la distribución. La @fig-gaussiana ilustra este modelo.

#figure(
  caption: [Modelo probabilístico de la regresión lineal. Para cada $x$, la etiqueta $y$ sigue una distribución normal centrada en la recta $theta^T fvec(x)$ (las campanas, dibujadas de lado) con la misma varianza $sigma^2$ en todos los puntos.],
  custom-plot(
    x-min: 0,
    x-max: 5,
    y-min: 0,
    y-max: 5,
    x-label: $x$,
    y-label: $y$,
    x-tick-step: 1,
    y-tick-step: 1,
    size: (8, 5.5),
    {
      let t0 = d-gauss.theta0
      let t1 = d-gauss.theta1
      let s = d-gauss.sigma

      // Recta del modelo
      plot.add(
        domain: (0, 5),
        x => t0 + t1 * x,
        style: (stroke: (paint: red.darken(20%), thickness: 1.8pt)),
      )

      // Campanas gaussianas de lado
      for x0 in (1, 2.5, 4) {
        let mu = t0 + t1 * x0
        plot.add-vline(x0, min: mu - 3.2 * s, max: mu + 3.2 * s, style: (stroke: (paint: gray, thickness: 0.6pt, dash: "dashed")))
        plot.add(
          domain: (mu - 3.2 * s, mu + 3.2 * s),
          samples: 80,
          t => (x0 + 0.55 * calc.exp(-calc.pow(t - mu, 2) / (2 * s * s)), t),
          style: (stroke: (paint: blue.darken(10%), thickness: 1.3pt)),
        )
      }

      // Datos
      plot.add(
        d-gauss.puntos,
        mark: "o",
        mark-size: 0.12,
        mark-style: (stroke: none, fill: black),
        style: (stroke: none),
      )
    },
  ),
) <fig-gaussiana>

Dados todos los datos, ¿cuál es la probabilidad de haber observado justamente las etiquetas $fvec(y)$? Vista como función de $theta$, esta probabilidad se llama verosimilitud (likelihood). Como los errores son independientes, es el producto de las densidades de cada ejemplo:

$
  L(theta) = p(fvec(y) | fvec(X); theta) = product_(i=1)^m p(y^(\(i\)) | fvec(x)^(\(i\)); theta) = product_(i=1)^m 1 / (sqrt(2 pi) sigma) exp(- (y^(\(i\)) - theta^T fvec(x)^(\(i\)))^2 / (2 sigma^2)).
$

El principio de máxima verosimilitud dice que debemos escoger el $theta$ que hace más probables los datos que observamos, es decir, el que maximiza $L(theta)$.

Maximizar $L(theta)$ es lo mismo que maximizar cualquier función creciente de $L(theta)$. Es más cómodo maximizar su logaritmo, porque convierte el producto en una suma:

$
  ell(theta) = log L(theta)
  &= sum_(i=1)^m log (1 / (sqrt(2 pi) sigma) exp(- (y^(\(i\)) - theta^T fvec(x)^(\(i\)))^2 / (2 sigma^2))) \
  &= m log 1 / (sqrt(2 pi) sigma) - 1 / sigma^2 dot 1/2 sum_(i=1)^m (y^(\(i\)) - theta^T fvec(x)^(\(i\)))^2.
$

El primer término no depende de $theta$, y $1 slash sigma^2$ es una constante positiva. Por lo tanto, maximizar $ell(theta)$ es exactamente lo mismo que minimizar

$
  1/2 sum_(i=1)^m (y^(\(i\)) - theta^T fvec(x)^(\(i\)))^2 = J(theta).
$

Es decir, si suponemos que los errores son gaussianos e independientes, minimizar el error cuadrático es lo mismo que encontrar el estimador de máxima verosimilitud de $theta$. Nótese también que el resultado no depende de $sigma^2$: obtendríamos el mismo $theta$ aunque no conociéramos la varianza del ruido.

Esto no significa que el error cuadrático solo tenga sentido cuando el ruido es gaussiano. Es una justificación, no un requisito. Lo valioso es la receta: escoger una distribución para $y$, escribir la verosimilitud, y maximizarla. Si en lugar de una normal suponemos otra distribución, obtenemos otra función de costo.

== Visión bayesiana: regularización como conocimiento previo (opcional)

En la sección anterior tratamos $theta$ como un valor fijo pero desconocido. Esta es la visión frecuentista. Otra forma de ver el problema es la visión bayesiana: tratar $theta$ como una variable aleatoria, y expresar lo que creemos sobre sus valores antes de ver los datos mediante una distribución $p(theta)$, llamada distribución a priori (prior).

Después de observar los datos $S$, actualizamos esa creencia con el teorema de Bayes y obtenemos la distribución a posteriori:

$
  p(theta | S) = (p(S | theta) p(theta)) / p(S) = ((product_(i=1)^m p(y^(\(i\)) | fvec(x)^(\(i\)), theta)) p(theta)) / p(S).
$

Calcular la distribución a posteriori completa suele ser muy difícil, porque $p(S)$ requiere integrar sobre todos los valores posibles de $theta$. Una aproximación común es quedarse solo con el valor de $theta$ más probable según la posteriori. A este valor se le llama estimador MAP (máximo a posteriori):

$
  theta_"MAP" = arg max_theta product_(i=1)^m p(y^(\(i\)) | fvec(x)^(\(i\)), theta) p(theta).
$

Como $p(S)$ no depende de $theta$, no hace falta calcularlo. Nótese también que ahora escribimos $p(y^(\(i\)) | fvec(x)^(\(i\)), theta)$ con coma y no con punto y coma: como $theta$ ahora es una variable aleatoria, sí tiene sentido condicionar en su valor. Nótese que es la misma fórmula que la de máxima verosimilitud, con un factor adicional: el prior $p(theta)$.

=== Prior gaussiano: Ridge

Supongamos que, antes de ver los datos, creemos que los parámetros (excepto $theta_0$) probablemente son pequeños. Podemos expresar esto con un prior normal centrado en cero, con varianza $tau^2$:

$
  theta_j tilde cal(N)(0, tau^2), quad j = 1, ..., n-1.
$

Tomando el logaritmo del objetivo MAP:

$
  log (product_(i=1)^m p(y^(\(i\)) | fvec(x)^(\(i\)), theta) p(theta))
  = ell(theta) + sum_(j=1)^(n-1) log p(theta_j)
  = "constante" - 1 / sigma^2 J(theta) - 1 / (2 tau^2) sum_(j=1)^(n-1) theta_j^2.
$

Maximizar esto es lo mismo que minimizar lo que está restando. Multiplicando por $sigma^2$:

$
  theta_"MAP" = arg min_theta J(theta) + sigma^2 / tau^2 dot 1/2 sum_(j=1)^(n-1) theta_j^2.
$

Esto es exactamente Ridge, con $lambda = sigma^2 slash tau^2$.

- Si $tau$ es pequeño, creemos con mucha seguridad que los parámetros son cercanos a cero, y eso corresponde a un $lambda$ grande.
- Si $tau$ es grande, nuestro prior es casi plano (no creemos casi nada de antemano), y $lambda$ se acerca a cero, es decir, volvemos a máxima verosimilitud.
- Si el ruido $sigma^2$ es grande, los datos son poco confiables, y el prior pesa más.

=== Prior de Laplace: Lasso

Si en lugar de una normal usamos una distribución de Laplace, $p(theta_j) prop exp(-abs(theta_j) slash b)$, que tiene un pico más agudo en cero, el mismo procedimiento da

$
  theta_"MAP" = arg min_theta J(theta) + sigma^2 / b sum_(j=1)^(n-1) abs(theta_j),
$

que es Lasso con $lambda = sigma^2 slash b$. El pico agudo en cero del prior de Laplace es la versión probabilística de las esquinas del rombo en la @fig-geometria.
