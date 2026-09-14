#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *

#let fvec(x) = math.bold(math.upright(x))

#set text(lang: "es")

#show: minimal-note.with(
  title: [Nociones básicas. Parte 1],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

= Escalares, vectores, matrices y conjuntos

Un *escalar* es un único valor:

$
  a in RR
$

Un *vector* es una colección ordenada de valores, los denotamos con letra negrita (ejemplo: $fvec(x), fvec(z)$):

$
  fvec(x) = mat(x_1; x_2; dots.v; x_d) in RR^d
$

Para diferenciar a los vectores de los puntos, se escriben como columna. Para facilitar la escritura, se suelen escribir transpuestos. Por ejemplo: $fvec(x)^T=[x_1, dots.h, x_d]^T$.

Los vectores representan conjuntos de características. Para una casa, por ejemplo:
$
  fvec(x) = mat(120; 3; 10) = mat("área"; "habitaciones"; "antigüedad")
$

Una *matriz* es un conjunto de valores ordenados en dos dimensiones, por ejemplo:

$
  X in RR^(n times d)
$

Por ejemplo:

$
  X =
  mat(
    120, 3, 10;
    80, 2, 20;
    150, 4, 5
  )
$

Un *conjunto* es una agrupación de elementos. Por ejemplo, el conjunto de entrenamiento de un modelo puede ser $italic(S) = {(fvec(x)_i, y_i)}_(i=1)^n$ o $italic(S) = {(fvec(x)_i, y_i) | i=1,2,dots.h,n}$, donde $fvec(x)_i$ son las características de una observación y $y_i$ es la etiqueta.

= Operaciones de vectores y matrices

== Suma, resta

Para vectores y matrices con las mimas dimensiones, la suma y la resta se realizan *por componente*.

$
  mat(x_1; x_2) + mat(y_1; y_2) = mat(x_1 + y_1; x_2 + y_2) \
  mat(x_1; x_2) - mat(y_1; y_2) = mat(x_1 - y_1; x_2 - y_2)
$

== Producto punto

Para dos vectores $fvec(x), fvec(w) in RR^d$ su producto punto es:

$
  fvec(w) dot fvec(x) = sum_(j=1)^d w_j x_j
$
Por ejemplo: $fvec(w)^T = [2, -1]^T, fvec(x) = [3, 4]^T$ entonces:

$
  fvec(w) dot fvec(x) = 2(3) - 1(4) = 2
$

El producto punto produce un *escalar*. Geométricamente:

$
  w^T x = norm(w) norm(x) cos(theta)
$

Por lo tanto, también está relacionado con el grado de alineación entre dos vectores. El producto punto de dos vectores perpendiculares es 0, es máximo cuando apuntan en la misma dirección y mínimo cuando apuntan en sentidos opuestos.

== Normas y distancias

La norma euclidiana ($L_2$): de un vector es:

$
  norm(x)_2 =
  sqrt(sum_i x_i^2)
$

La distancia euclidiana entre dos puntos es:

$
  d(x, z) = norm(x - z)_2
$

La norma $L_1$:

$
  norm(x)_1 = sum_i abs(x_i)
$


== Multiplicación matricial

Si $fvec(A) in RR^(m times n)$ y $fvec(B) in RR^(n times p)$, entonces:
$
  fvec(A) fvec(B) in RR^(m times p)
$

Las dimensiones internas deben coincidir.

Por ejemplo, si $fvec(X) in RR^(100 times 5)$ y $fvec(w) in RR^5$, entonces:

$
  fvec(X) fvec(w) in RR^100
$

=== Propiedades de la multiplicación de matrices

1. Asociativa: $fvec(A)(fvec(B)fvec(C))=(fvec(A)fvec(B))fvec(C)$
2. Distributiva respecto a la suma: $fvec(A)(fvec(B) + fvec(C)) = fvec(A)fvec(B) + fvec(A)fvec(C)$
3. Elemento neutro: $fvec(A)fvec(I) = fvec(A)$ donde $fvec(I)$ es la matriz identidad

== Transpuesta

La transpuesta intercambia filas y columnas:

$
  fvec(A) in RR^(m times n)
  quad arrow.r quad
  fvec(A)^T in RR^(n times m)
$

Es distributiva: $(fvec(A) fvec(B))^T = fvec(A)^T fvec(B)^T$

Como se mencionó anteriomente, para un vector columna $fvec(x)$, podemos escribir $fvec(x)^T$ como vector fila. Por eso:

$
  fvec(w)^T fvec(x) = fvec(w) dot fvec(x)
$

= Transformaciones lineales

Una transformación es un nombre que se le da a una función entre espacios vectoriales:

$
  T: RR^n arrow RR^m
$

Por ejemplo de $RR arrow RR^2$.

$
  T(x) = mat(2x; 3x)
$

Otro ejemplo, de $RR^2 arrow RR^2$.

$
  T(fvec(x))= mat(2x_1+x_2; x_1-3x_2)
$

Una transformación es lineal si satisface dos condiciones:

$
  T(x + y) = T(x) + T(y) \
  T(alpha x) = alpha T(x)
$

Visualmente, una transformación lineal mantiene el origen fijo, y todas las líneas rectas se mantienen rectas.

Toda transformación lineal puede representarse mediante una matriz:

$
  T(fvec(x)) = fvec(A) fvec(x)
$

El ejemplo anterior de $RR^2 arrow RR^2$ puede expresarse de forma matricial como:

$
  fvec(A) = mat(2, 1; 1, -3)
$

Geométricamente, una transformación lineal puede realizar operaciones como escalamiento, rotación, reflexión o proyección. En este enlace se pueden ver #link("https://www.khanacademy.org/math/linear-algebra/matrix-transformations/linear-transformations/a/visualizing-linear-transformations")[más ejemplos de transformaciones lineales]

= Transformaciones afines

Una transformación afín es una transformación lineal seguida de una traslación, la forma general es:

$
  T(fvec(x)) = fvec(A) fvec(x) + fvec(b)
$

El origen no se mantiene fijo, por lo que no es estrictamente lineal. Esta forma es muy común en aprendizaje automático, donde $fvec(A)$ representa la combinación lineal de características y $fvec(b)$ es un sesgo.

= Interpretación geométrica de $fvec(w)^T fvec(x) + fvec(b) = 0$

Consideremos:

$
  fvec(w)^T fvec(x) + fvec(b) = 0
$

En dos dimensiones:

$
  w_1 x_1 + w_2 x_2 + b = 0
$

describe una recta.

En tres dimensiones describe un plano ($w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$).

En $d$ dimensiones describe un *hiperplano*, donde el vector $fvec(w)$ es perpendicular a dicho hiperplano.

Por ejemplo, la recta $-2x_1+3x_2 = 0$ tiene $fvec(w)^T=[-2, 3]^T$:

#align(center)[
  #coordinate-system({
    plot.add(
      domain: (-3, 3),
      x => (-2 / 3) * x,
    )
    vector((0, 0), (2, 3), color: black, label: [$fvec(w)$], label-position: "head")
  })
]


= Sistemas lineales

Un sistema de ecuaciones lineales puede escribirse como:

$
  A x = b
$

Si $A$ es cuadrada e invertible:

$
  x = A^(-1) b
$

Sin embargo, en aplicaciones numéricas normalmente no calculamos explícitamente la inversa de una matriz grande. Se utilizan algoritmos específicos para resolver el sistema.

= Autovalores y autovectores

Un vector no nulo $fvec(v)$ es un autovector de $fvec(A)$ si:

$
  fvec(A) fvec(v) = lambda fvec(v)
$

donde $lambda$ es el autovalor correspondiente. La transformación cambia la magnitud del vector, pero mantiene su dirección:

= Nociones básicas de probabilidad

== Variable aleatoria y espacio muestral

Una *variable aleatoria* $X$ es una función que asigna un valor numérico a cada resultado de un fenómeno aleatorio. Pueden ser discretas o continuas:

- *Discretas*: Toman valores finitos o contables (ej. la etiqueta de clase $Y in {0, 1}$).
- *Continuas*: Toman valores en un intervalo continuo (ej. el valor objetivo $Y in RR$).

== Distribución de probabilidad

Describe cómo se distribuyen las probabilidades sobre los posibles valores de $X$:

- *Función de masa de probabilidad (PMF)* para variables discretas:
$
  P(X = x) = p(x), quad "con" sum_x p(x) = 1
$

- *Función de densidad de probabilidad (PDF)* para variables continuas:
$
  f(x) = 0, quad "con" integral_(-infinity)^(infinity) f(x) dif x = 1
$

== Esperanza matemática, varianza y covarianza

- *Esperanza (Valor esperado)* $E[X]$: Representa la media ponderada de los valores que toma $X$:
$
  E[X] = sum_x x P(X = x) quad "o" quad E[X] = integral_(-infinity)^(infinity) x f(x) dif x
$

- *Varianza* $"Var"(X)$: Mide la dispersión de la variable aleatoria respecto a su esperanza:
$
  "Var"(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$

- *Covarianza* $"Cov"(X, Y)$: Cuantifica el grado de variación conjunta entre dos variables:
$
  "Cov"(X, Y) = E[(X - E[X])(Y - E[Y])]
$

== Probabilidad conjunta, marginal y condicional

Dadas dos variables aleatorias $X$ e $Y$:

- *Probabilidad conjunta* $P(X = x, Y = y)$: Probabilidad de que $X$ e $Y$ ocurran simultáneamente.
- *Probabilidad marginal* $P(X = x)$: Se obtiene al sumar o integrar sobre la otra variable:
$
  P(X = x) = sum_y P(X = x, Y = y)
$
- *Probabilidad condicional* $P(Y = y | X = x)$: Probabilidad de que ocurra $Y$ dado que se ha observado $X$:
$
  P(Y = y | X = x) = (P(X = x, Y = y)) / (P(X = x))
$

== Independencia estocástica

Dos variables $X$ e $Y$ son *independientes* si el conocimiento sobre una no altera la probabilidad de la otra:

$
  P(X = x, Y = y) = P(X = x) dot P(Y = y) arrow.r P(Y = y | X = x) = P(Y = y)
$

Es decir, si $X$ e $Y$ son independientes, la probabilidad condicional de $Y$ dado $X$ es simplemente la probabilidad marginal de $Y$. Y la probabilidad conjunta se puede factorizar como el producto de las probabilidades marginales.

== Teorema de Bayes

Permite actualizar la probabilidad de una hipótesis $H$ dada la evidencia observada $E$.

$
  P(H | E) = (P(E | H) dot P(H)) / (P(E))
$

Donde:
- $P(H | E)$ es la *probabilidad a posteriori*.
- $P(E | H)$ es la *verosimilitud*.
- $P(H)$ es la *probabilidad a priori*.
- $P(E) = sum_l P(E | H = l) P(H = l)$ es la *evidencia*.

En el siguiente enlace se encuentra una #link("https://www.3blue1brown.com/lessons/bayes-theorem/")[explicación más detallada del Teorema de Bayes] (puede leerse el artículo completo o visualizar el video).


= Notación para problemas de optimización

Formalizamos un problema de minimización como:

$
  hat(y) = min_(x in X) f(x)
$

Donde: 

- $hat(y)$ es el valor mínimo de la función objetivo $f(x)$.
- $X subset RR^k$ es el conjunto de posibles soluciones (conjunto factible).
- $f(x): RR^k arrow.r RR$ es la función objetivo que queremos minimizar.

Si en cambio, queremos retornar el valor de $x$ que minimiza la función, lo denotamos como:

$
  hat(x) = argmin_(x in X) f(x)
$

Por ejemplo, si queremos minimizar la función $f(x) = 3(x - 2)^2 + 100$, entonces:
$
  hat(y) = min_(x in RR) 3(x - 2)^2 + 100 \
  hat(y) = 100 \ \
  hat(x) = argmin_(x in RR) 3(x - 2)^2 + 100 \
  hat(x) = 2
$