// OUTLINE (3 horas)
// Introducción: aprendizaje no supervisado
// 1. K-means
// 1.1. El problema de agrupamiento
// 1.2. El algoritmo (Ejercicio 1, en papel)
// 1.3. La distorsión y la convergencia
//      -> Lab, Parte 1: programar K-means
// 1.4. Inicialización
// 1.5. Elección de k (Ejercicio 2, en papel)
// 1.6. K-means en la práctica
// 1.7. Aplicación: los colores de una foto
//      -> Lab, Parte 2: elegir k y comprimir una foto
// 1.8. Limitaciones
// 2. DBSCAN
// 2.1. Grupos como regiones densas (Ejercicio 3, en papel)
// 2.2. Cómo se forman los grupos
// 2.3. Elegir eps
// 2.4. Limitaciones
// 3. Agrupamiento jerárquico
// 3.1. Unir los grupos más cercanos (Ejercicio 4, en papel)
// 3.2. Distancia entre grupos
// 4. Densidad contra particional
//      -> Lab, Parte 3: los tres algoritmos sobre los mismos datos
//
// Hay dos versiones de estas notas, generadas desde este mismo archivo:
//   pública:      typst compile --root . notes/09-agrupamiento.typ
//   del profesor: typst compile --root . --input version=profesor notes/09-agrupamiento.typ notes/09-agrupamiento-profesor.pdf
//
// Las figuras están en notes/09-figuras.typ (las comparten las diapositivas).
// El código está en notes/code/09/, y su salida se genera al ejecutarlo con
// `python notes/scripts/09-agrupamiento-figuras.py`, que también calcula los datos de las figuras.

#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *
#import "09-figuras.typ": *

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
  title: if profesor [Agrupamiento (versión del profesor)] else [Agrupamiento],
  author: [Tomas Santana],
  date: datetime.today().display("[month repr:long], [year]"),
)

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

#let argmin = math.op("argmin", limits: true)

#if profesor [
  #text(fill: red, weight: "bold", size: 12pt)[Plan de la clase (3 horas)]

  #prof[
    - *0:00–0:10 · Introducción.*
    - *0:10–0:35 · K-means.* Secciones 1.1 a 1.3. Ejercicio 1 en papel.
    - *0:35–0:55 · Lab, Parte 1.* Programar K-means (labs/07).
    - *0:55–1:15 · K-means.* Secciones 1.4 a 1.6. Ejercicio 2 en papel.
    - *1:15–1:25 · Descanso.*
    - *1:25–1:30 · Sección 1.7* (la foto).
    - *1:30–1:45 · Lab, Parte 2.* Elegir $k$ y comprimir una foto.
    - *1:45–1:50 · Sección 1.8* (limitaciones).
    - *1:50–2:12 · DBSCAN.* Parte 2. Ejercicio 3 en papel.
    - *2:12–2:30 · Jerárquico.* Parte 3. Ejercicio 4 en papel.
    - *2:30–2:35 · Parte 4.* Comparación.
    - *2:35–2:55 · Lab, Parte 3.* Los tres algoritmos sobre los mismos datos.
    - *2:55–3:00 · Cierre.* Resumen.
    - *Plan B.* Si van atrasados, dejar como lectura las secciones 2.4 y 3.2 y la Parte 4: la Parte 3 del lab muestra lo mismo.
  ]
]

= Introducción: aprendizaje no supervisado

Hasta ahora, todos los problemas vistos han sido de aprendizaje supervisado: cada ejemplo $x^((i))$ venía con su etiqueta $y^((i))$, y el objetivo era predecir $y$ a partir de $x$. En el aprendizaje no supervisado solo se tienen los $x^((i))$, sin etiquetas. El objetivo es encontrar estructura en los datos.

En esta clase se ve el problema más común de este tipo: el agrupamiento (clustering), que consiste en dividir los datos en grupos de ejemplos parecidos. Algunos usos:

- *Segmentar clientes.* Una tienda agrupa a sus clientes según lo que compran, para diseñar una campaña distinta para cada grupo.
- *Agrupar documentos.* Un sitio de noticias junta las noticias que hablan del mismo tema.
- *Reducir los colores de una foto.* Se agrupan los colores de los píxeles y cada píxel se reemplaza por el color de su grupo. Se hace en la sección 1.7.

#prof[Preguntar: ¿qué otros datos agruparían sin tener etiquetas? Pedir dos o tres ejemplos.]

= Parte 1: K-means

== El problema de agrupamiento

Se tiene un conjunto de entrenamiento ${x^((1)), ..., x^((n))}$, con $x^((i)) in RR^d$, sin etiquetas. El objetivo es dividirlo en $k$ grupos (clusters), de forma que los ejemplos de un mismo grupo estén cerca entre sí. El número de grupos $k$ se escoge de antemano.

== El algoritmo

K-means representa cada grupo $j$ con un punto $mu_j in RR^d$, su *centroide*, que es el centro del grupo. El algoritmo sigue los siguientes pasos:

+ Inicializar los centroides $mu_1, ..., mu_k in RR^d$ al azar. Por ejemplo, escoger $k$ ejemplos al azar y usarlos como centroides.
+ Repetir hasta que nada cambie:
  + Asignar. Para cada ejemplo $i$, asignarlo al centroide más cercano:
    $
      c^((i)) := argmin_j norm(x^((i)) - mu_j)^2.
    $
  + Mover. Para cada grupo $j$, mover el centroide al promedio de los ejemplos asignados a él:
    $
      mu_j := (sum_(i=1)^n 1{c^((i)) = j} x^((i))) / (sum_(i=1)^n 1{c^((i)) = j}).
    $

Aquí $c^((i)) in {1, ..., k}$ es el grupo del ejemplo $i$, y $1{c^((i)) = j}$ vale 1 si el ejemplo $i$ está en el grupo $j$ y 0 si no. La fórmula de $mu_j$ dice exactamente eso: sumar los ejemplos del grupo $j$ y dividir entre cuántos son.

La @fig-pasos muestra el algoritmo con $k = 3$. Los centroides empiezan en lugares aleatorios, pero después de unas pocas iteraciones cada uno queda en el centro de un grupo.

#figure(
  caption: [K-means con $k = 3$. Los ejemplos son puntos y los centroides, cruces. (a) Los datos, sin etiquetas. (b) Centroides iniciales al azar. (c) Cada ejemplo se pinta del color de su centroide más cercano. (d) Cada centroide se mueve al promedio de sus ejemplos (las flechas). (e) Se vuelve a asignar. (f) Después de #d.pasos.iteraciones iteraciones, nada cambia.],
  fig-pasos(),
) <fig-pasos>

#ejercicio("Ejercicio 1")[
  Considerar los datos en una dimensión $x = 1, 2, 4, 7, 8, 10$ y $k = 2$, con centroides iniciales $mu_1 = 1$ y $mu_2 = 2$.
  + Aplicar K-means hasta que nada cambie. En cada iteración, anotar a qué grupo va cada punto y los nuevos centroides.
  + Calcular $J = sum_i (x^((i)) - mu_(c^((i))))^2$ después de mover los centroides en cada iteración. ¿Qué pasa con $J$?
]

#prof[Solución. Iteración 1: grupos {1} y {2, 4, 7, 8, 10}; $mu = (1, 6.2)$; $J = 40.8$. Iteración 2: {1, 2} y {4, 7, 8, 10}; $mu = (1.5, 7.25)$; $J = 19.25$. Iteración 3: {1, 2, 4} y {7, 8, 10}; $mu = (2.33, 8.33)$; $J = 9.33$. Iteración 4: nada cambia. $J$ baja en cada iteración: es la idea de la sección 1.3.]

== La distorsión y la convergencia

¿Cómo se sabe que el algoritmo termina? Se define la distorsión:

$
  J(c, mu) = sum_(i=1)^n norm(x^((i)) - mu_(c^((i))))^2.
$

$J$ es la suma de las distancias al cuadrado de cada ejemplo a su centroide. Un $J$ pequeño significa grupos compactos. Los dos pasos de K-means reducen $J$ (o la mantienen igual):

- *Asignar* deja fijos los centroides y le da a cada ejemplo el centroide más cercano. Cada término de la suma queda igual o más pequeño.
- *Mover* deja fijos los grupos y pone cada centroide en el promedio de su grupo. El promedio es el punto que minimiza la suma de las distancias al cuadrado a un conjunto de puntos (es lo mismo que pasa en regresión lineal con un modelo constante), así que $J$ tampoco aumenta.

Como $J$ nunca aumenta, y solo hay un número finito de formas de repartir los ejemplos en $k$ grupos, el algoritmo termina. En la @fig-pasos, $J$ pasa de #num(d.pasos.J.at(0)) a #num(d.pasos.J.last()).

Pero que termine no quiere decir que encuentre el mejor agrupamiento posible: puede caer en un *mínimo local* de $J$.

#prof[Lab, Parte 1 (0:35–0:55): programar `assign_clusters`, `update_centroids` y `distortion` en `agrupamiento.py`, y ver cómo baja $J$.]

== Inicialización

El resultado de K-means depende de dónde empiezan los centroides. La @fig-minimos muestra los mismos datos agrupados con dos inicializaciones distintas. A la derecha, dos centroides empezaron en el mismo grupo: el algoritmo terminó, pero en un mínimo local con una distorsión cuatro veces mayor.

#figure(
  caption: [Los mismos datos con $k = 4$ y dos inicializaciones distintas. A la izquierda, el mejor agrupamiento. A la derecha, un mínimo local: dos centroides se repartieron un grupo y otro centroide quedó entre dos grupos.],
  fig-minimos(),
) <fig-minimos>

Pueden usarse varias estrategias para evitar caer en mínimos locales:

- *Repetir varias veces.* Correr K-means con varias inicializaciones al azar y quedarse con el resultado de menor $J$. En scikit-learn, el número de repeticiones es `n_init`.
- *k-means++.* En lugar de escoger los centroides iniciales completamente al azar, se escogen separados entre sí. El primero es un ejemplo al azar. Cada uno de los siguientes también es un ejemplo al azar, pero los ejemplos lejos de los centroides ya escogidos tienen más probabilidad de salir (la probabilidad es proporcional a la distancia al cuadrado al centroide más cercano). Así, es más difícil que dos centroides empiecen en el mismo grupo. Es el método por defecto en scikit-learn.

== Elección de $k$

No se puede escoger el $k$ con menor $J$: más grupos siempre dan grupos más pequeños y compactos, y con $k = n$ cada ejemplo es su propio grupo y $J = 0$. Hay dos herramientas comunes.

*El método del codo.* Graficar $J$ contra $k$. $J$ baja rápido mientras agregar un grupo separa grupos reales, y después baja despacio, porque solo está partiendo grupos que ya estaban bien separados. Se escoge el $k$ donde la curva hace un "codo" (@fig-codo, izquierda).

*El coeficiente de silueta.* Para cada ejemplo $i$ se calcula:

- $a(i)$: la distancia promedio de $x^((i))$ a los demás ejemplos de su grupo;
- $b(i)$: la distancia promedio de $x^((i))$ a los ejemplos del grupo más cercano que no es el suyo;
- la silueta del ejemplo:
  $
    s(i) = (b(i) - a(i)) / max(a(i), b(i)).
  $

$s(i)$ está entre $-1$ y $1$. Si es cercana a 1, el ejemplo está mucho más cerca de su grupo que del siguiente: está bien agrupado. Si es cercana a 0, está en la frontera entre dos grupos. Si es negativa, probablemente está en el grupo equivocado. La silueta del agrupamiento es el promedio de las $s(i)$, y se escoge el $k$ con la silueta más alta (@fig-codo, derecha).

#figure(
  caption: [La distorsión $J$ y la silueta para distintos valores de $k$, en datos con cuatro grupos. $J$ baja rápido hasta $k = 4$ y luego casi no cambia (el codo). La silueta es máxima en $k = 4$.],
  fig-codo(),
) <fig-codo>

En muchos problemas, $k$ también depende del uso. Si una tienda tiene presupuesto para tres campañas, querrá tres grupos de clientes, sin importar lo que digan el codo o la silueta.

#ejercicio("Ejercicio 2")[
  Con el resultado final del Ejercicio 1 (grupos ${1, 2, 4}$ y ${7, 8, 10}$):
  + Calcular $a$, $b$ y la silueta $s$ del punto $x = 4$.
  + Hacer lo mismo para el punto $x = 1$.
  + ¿Cuál de los dos puntos está mejor agrupado? ¿Tiene sentido viendo los datos?
]

#prof[Solución. Para $x = 4$: $a = (3 + 2) slash 2 = 2.5$, $b = (3 + 4 + 6) slash 3 = 4.33$, $s = (4.33 - 2.5) slash 4.33 = 0.42$. Para $x = 1$: $a = (1 + 3) slash 2 = 2$, $b = (6 + 7 + 9) slash 3 = 7.33$, $s = (7.33 - 2) slash 7.33 = 0.73$. El 1 está mejor agrupado; el 4 está más cerca de la frontera entre los dos grupos.]

== K-means en la práctica

En scikit-learn, K-means se usa como cualquier modelo, pero `fit` no recibe etiquetas:

#codigo("kmeans")
#salida("kmeans")

- `fit_predict` agrupa los datos y devuelve el grupo de cada ejemplo.
- `cluster_centers_` son los centroides y `inertia_` es la distorsión $J$.
- `predict` asigna ejemplos nuevos al centroide más cercano.

Para escoger $k$, se corre K-means con varios valores y se comparan $J$ y la silueta (`silhouette_score`). Con los datos de la @fig-codo:

#codigo("elegir_k")
#salida("elegir_k")

Una consideración importante es que K-means usa distancias, así que una feature con números grandes domina a las demás. La @fig-escala muestra clientes descritos por su edad (en años) y su ingreso (en dólares al año). Hay dos grupos claros por edad, pero las diferencias de ingreso son de miles de dólares y las de edad, de unos pocos años, así que K-means sin estandarizar los separa por ingreso. Por eso, antes de agrupar, se estandariza cada feature (restar su media y dividir entre su desviación estándar) con `StandardScaler`, como en las clases anteriores.

#figure(
  caption: [Clientes con dos features: la edad y el ingreso. Sin estandarizar, K-means separa por ingreso, que tiene números mucho más grandes. Estandarizando, encuentra los dos grupos de edad.],
  fig-escala(),
) <fig-escala>

== Aplicación: los colores de una foto

Una foto a color es una tabla de píxeles, y cada píxel tiene tres números: cuánto rojo, verde y azul tiene (RGB). Si se ve cada píxel como un punto en $RR^3$, se pueden agrupar los píxeles con K-means. Los centroides son los $k$ colores "típicos" de la foto. Si se reemplaza cada píxel por el centroide de su grupo, se obtiene la misma foto con solo $k$ colores (@fig-colores).

#figure(
  caption: [Una foto y su versión con 2, 4 y 16 colores, encontrados con K-means. Debajo, los colores (centroides) de cada versión. Con 16 colores, la foto se ve casi igual que la original, que tiene decenas de miles de colores distintos.],
  fig-colores(),
) <fig-colores>

#codigo("colores")
#salida("colores")

La foto tiene #d.colores.alto × #d.colores.ancho = 273 280 píxeles. Para que sea rápido, K-means se entrena con una muestra de 5000 píxeles, y después `predict` asigna cada píxel de la foto a su color más cercano. Esto es una forma de *compresión*: en lugar de guardar tres números por píxel, basta guardar los 16 colores y, para cada píxel, cuál de los 16 le toca.

#prof[Lab, Parte 2 (1:30–1:45): elegir $k$ con el codo y la silueta en los datos de vinos (y ver el efecto de estandarizar), y comprimir la foto con distintos $k$.]

== Limitaciones

K-means funciona bien cuando los grupos son "redondos", de tamaño parecido y bien separados, porque cada ejemplo va al centroide más cercano. Tiene tres limitaciones:

- Debe definirse $k$.
- Todo ejemplo va a algún grupo, incluso los valores atípicos, que no se parecen a ninguno.
- Solo encuentra grupos redondos. En la @fig-kmeans-lunas hay dos grupos en forma de luna. K-means con $k = 2$ los corta por la mitad, porque divide el plano en dos regiones según cuál centroide está más cerca, y ninguna recta separa las dos lunas.

#figure(
  caption: [Dos grupos en forma de luna, con algunos valores atípicos. K-means con $k = 2$ no los encuentra: corta el plano en dos mitades.],
  fig-kmeans-lunas(),
) <fig-kmeans-lunas>

= Parte 2: DBSCAN

== Grupos como regiones densas

DBSCAN (density-based spatial clustering of applications with noise) usa otra idea de lo que es un grupo: una región donde hay muchos ejemplos juntos, separada de las demás por zonas con pocos ejemplos. Tiene dos parámetros:

- `eps` ($epsilon$): el radio del *vecindario* de un ejemplo. El vecindario de $x$ son todos los ejemplos a una distancia de $x$ menor o igual a $epsilon$, incluyendo al propio $x$.
- `min_samples`: cuántos ejemplos tiene que haber en un vecindario para considerarlo denso.

Con ellos, cada ejemplo es de uno de tres tipos (@fig-tipos):

- *Núcleo*: su vecindario tiene al menos `min_samples` ejemplos.
- *Frontera*: no es núcleo, pero está en el vecindario de algún núcleo.
- *Ruido*: no es ninguno de los dos.

#figure(
  caption: [Los tres tipos de puntos con `eps` igual al radio de los círculos y `min_samples` $= #d.tipos.min$. Los puntos núcleo tienen al menos 3 puntos en su círculo (contándose a sí mismos). El punto de frontera solo tiene 2, pero está dentro del círculo de un núcleo. Los dos puntos de ruido están solos.],
  fig-tipos(),
) <fig-tipos>

#ejercicio("Ejercicio 3")[
  Considerar los datos en una dimensión $x = 1, 2, 3, 5, 9, 10, 11, 15$, con `eps` $= 1.5$ y `min_samples` $= 3$.
  + Escribir el vecindario de cada punto y decir si es núcleo, frontera o ruido.
  + ¿Qué grupos forma DBSCAN?
  + Repetir con `eps` $= 2.5$. ¿Qué cambia?
]

#prof[Solución con `eps` $= 1.5$: núcleos 2 y 10; fronteras 1, 3, 9 y 11; ruido 5 y 15. Grupos {1, 2, 3} y {9, 10, 11}. Con `eps` $= 2.5$: núcleos 1, 2, 3, 9, 10 y 11; frontera 5 (está a 2 del núcleo 3); ruido 15. Grupos {1, 2, 3, 5} y {9, 10, 11}. Un `eps` más grande junta más puntos.]

== Cómo se forman los grupos

Dos núcleos que están en el vecindario uno del otro van en el mismo grupo. Un grupo de DBSCAN es un conjunto de núcleos conectados de esta forma, en cadena, más los puntos de frontera que están cerca de ellos. El algoritmo es:

+ Tomar un punto núcleo que todavía no tenga grupo y crear un grupo nuevo con él.
+ Agregar al grupo todos los puntos de su vecindario. Para cada núcleo que se agregó, agregar también los de su vecindario, y así hasta que no se agregue nadie.
+ Repetir con otro núcleo sin grupo. Al final, los puntos que no quedaron en ningún grupo son ruido.

Así, un grupo puede tener cualquier forma: basta que se pueda recorrer de punto en punto sin cruzar una zona vacía. La @fig-dbscan-lunas muestra el resultado en las lunas: DBSCAN encuentra los dos grupos y marca los valores atípicos como ruido. Además, no hubo que darle el número de grupos.

#figure(
  caption: [DBSCAN en los datos de la @fig-kmeans-lunas, con `eps` $= 0.15$ y `min_samples` $= 5$. Encuentra las dos lunas y marca #d.lunas.ruido puntos como ruido (✕).],
  fig-dbscan-lunas(),
) <fig-dbscan-lunas>

#codigo("dbscan")
#salida("dbscan")

En scikit-learn, el grupo de los puntos de ruido es $-1$.

== Elegir `eps`

El resultado depende mucho de `eps` (@fig-eps). Si es muy pequeño, casi ningún punto es núcleo y casi todo es ruido. Si es muy grande, todos los puntos quedan conectados y hay un solo grupo.

#figure(
  kind: image,
  caption: [DBSCAN en las lunas con `min_samples` $= 5$ y tres valores de `eps`. Con 0.05, casi todo es ruido (gris) y aparecen muchos grupos pequeños. Con 0.15, los dos grupos correctos. Con 0.5, todo es un solo grupo.],
  fig-eps(),
) <fig-eps>

Una forma de escoger `eps`:

+ Escoger `min_samples`. Un valor común es 5, o el doble del número de features.
+ Para cada punto, calcular la distancia a su vecino número `min_samples` (contándose a sí mismo).
+ Ordenar esas distancias y graficarlas (@fig-kdist). La mayoría de los puntos están en zonas densas y tienen distancias pequeñas; los puntos de ruido tienen distancias grandes. Se escoge `eps` donde la curva empieza a subir rápido.

#figure(
  caption: [La distancia de cada punto de las lunas a su 5.º vecino, ordenadas de menor a mayor. La curva sube rápido al final, en los puntos aislados. La línea roja es `eps` $= 0.15$.],
  fig-kdist(),
) <fig-kdist>

== Limitaciones

DBSCAN usa un solo `eps` para todos los datos, así que tiene problemas cuando los grupos tienen *densidades distintas*. En la @fig-densidades hay un grupo muy denso y otro disperso. Un `eps` que funciona para el grupo denso deja casi todo el grupo disperso como ruido; un `eps` que funciona para el disperso juntaría todo. Además, en muchas dimensiones las distancias entre puntos se parecen todas, y es difícil escoger `eps`.

#figure(
  caption: [Un grupo denso a la izquierda y uno disperso a la derecha. Con `eps` $= 0.3$, DBSCAN encuentra el grupo denso, pero marca como ruido la mayor parte del disperso.],
  fig-densidades(),
) <fig-densidades>

= Parte 3: Agrupamiento jerárquico

== Unir los grupos más cercanos

El *agrupamiento jerárquico aglomerativo* construye los grupos de abajo hacia arriba:

+ Empezar con cada ejemplo en su propio grupo: $n$ grupos.
+ Unir los dos grupos más cercanos.
+ Repetir el paso 2 hasta que quede un solo grupo.

El resultado no es un solo agrupamiento, sino toda la historia de uniones, que se dibuja como un *dendrograma* (@fig-dendrograma). Cada unión es una línea horizontal, y su altura es la distancia entre los dos grupos que se unieron. Para obtener grupos, se *corta* el dendrograma a una altura: los ejemplos que quedan conectados por debajo del corte forman un grupo. Cortar más arriba da menos grupos. Así, en lugar de escoger $k$ antes de agrupar, se puede mirar el dendrograma y cortar donde haya un salto grande de altura.

#figure(
  caption: [Siete puntos y su dendrograma. Primero se unen A y B, que son los más cercanos; después D y E; después C con A y B, y así hasta unir todo. Cortando a la altura de la línea roja quedan dos grupos.],
  fig-dendrograma(),
) <fig-dendrograma>

#ejercicio("Ejercicio 4")[
  Considerar los datos en una dimensión $x = 1, 2, 4, 8, 9$. La distancia entre dos grupos es la distancia entre sus dos puntos más cercanos.
  + Aplicar el agrupamiento aglomerativo: anotar qué grupos se unen en cada paso y a qué distancia.
  + Dibujar el dendrograma.
  + ¿A qué altura hay que cortar para obtener 2 grupos? ¿Y 3 grupos?
]

#prof[Solución. Uniones: {1} y {2} a distancia 1; {8} y {9} a distancia 1; {1, 2} y {4} a distancia 2; {1, 2, 4} y {8, 9} a distancia 4. Para 2 grupos, cortar entre 2 y 4 (por ejemplo, en 3): {1, 2, 4} y {8, 9}. Para 3 grupos, cortar entre 1 y 2: {1, 2}, {4} y {8, 9}.]

== Distancia entre grupos

En el paso 2 hay que decidir qué significa que dos grupos estén cerca. A esta decisión se le llama el *enlace* (linkage), y cambia mucho el resultado (@fig-enlace). Dos opciones:

- *Enlace single:* la distancia entre los dos puntos más cercanos de los grupos. Es la que se usó en el Ejercicio 4. Une grupos en cadena, punto a punto, así que puede encontrar grupos de cualquier forma, como las lunas. Pero si dos grupos están conectados por unos pocos puntos, los une.
- *Enlace de Ward:* une los dos grupos que hacen crecer menos la distorsión $J$ de K-means. Encuentra grupos redondos y de tamaño parecido, como K-means.

#figure(
  caption: [Agrupamiento jerárquico con dos grupos en las lunas. El enlace single encuentra las lunas; el de Ward las corta, como K-means.],
  fig-enlace(),
) <fig-enlace>

#codigo("jerarquico")
#salida("jerarquico")

Este método tiene que calcular la distancia entre todos los pares de ejemplos, así que es lento con muchos datos (con $n$ ejemplos hay del orden de $n^2$ pares).

= Parte 4: Densidad contra particional

K-means es un método *particional*: divide el espacio en $k$ regiones, una por centroide, y cada ejemplo va a la región de su centroide más cercano. DBSCAN es un método *basado en densidad*: un grupo es una zona con muchos ejemplos, y lo que no está en ninguna es ruido. La @fig-comparacion muestra los tres métodos en tres conjuntos de datos.

#figure(
  caption: [Los tres métodos sobre tres conjuntos de datos (con las features estandarizadas). K-means acierta con las bolas, pero falla con las lunas y los círculos. DBSCAN y el enlace single aciertan con las lunas y los círculos, pero juntan las dos bolas de arriba, que están muy cerca. Las ✕ grises son ruido.],
  fig-comparacion(),
) <fig-comparacion>

#align(center, table(
  columns: 4,
  align: (left, left, left, left),
  table.header([], [K-means], [DBSCAN], [Jerárquico]),
  [Hay que dar], [$k$], [`eps` y `min_samples`], [$k$, o dónde cortar],
  [Forma de los grupos], [redondos, de tamaño parecido], [cualquiera], [depende del enlace],
  [Ruido], [no: todo va a un grupo], [sí], [no],
  [Muchos datos], [rápido], [rápido], [lento],
  [Además da], [los centroides], [qué puntos son ruido], [el dendrograma],
))

En la práctica, no hay un método mejor que los demás. Conviene empezar con K-means, que es rápido y fácil de interpretar, graficar los resultados (o mirar la silueta) y probar DBSCAN o el jerárquico si los grupos no parecen redondos o hay valores atípicos.

#prof[Lab, Parte 3 (2:35–2:55): los tres algoritmos sobre los mismos datos.]

= Resumen

- En el *aprendizaje no supervisado* no hay etiquetas. El *agrupamiento* divide los datos en grupos de ejemplos parecidos.
- *K-means* repite dos pasos: asignar cada ejemplo a su centroide más cercano y mover cada centroide al promedio de su grupo. Cada paso baja la distorsión $J$, así que el algoritmo termina, pero puede quedarse en un mínimo local. Por eso se repite con varias inicializaciones (`n_init`) y se usa k-means++.
- Para escoger $k$ se usan el *método del codo* y la *silueta*. Antes de agrupar hay que estandarizar las features.
- *DBSCAN* define los grupos como regiones densas: con `eps` y `min_samples`, cada punto es núcleo, frontera o ruido. Encuentra grupos de cualquier forma y no necesita $k$, pero depende mucho de `eps` y falla con densidades distintas.
- El *agrupamiento jerárquico* une los grupos más cercanos, uno a la vez, y el resultado es un dendrograma que se corta para obtener grupos. El *enlace* decide qué es estar cerca: single encuentra grupos de cualquier forma y Ward se parece a K-means.
