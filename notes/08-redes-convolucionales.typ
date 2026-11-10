// OUTLINE (3 horas)
// 1. Repaso: backpropagation
// 1.1. La regla de la cadena, paso a paso (Ejercicio 1, en papel)
// 1.2. backward() en PyTorch
// 2. Redes como clases de PyTorch
// 2.1. nn.Module: __init__ y forward
// 2.2. Entrenar por mini-batches: épocas, mini-batches y pasos (Ejercicio 2)
// 2.3. Evaluar el modelo
// 3. Redes convolucionales
// 3.1. El problema de las imágenes
// 3.2. La convolución (Ejercicio 3, en papel)
// 3.3. Qué detecta un filtro
// 3.4. Varios filtros y canales
// 3.5. Padding y stride
// 3.6. Max pooling (Ejercicio 4, en papel)
// 3.7. Una red convolucional completa (Ejercicio 5, en papel)
// 3.8. Por qué funcionan mejor con imágenes
// 3.9. (Opcional) Reglas prácticas para diseñar una CNN
// 4. Sobreajuste y regularización
// 4.1. Las curvas de entrenamiento y validación
// 4.2. Dropout, model.train() y model.eval()
// 4.3. Weight decay y early stopping
//
// Hay dos versiones de estas notas, generadas desde este mismo archivo:
//   pública:      typst compile --root . notes/08-redes-convolucionales.typ
//   del profesor: typst compile --root . --input version=profesor notes/08-redes-convolucionales.typ notes/08-redes-convolucionales-profesor.pdf
// La versión del profesor muestra, en rojo y negrita, las indicaciones para la clase
// y las soluciones de los ejercicios.
//
// Las figuras están en notes/08-figuras.typ (las comparten las diapositivas).
// El código que se muestra está en notes/codigo/08/, y su salida se genera al ejecutarlo con
// `python notes/scripts/08-redes-convolucionales-figuras.py`, que también entrena las redes en MNIST.

#import "@preview/minimal-note:0.10.1": *
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "../lib/plot.typ": *
#import "../lib/op.typ": *
#import "08-figuras.typ": *

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
  title: if profesor [Redes convolucionales (versión del profesor)] else [Redes convolucionales],
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

#let r-mlp = d-mnist.resultados.mlp
#let r-cnn = d-mnist.resultados.cnn

#if profesor [
  #text(fill: red, weight: "bold", size: 12pt)[Plan de la clase (3 horas)]

  #prof[
    - *0:00–0:20 · Repaso.* Parte 1. Ejercicio 1 en papel. Proyectar `backward()` dando −32.
    - *0:20–0:45 · Clases y mini-batches.* Parte 2, en vivo con el código. Ejercicio 2.
    - *0:45–1:10 · Lab 1.* `MyMLP` en MNIST (labs/06).
    - *1:10–1:20 · Descanso.*
    - *1:20–2:05 · Redes convolucionales.* Secciones 3.1 a 3.7. Ejercicios 3, 4 y 5 en papel. Cada figura se compara con su código.
    - *2:05–2:45 · Lab 2.* `MyCNN` en MNIST. Al final, mostrar la sección 3.8 (resultados y filtros).
    - *2:45–3:00 · Cierre.* Parte 4 (10 min) y presentar la Tarea 4.
    - *Plan B (decidir a las 2:05).* Si van atrasados: cortar el Lab 2. Proyectar la salida de `formas.py` y la sección 3.8 (5 min), dar la Parte 4 y dejar el Lab 2 como actividad para la casa en Classroom: cada pareja pega la salida de su tabla de formas y su exactitud de validación, con una arquitectura distinta a las demás, y comenta la de otra pareja. Fecha límite antes de la entrega de la Tarea 4.
    - Lo que no se puede cortar: el trazado de formas de la sección 3.7 (Ejercicio 5). Es donde se van a trabar en la tarea.
  ]
]

= Parte 1: Repaso: backpropagation

#prof[Bloque 1 (0:00). La clase pasada no alcanzó para backpropagation completo. Aquí solo se repasa la idea; para la tarea no necesitan derivar nada, pero sí entender qué hace `backward()`.]

== La regla de la cadena

Recordemos la red más pequeña que se describió anteriormente: una unidad oculta con ReLU y una salida sin activación,

$
  z = w_1 x, quad quad a = "ReLU"(z), quad quad hat(y) = w_2 a, quad quad L = (hat(y) - y)^2,
$

con $x = 2$, $y = 10$, $w_1 = 1$ y $w_2 = 1$. El forward pass calcula los valores de izquierda a derecha: $z = 2$, $a = 2$, $hat(y) = 2$ y $L = 64$ (@fig-grafo, arriba).

Para el descenso por gradiente se necesitan $(partial L) / (partial w_1)$ y $(partial L) / (partial w_2)$. Backpropagation las calcula de derecha a izquierda (@fig-grafo, abajo). Se empieza con la derivada de la pérdida respecto a la predicción, $(partial L) / (partial hat(y)) = 2 (hat(y) - y) = -16$, a la que llamamos la señal de error. Después, en cada paso hacia atrás aplicamos la regla de la cadena:

- *Al cruzar una conexión*, la señal se multiplica por su peso: $(partial L) / (partial a) = (partial L) / (partial hat(y)) dot w_2 = -16$.
- *Al cruzar una activación*, se multiplica por su pendiente. La ReLU tiene pendiente 1 si $z > 0$ y 0 si $z < 0$; aquí $z = 2$, así que $(partial L) / (partial z) = -16 dot 1 = -16$.
- *La derivada de un peso* es la señal que llega multiplicada por la entrada que recibe ese peso: $(partial L) / (partial w_2) = -16 dot a = -32$ y $(partial L) / (partial w_1) = -16 dot x = -32$.

#figure(
  caption: [El cálculo de la red de dos pesos como una cadena de operaciones. Arriba, en negro, el forward pass. Abajo, en rojo, backpropagation: la derivada de la pérdida respecto a cada valor se calcula a partir de la del valor siguiente.],
  fig-grafo(),
) <fig-grafo>

Con una red de millones de pesos el procedimiento es el mismo: un forward pass guarda los valores intermedios, y un recorrido hacia atrás reutiliza la señal de cada capa para calcular la de la capa anterior.

#ejercicio("Ejercicio 1")[
  Para la misma red, con $x = 1$, $y = 4$, $w_1 = 2$ y $w_2 = 3$:
  + Calcular el forward pass: $z$, $a$, $hat(y)$ y $L$.
  + Calcular, de derecha a izquierda, $(partial L) / (partial hat(y))$, $(partial L) / (partial w_2)$, $(partial L) / (partial a)$, $(partial L) / (partial z)$ y $(partial L) / (partial w_1)$.
  + Dar un paso de descenso por gradiente con $alpha = 0.01$. ¿Los pesos suben o bajan? ¿Por qué tiene sentido, viendo $hat(y)$ y $y$?
]

#prof[Solución: $z = 2$, $a = 2$, $hat(y) = 6$, $L = 4$. Señal: $2(6 - 4) = 4$. $(partial L) / (partial w_2) = 4 dot 2 = 8$; $(partial L) / (partial a) = 4 dot 3 = 12$; $(partial L) / (partial z) = 12$ (ReLU encendida); $(partial L) / (partial w_1) = 12 dot 1 = 12$. Nuevos pesos: $w_1 = 1.88$, $w_2 = 2.92$. Bajan, porque la predicción (6) es más grande que la etiqueta (4). Con los pesos nuevos, $hat(y) = 5.49$ y $L = 2.22$: la pérdida bajó.]

== `backward()` en PyTorch

PyTorch hace exactamente este cálculo. Al crear un tensor con `requires_grad=True`, PyTorch registra las operaciones del forward pass; al llamar `L.backward()`, recorre esas operaciones hacia atrás y guarda en `.grad` la derivada de `L` respecto a cada tensor.

#codigo("backward")
#salida("backward")

#prof[Proyectar y señalar que da −32 y −32, lo mismo que la figura. Fin del repaso (0:20).]

= Parte 2: Redes como clases de PyTorch

== `nn.Module`: `__init__` y `forward`

Anteriormente se definieron redes utilizando `nn.Sequential`, que aplica una lista de capas en orden. Es cómodo para redes simples, pero en la práctica las redes se definen como una clase que extiende `nn.Module`. Esta clase tiene dos métodos especiales:

- *`__init__` declara las capas que tienen parámetros.* Cada capa se guarda como un atributo de `self` (por ejemplo, `self.hidden`). Así PyTorch las encuentra: `model.parameters()` recorre los pesos y sesgos de todas las capas guardadas en `self`, y eso es lo que le pasamos al optimizador. Lo primero que hace `__init__` es llamar `super().__init__()`, que prepara la clase para registrar esas capas.
- *`forward` describe el cálculo:* qué le pasa a la entrada `x`, en qué orden, hasta llegar a la salida. Aquí pueden aparecer también operaciones sin parámetros, como la ReLU o `flatten`, que no hace falta declarar en `__init__`.

Por ejemplo, esta es una red con una capa oculta para clasificar los dígitos de MNIST:

#codigo("mlp")
#salida("mlp")

Algunos detalles:

- La entrada es un mini-batch de imágenes con forma $(N, 1, 28, 28)$: $N$ imágenes, de 1 canal (escala de grises), de $28 times 28$ píxeles. Una capa densa espera un vector por ejemplo, así que `forward` primero aplana cada imagen a un vector de 784 números. Con `start_dim=1` se aplanan todas las dimensiones excepto la primera, evitando mezclar los ejemplos del mini-batch.
- La salida tiene 10 números por imagen, uno por dígito. Son los scores (logits), sin softmax: como se vio anteriormente, `nn.CrossEntropyLoss` aplica softmax internamente.
- Para usar la red se escribe `model(x)`, no `model.forward(x)`.
- La red tiene $784 dot 128 + 128 + 128 dot 10 + 10 = 101770$ parámetros, como confirma la última línea.

#prof[Escribir la clase en vivo. Insistir en los cuatro puntos: `super().__init__()`, capas en `__init__` como atributos de `self`, cálculo en `forward`, y llamar `modelo(x)`. Preguntar: "¿qué pasa si declaro la capa dentro de `forward`?" Se crearía una capa nueva, con pesos al azar, en cada llamada, y el optimizador no la conocería.]

== Entrenar por mini-batches: épocas, mini-batches y pasos

Anteriormente se entrenó la red con todos los datos a la vez. Con 50000 imágenes eso es lento y ocupa mucha memoria, así que se usa descenso por gradiente con mini-batches: en cada paso, el gradiente se calcula con un grupo pequeño de ejemplos.

- Un *mini-batch* es un grupo pequeño de ejemplos de entrenamiento, por ejemplo 64.
- Un *paso* es una actualización de los parámetros: forward pass, pérdida, `backward()` y `step()` sobre un mini-batch.
- Una *época* es una pasada completa por los datos de entrenamiento: tantos pasos como mini-batches caben en los datos.

Por ejemplo, con 50000 ejemplos y mini-batches de 64, una época tiene $ceil(50000 slash 64) = 782$ pasos (el último mini-batch tiene solo 16 ejemplos). En PyTorch, un `DataLoader` se encarga de mezclar los datos en cada época y entregarlos en mini-batches:

#codigo("entrenamiento", desde: 13)

#ejercicio("Ejercicio 2")[
  + ¿Cuántos pasos da `entrenar` con 50000 ejemplos, mini-batches de 64 y 3 épocas?
  + ¿Y si los mini-batches son de 128? ¿Cuántas veces ve la red cada ejemplo en los dos casos?
]

#prof[Solución: $782 dot 3 = 2346$ pasos. Con 128: $ceil(50000 slash 128) = 391$ pasos por época, $1173$ en total. En los dos casos la red ve cada ejemplo 3 veces (una por época); con mini-batches más grandes da menos pasos, cada uno con un gradiente menos ruidoso. Es el tema 5 del banco de preguntas de la Tarea 4.]

== Evaluar el modelo

Para medir la exactitud se utiliza esta función:

#codigo("entrenamiento", desde: 6, hasta: 10)

La clase predicha es la de mayor score (`argmax`). Hay dos detalles nuevos. `torch.no_grad()` le dice a PyTorch que no registre las operaciones, porque no se llamará a `backward()`; así la evaluación es más rápida. Y `model.eval()` pone la red en modo evaluación, mientras que `model.train()`, al principio de cada época, la pone en modo entrenamiento. Por ahora no cambia nada, pero algunas capas se comportan distinto al entrenar y al evaluar; lo que se verá en la Parte 4.

Entrenando `MyMLP` sobre #d-mnist.n_train imágenes de MNIST, con #d-mnist.n_val imágenes de validación, obtenemos:

#salida("entrenar-mlp")

#prof[Lab 1 (0:45–1:10): labs/06 (lab-06.zip). Escriben `MyMLP` en `modelos.py` y la verifican y entrenan desde `lab.ipynb` (Parte 1). Avisar: guardar `modelos.py` antes de volver a ejecutar la celda. Fin del Bloque 1, descanso (1:10).]

= Parte 3: Redes convolucionales

#prof[Bloque 2 (1:20).]

== El problema de las imágenes

La red `MyMLP` llega al #pct(r-mlp.prueba) de exactitud en los datos de prueba. Funciona, pero tiene dos problemas que se agravan con imágenes más grandes.

- *Demasiados parámetros.* Cada unidad de la primera capa está conectada con todos los píxeles. Con imágenes de $28 times 28$ en escala de grises son 784 pesos por unidad. Con una foto a color de $224 times 224$ píxeles, la entrada tiene $224 dot 224 dot 3 = 150528$ números, y una capa oculta de 1000 unidades ya tiene más de 150 millones de pesos.

- *Ignora la estructura de la imagen.* Al aplanar la imagen, la red no sabe qué píxeles son vecinos: para ella, los 784 píxeles son 784 features sin ningún orden. Además, un mismo patrón en dos lugares distintos de la imagen es, para la red, una entrada completamente distinta. La @fig-desplazado lo muestra: si desplazamos un 4 cuatro píxeles a la derecha, la distancia entre las dos imágenes, vistas como vectores de 784 píxeles, es más del doble que la distancia del 4 original al 9 más parecido de los datos. 

Esto significa que la red es muy sensible a pequeños desplazamientos o rotaciones.

#figure(
  caption: [Para un modelo que trata la imagen como un vector de 784 píxeles, el 4 desplazado #d-mnist.desplazamiento píxeles está mucho más lejos del original (distancia euclidiana entre los vectores de píxeles) que el #d-mnist.cercano_clase más parecido de los datos de entrenamiento.],
  fig-desplazado(),
) <fig-desplazado>

Las redes convolucionales (CNN) resuelven los dos problemas con una idea simple: en lugar de conectar cada unidad con toda la imagen, usan un filtro pequeño que recorre la imagen buscando el mismo patrón en todas las posiciones.

#prof[Mostrar la figura del 4 y preguntar: "¿Para la red, el 4 desplazado se parece al 4?" Que respondan antes de ver las distancias.]

== La convolución

Una convolución toma una imagen $X$ de $n times n$ y un filtro (o kernel) $K$ de $k times k$, más pequeño. El filtro se coloca sobre una ventana de $k times k$ de la imagen, se multiplican los valores que quedan uno encima del otro y se suman los productos. Ese número es una entrada de la salida. Después el filtro se desplaza una posición y se repite, hasta recorrer toda la imagen (@fig-conv). En fórmula, la entrada $(i, j)$ de la salida es

$
  y_(i j) = sum_(u = 1)^k sum_(v = 1)^k x_(i + u - 1, j + v - 1) K_(u v).
$

Como el filtro cabe en $n - k + 1$ posiciones en cada dirección, la salida es de $(n - k + 1) times (n - k + 1)$.

#figure(
  caption: [Una convolución de una entrada de $5 times 5$ con un filtro de $3 times 3$. La ventana resaltada produce la entrada resaltada de la salida: $#calculo-conv(X-conv, K-conv, (0, 2))$.],
  fig-conv(X-conv, K-conv, pos: (0, 2)),
) <fig-conv>

Por ejemplo, la entrada de la esquina superior izquierda de la salida es

$
  #calculo-conv(X-conv, K-conv, (0, 0)).
$

En PyTorch, una capa convolucional es `nn.Conv2d`. Se pueden comprobar los cálculos de la figura con este código:

#codigo("convolucion")
#salida("convolucion")

PyTorch representa las imágenes como tensores de cuatro dimensiones, $(N, C, H, W)$: el número de imágenes del mini-batch, el número de canales, el alto y el ancho. Nuestra imagen es un mini-batch de 1 imagen de 1 canal de $5 times 5$, es decir, $(1, 1, 5, 5)$, y la salida es $(1, 1, 3, 3)$. La matriz de la salida es la misma de la figura.

Una aclaración sobre nombres. En matemáticas y en procesamiento de señales, la convolución voltea el filtro (lo gira 180°) antes de recorrer la imagen. Dentro de una CNN esto no es necesario (técnicalmente, a esto se le llama correlación cruzada). Como los filtros se aprenden, voltearlos no cambia nada, y nadie lo hace.

#ejercicio("Ejercicio 3 (en papel)")[
  Calcular la convolución de la entrada $X$ con el filtro $K$:
  $
    X = mat(1, 2, 0, 1; 0, 1, 3, 1; 2, 0, 1, 0; 1, 1, 0, 2), quad quad K = mat(1, 0; 0, -1).
  $
  + ¿De qué tamaño es la salida? Calcular sus entradas.
  + Aplicar la ReLU a la salida.
  + Si $X$ fuera un tensor de PyTorch de forma $(1, 1, 4, 4)$, ¿qué forma tendría la salida de `nn.Conv2d(1, 1, kernel_size=2)`?
]

#prof[Solución: $3 times 3$. Cada entrada es $x_(i j) - x_(i+1, j+1)$: $mat(0, -1, -1; 0, 0, 3; 1, 0, -1)$. Con ReLU: $mat(0, 0, 0; 0, 0, 3; 1, 0, 0)$. Forma $(1, 1, 3, 3)$.]

== Qué detecta un filtro

¿Para qué sirve esta operación? Una entrada de la salida es grande cuando la ventana de la imagen se parece al filtro: los píxeles que coinciden con pesos positivos son altos y los que coinciden con pesos negativos son bajos. El filtro de la @fig-conv tiene $-1$ a la izquierda y $+1$ a la derecha, así que responde a ventanas con valores más altos a la derecha que a la izquierda: detecta *bordes verticales*. En la figura, la salida es positiva donde la franja de valores altos empieza y negativa donde termina.

La @fig-bordes aplica dos filtros así a un dígito real de MNIST, seguidos de una ReLU: el de bordes verticales y uno de bordes horizontales. Cada salida es una imagen nueva, llamada *mapa de activación* (feature map), que muestra dónde aparece el patrón del filtro.

#figure(
  caption: [Dos filtros de $3 times 3$ aplicados a un dígito de MNIST, seguidos de una ReLU. El primero detecta bordes verticales (el trazo empieza a la derecha de un fondo vacío) y el segundo, bordes horizontales (hay trazo arriba y no abajo). Las salidas son de $28 times 28$ porque se agregó padding (sección 3.5).],
  fig-bordes(),
) <fig-bordes>

Dentro de una CNN, los filtros se aprenden: los pesos se ajustan durante el entrenamiento para que cada filtro detecte un patrón útil para la tarea. 

== Varios filtros y canales

Un solo filtro detecta un solo patrón. Una capa convolucional usa varios filtros, cada uno con sus propios pesos, y produce un mapa de activación por filtro. Los mapas se apilan en un tensor: si la capa tiene 8 filtros, la salida tiene 8 *canales* (@fig-canales). Además, cada filtro tiene un sesgo.

#figure(
  caption: [Una capa convolucional con 8 filtros de $3 times 3$ sobre una imagen de $28 times 28$ en escala de grises (1 canal). Cada filtro produce un mapa de $26 times 26$, y la salida apila los 8 mapas.],
  fig-canales(1, 28, 8, 3, 26),
) <fig-canales>

#codigo("canales")
#salida("canales")

`nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)` crea 8 filtros. Sus pesos se guardan en un tensor de forma $(8, 1, 3, 3)$: 8 filtros de $1 times 3 times 3$. La salida de un mini-batch de 64 imágenes es $(64, 8, 26, 26)$.

*Entradas con varios canales.* ¿Qué pasa si la entrada ya tiene varios canales, por ejemplo los 8 mapas de la capa anterior, o los 3 canales (RGB) de una foto a color? Entonces cada filtro tiene un canal por cada canal de la entrada: es un bloque de $C_"in" times k times k$. El filtro se coloca sobre una ventana de $k times k$ de *todos* los canales a la vez, multiplica y suma los $C_"in" k^2$ productos, y produce un solo número. Cada filtro sigue produciendo un solo mapa, sin importar cuántos canales tenga la entrada (@fig-canales2).

#figure(
  caption: [Una capa con 16 filtros sobre una entrada de 8 canales. Cada filtro es un bloque de $8 times 3 times 3$ que combina los 8 canales de la entrada en un solo mapa.],
  fig-canales(8, 14, 16, 3, 12),
) <fig-canales2>

#codigo("canales2")
#salida("canales2")

*Número de parámetros.* Una capa con $C_"in"$ canales de entrada, $C_"out"$ filtros de $k times k$ tiene $C_"out"$ filtros de $C_"in" k^2$ pesos, más un sesgo por filtro:

$
  (C_"in" k^2 + 1) C_"out" "parámetros".
$

Notar que el número de parámetros *no depende del tamaño de la imagen*: el mismo filtro se usa en todas las posiciones.

#prof[Este es el tema 2 y 3 del banco de preguntas. Hacer el conteo en la pizarra antes de mostrar la salida del código.]

== Padding y stride

Cada convolución hace la imagen más pequeña: con un filtro de $3 times 3$, una imagen de $28 times 28$ pasa a $26 times 26$. Además, los píxeles del borde participan en menos ventanas que los del centro. Para evitarlo se agrega *padding*: un borde de ceros alrededor de la entrada. 

Por ejemplo, con un padding de 1 y un filtro de $3 times 3$, la salida tiene el mismo tamaño que la entrada (@fig-padding).

#figure(
  caption: [La misma convolución de la @fig-conv, con padding de 1: la entrada se rodea de ceros (en gris) y la salida queda de $5 times 5$, igual que la entrada original.],
  fig-conv(X-conv, K-conv, pos: (0, 0), padding: 1, s: 0.5, tam: 9.5pt),
) <fig-padding>

El *stride* es cuántas posiciones avanza el filtro en cada paso. Con stride 1 recorre todas las posiciones; con stride 2 salta una de cada dos, y la salida queda de la mitad del tamaño.

En general, con una entrada de $n times n$, un filtro de $k times k$, padding $p$ y stride $s$, la salida es de $m times m$ con

$
  m = floor((n + 2 p - k) / s) + 1.
$

Por ejemplo, para $n = 28$ y $k = 3$: sin padding, $m = 26$; con $p = 1$, $m = 28$; con $p = 1$ y $s = 2$, $m = floor(27 slash 2) + 1 = 14$. Lo comprobamos:

#codigo("padding")
#salida("padding")

== Max pooling

Después de una convolución se suele agregar una capa de *pooling*, que tiene dos efectos útiles. Primero, reduce el tamaño de los mapas, y con eso el número de parámetros de las capas siguientes. Segundo, hace a la red un poco más tolerante a desplazamientos pequeños: si el filtro anterior detecta un borde, el pooling conserva esa información aunque pierda su posición exacta.

La más común es el *max pooling* de $2 times 2$: divide cada mapa en bloques de $2 times 2$ y se queda con el máximo de cada bloque (@fig-pooling). Así un mapa de $28 times 28$ pasa a $14 times 14$. 

#figure(
  caption: [Max pooling de $2 times 2$: cada bloque de color de la entrada se reduce a su máximo.],
  fig-pooling(X-pool),
) <fig-pooling>

El pooling se aplica a cada canal por separado, así que no cambia el número de canales.

#codigo("pooling")
#salida("pooling")

#ejercicio("Ejercicio 4")[
  + Aplicar max pooling de $2 times 2$ a la matriz
    $
      Z = mat(2, 0, 1, 5; 3, 1, 0, 2; 0, 4, 2, 2; 1, 1, 3, 0).
    $
  + Un tensor de forma $(64, 16, 14, 14)$ pasa por `nn.MaxPool2d(2)`. ¿Qué forma tiene la salida? ¿Cuántos parámetros tiene la capa?
]

#prof[Solución: $mat(3, 5; 4, 3)$. Forma $(64, 16, 7, 7)$; 0 parámetros.]

== Una red convolucional completa

Una CNN típica repite varias veces un *bloque* de convolución, ReLU y pooling. Cada bloque detecta patrones a partir de los mapas del bloque anterior: los primeros detectan bordes, y los siguientes combinan bordes en patrones más grandes. Mientras tanto, los mapas se hacen más pequeños y el número de canales crece. Al final, los mapas se aplanan en un vector y pasan por una o más capas densas, que hacen la clasificación.

La @fig-arquitectura muestra la red que vamos a usar con MNIST, con dos bloques y una capa densa.

#figure(
  caption: [La arquitectura de `MyCNN`. Debajo de cada tensor, su forma (canales × alto × ancho), sin contar la dimensión del mini-batch.],
  ajustar(fig-arquitectura()),
) <fig-arquitectura>

#codigo("cnn")

Como siempre, `__init__` declara las capas con parámetros y `forward` describe el cálculo. La misma capa `self.pool` se usa dos veces, porque no tiene parámetros. El número más difícil de este código es el `16 * 7 * 7` de la capa densa: es el número de entradas que recibe después de aplanar, y hay que calcularlo siguiendo las formas desde la entrada:

#align(center, table(
  columns: 3,
  align: (left, center, right),
  table.header([Capa], [Forma de la salida], [Parámetros]),
  [entrada], $1 times 28 times 28$, [—],
  [conv1: `Conv2d(1, 8, 3, padding=1)` + ReLU], $8 times 28 times 28$, $(1 dot 9 + 1) dot 8 = 80$,
  [pool: `MaxPool2d(2)`], $8 times 14 times 14$, $0$,
  [conv2: `Conv2d(8, 16, 3, padding=1)` + ReLU], $16 times 14 times 14$, $(8 dot 9 + 1) dot 16 = 1168$,
  [pool: `MaxPool2d(2)`], $16 times 7 times 7$, $0$,
  [aplanar], $784$, $0$,
  [densa: `Linear(784, 10)`], $10$, $784 dot 10 + 10 = 7850$,
  [*Total*], [], [*9098*],
))

Si el número de entradas de la capa densa está mal, PyTorch da un error de dimensiones en el primer forward pass. La forma más segura de verificarlo es hacer un *forward de prueba*: pasar una imagen inventada por la red, capa por capa, e imprimir la forma después de cada una. Y para contar los parámetros de cada capa se usa `p.numel()`:

#codigo("formas")
#salida("formas")

#ejercicio("Ejercicio 5")[
  Considerar esta red para imágenes de $1 times 28 times 28$, sin padding: `Conv2d(1, 6, kernel_size=5)`, ReLU, `MaxPool2d(2)`, `Conv2d(6, 16, kernel_size=5)`, ReLU, `MaxPool2d(2)`, aplanar y `Linear(?, 10)`.
  + Calcular la forma de la salida de cada capa.
  + ¿Cuántas entradas tiene la capa densa?
  + ¿Cuántos parámetros tiene cada capa? ¿Y la red completa?
]

#prof[Solución: $6 times 24 times 24 -> 6 times 12 times 12 -> 16 times 8 times 8 -> 16 times 4 times 4 -> 256 -> 10$. La densa es `Linear(256, 10)`. Parámetros: $(25 + 1) dot 6 = 156$; $(6 dot 25 + 1) dot 16 = 2416$; $256 dot 10 + 10 = 2570$; total 5142. Es (casi) LeNet-5, la red de Yann LeCun para leer códigos postales y cheques (1998). Este ejercicio no se puede cortar: es donde se van a trabar en la tarea.]

#prof[Lab 2 (2:05–2:45): Parte 2 de `lab.ipynb`. Primero el Ejercicio 2a en papel (tabla de formas), después escriben `MyCNN` en `modelos.py`. Plan B: la Parte 3 del notebook (`MyCNNVariante` y el resumen para Classroom) es la actividad para la casa.]

== Por qué funcionan mejor con imágenes

Entrenando `MyCNN` con el mismo código y los mismos datos que `MyMLP`:

#salida("entrenar-cnn")

#align(center, table(
  columns: 4,
  align: (left, right, center, center),
  table.header([Red], [Parámetros], [Exactitud de validación], [Exactitud de prueba]),
  [`MyMLP`], [#miles(r-mlp.parametros)], [#pct(r-mlp.validacion)], [#pct(r-mlp.prueba)],
  [`MyCNN`], [#miles(r-cnn.parametros)], [#pct(r-cnn.validacion)], [#pct(r-cnn.prueba)],
))

La CNN es más precisa y tiene once veces menos parámetros. La razón está en dos propiedades de la convolución:

- *Conexiones locales.* Cada entrada de un mapa depende solo de una ventana pequeña de la imagen, no de todos los píxeles. Los patrones de una imagen (bordes, esquinas, trazos) son locales, así que no se pierde nada.
- *Pesos compartidos.* El mismo filtro se usa en todas las posiciones de la imagen. Un detector de bordes que funciona arriba a la izquierda también funciona abajo a la derecha.

La @fig-filtros muestra los 8 filtros de la primera capa de `MyCNN` después de entrenar, y sus mapas de activación para un dígito de prueba. Varios se parecen a los detectores de bordes de la @fig-bordes.

#figure(
  caption: [Arriba, los 8 filtros de $3 times 3$ de la primera capa de `MyCNN` después de entrenar (oscuro: peso positivo; claro: negativo). Abajo, el mapa de activación de cada filtro (después de la ReLU) para un 7 de los datos de prueba.],
  fig-filtros(),
) <fig-filtros>

#prof[Mostrar esta sección al terminar el Lab 2 (o en el plan B). Los filtros son el último punto de la Tarea 4: que vean que se grafican con `modelo.conv1.weight`.]

== (Opcional) Reglas prácticas para diseñar una CNN

#prof[OPCIONAL, lectura para la casa. No darlo en clase. Mencionarlo al presentar la Tarea 4: les sirve para escoger las variantes del punto 3.]

¿Cuántas capas, cuántos filtros, de qué tamaño? No hay una fórmula que dé la mejor arquitectura para un problema, y en la práctica se prueban varias y se comparan con los datos de validación. Pero la mayoría de las CNN siguen el mismo patrón general:

$
  "entrada" -> [ ["conv" -> "ReLU"] times N -> "pool" ] times M -> ["densa" -> "ReLU"] times K -> "densa".
$

Es decir, $M$ bloques, cada uno con $N$ convoluciones (con su ReLU) seguidas de un pooling, y al final $K$ capas densas ocultas antes de la capa de salida. Lo usual es $N$ entre 1 y 3 y $K$ entre 0 y 2; $M$ depende del tamaño de las imágenes. Nuestra `MyCNN` tiene $N = 1$, $M = 2$ y $K = 0$.

Dentro de ese patrón, estas reglas funcionan bien como punto de partida:

- *Filtros pequeños.* Usar filtros de $3 times 3$ (a lo sumo $5 times 5$) con stride 1. Para detectar patrones grandes es mejor apilar varias capas de filtros pequeños que usar un filtro grande: cada capa combina los patrones de la anterior, y entre capa y capa hay una ReLU.
- *Padding para conservar el tamaño.* Con un filtro de $k times k$ y stride 1, un padding de $p = (k - 1) slash 2$ deja la salida del mismo tamaño que la entrada: $p = 1$ para $k = 3$ y $p = 2$ para $k = 5$ (se puede comprobar con la fórmula de la sección 3.5). Así las convoluciones no cambian el tamaño, solo el pooling lo hace, y es mucho más fácil seguir las formas.
- *Pooling de $2 times 2$.* Cada pooling divide el alto y el ancho entre 2. Conviene que el tamaño de las imágenes se pueda dividir entre 2 varias veces. Con MNIST, dos poolings llevan de $28 times 28$ a $7 times 7$; un tercero daría $3 times 3$, porque PyTorch descarta la última fila y la última columna cuando el tamaño es impar.
- *Más filtros a medida que la imagen se achica.* Lo usual es duplicar el número de filtros después de cada pooling, como en `MyCNN` (8 y luego 16). Los mapas son cada vez más pequeños, así que podemos pagar más canales, y las capas profundas necesitan más filtros porque combinan los patrones de las anteriores de muchas formas distintas.
- *Cuidado con la primera capa densa.* Casi todos los parámetros de una CNN suelen estar en la primera capa densa, que recibe todos los mapas aplanados. En `MyCNN`, 7850 de los 9098 parámetros (el 86%) están en la capa densa. Por eso, agregar un bloque más (que achica los mapas) puede *reducir* el número de parámetros, mientras que agregar una capa densa oculta lo multiplica.

La tabla muestra el efecto de algunos cambios a `MyCNN`. Todas las convoluciones usan padding $p = (k - 1) slash 2$.

#align(center, table(
  columns: 4,
  align: (left, center, right, right),
  table.header([Variante], [Forma antes de aplanar], [Parámetros], [En capas densas]),
  [`MyCNN` (8 y 16 filtros de $3 times 3$)], $16 times 7 times 7$, [9098], [7850],
  [Más filtros (16 y 32)], $32 times 7 times 7$, [20490], [15690],
  [Filtros de $5 times 5$], $16 times 7 times 7$, [11274], [7850],
  [Un tercer bloque (8, 16 y 32 filtros)], $32 times 3 times 3$, [8778], [2890],
  [Una capa densa oculta de 64 unidades], $16 times 7 times 7$, [52138], [50890],
))

Por último, al comparar variantes conviene *cambiar una sola cosa a la vez*. Si se cambian el número de filtros y el tamaño del filtro al mismo tiempo y la exactitud mejora, no hay forma de saber cuál de los dos cambios ayudó. Y la comparación se hace con los datos de validación: los de prueba se usan una sola vez, con el modelo escogido al final.

= Parte 4: Sobreajuste y regularización

#prof[Cierre (2:45). 10 minutos. Lo importante para la tarea: leer las curvas, qué hace dropout y la diferencia entre `train()` y `eval()`.]

== Las curvas de entrenamiento y validación

Una red con muchos parámetros puede memorizar los datos de entrenamiento, igual que un árbol muy profundo. Se nota en las curvas de la pérdida por época: la de entrenamiento sigue bajando, pero la de validación deja de bajar y empieza a subir. Desde ese momento, la red está aprendiendo detalles de los datos de entrenamiento que no se repiten en datos nuevos.

La @fig-sobreajuste lo muestra con una red densa grande (dos capas ocultas de 512 unidades, casi 670 mil parámetros) entrenada con solo 1000 imágenes de MNIST. Para evitar el sobreajuste, se usan más datos, modelos más pequeños, o técnicas de regularización.

#figure(
  caption: [Pérdida de entrenamiento y de validación de una red densa grande entrenada con 1000 imágenes. La pérdida de entrenamiento llega casi a cero, mientras que la de validación baja hasta la época #(d-mnist.sobreajuste.sin.val.sorted(key: p => p.at(1)).at(0).at(0)) y luego sube: la red sobreajusta. Con dropout (en verde), la pérdida de validación llega más abajo y sube menos.],
  [
    #fig-sobreajuste()
    #v(0.2cm)
    #leyenda-sobreajuste()
  ],
) <fig-sobreajuste>

== Dropout

*Dropout* es una técnica de regularización específica de las ANN. En cada paso de entrenamiento, cada unidad de una capa se "apaga" (su activación se pone en cero) con probabilidad $p$, por ejemplo $p = 0.5$, escogiendo al azar unidades distintas en cada paso. Como la red no puede contar con que una unidad en particular esté presente, no puede depender demasiado de ninguna, y aprende features más robustas. Para que la suma de las activaciones no cambie en promedio, las unidades que quedan encendidas se multiplican por $1 slash (1 - p)$.

Al evaluar, en cambio, queremos usar la red completa, sin azar. Aquí es donde importan `model.train()` y `model.eval()`: en modo entrenamiento, `nn.Dropout` apaga unidades; en modo evaluación, no hace nada.

#codigo("dropout")
#salida("dropout")

Con $p = 0.5$, en modo entrenamiento la mitad de las unidades (en promedio) se apagan y las demás se multiplican por 2. En modo evaluación la entrada pasa sin cambios. En una red, la capa de dropout se declara en `__init__` (`self.dropout = nn.Dropout(0.5)`) y se aplica en `forward`, normalmente después de la ReLU de las capas densas.

== Weight decay y early stopping

*Weight decay* es regularización L2. En PyTorch se activa con un argumento del optimizador, por ejemplo `torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)`. 

Esto equivale a sumar a la pérdida una penalización $lambda / 2 norm(theta)^2$ con $lambda = 10^(-4)$.

*Early stopping* (parada temprana) consiste en vigilar la pérdida de validación durante el entrenamiento y quedarse con los parámetros de la época en la que fue mínima, en lugar de los de la última época. Para implementarlo, en cada época se compara la pérdida de validación con la mejor hasta ahora y, si mejoró, se guarda una copia de los parámetros, por ejemplo con `torch.save(model.state_dict(), "mejor.pt")`.

#prof[Presentar la Tarea 4 (2:55). Recordar que la entrega es en clase y que se evalúa la explicabilidad: cada tema del banco aparece en estas notas.]