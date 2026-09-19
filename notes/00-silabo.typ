#import "@preview/minimal-note:0.10.1": *

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
  title: [252T19. Inteligencia Artificial. Sílabo],
  author: [Tomas Santana],
  date: [], 
)

= Información del profesor
- Nombre: Tomas Santana
- Correo electrónico: tomas.santana.205635\@uru.edu
- Dudas y consultas: En clase o por correo electrónico. 

= Descripción del curso

Durante este curso, se diseñan, implementan y evalúan algoritmos de aprendizaje automático (ML) y aprendizaje profundo (DL) para resolver problemas a través de datos, aplicando técnicas de aprendizaje supervisado y no supervisado, y redes neuronales. Se estudian los fundamentos matemáticos que intervienen en la construcción de estos algoritmos de inteligencia artificial (IA). El curso aborda el proceso completo de modelado, desde la formulación del problema y la preparación de datos, hasta el entrenamiento, y la evaluación de modelos; e introduce aplicaciones modernas como los sistemas de recomendación y las bases de datos vectoriales, así como principios de ética e inteligencia artificial responsable. Es un curso teórico, pero con metodología de evaluación proyectiva.

= Unidades de aprendizaje

== Unidad 1: Fundamentos del aprendizaje automático

#table(
  columns: (0.5fr, 2fr, 1fr),
  align: (center, left, center),
  [*Semana*], [*Contenido*], [*Evaluación*],

  [1],
  [Introducción a la IA y al ML; tipos de aprendizaje . Matemáticas para ML I: vectores, álgebra lineal, transformaciones lineales.],
  [Participación.],

  [2],
  [Matemáticas para ML II: cálculo vectorial, gradiente, probabilidad. Flujo de trabajo del ML, generalización, sesgo-varianza e introducción a librerías.],
  [Participación.],
)

== Unidad 2: Aprendizaje supervisado

#table(
  columns: (0.5fr, 2fr, 1fr),
  align: (center, left, center),
  [*Semana*], [*Contenido*], [*Evaluación*],

  [3],
  [Regresión lineal: función de costo (MSE), mínimos cuadrados. Optimización: descenso de gradiente.],
  [Participación.
    Tarea 1.
  ],

  [4],
  [Ingeniería de características, regresión logística. Regularización (L1/L2) y sesgo-varianza: sobreajuste vs. subajuste.],
  [Participación.
  ],

  [5],
  [Máquinas de Vectores de Soporte (SVM): margen máximo y truco del kernel. Vecinos más próximos (kNN) y evaluación de modelos (métricas de evaluación).],
  [Participación. Tarea 2.],

  [6],
  [Árboles de decisión: criterios de división, interpretabilidad. Métodos por conjuntos (_ensemble methods_): embolsado (_bagging_), bosques aleatorios, boosting.],
  [Participación.],
)

== Unidad 3: Redes neuronales

#table(
  columns: (0.5fr, 2fr, 1fr),
  align: (center, left, center),
  [*Semana*], [*Contenido*], [*Evaluación*],

  [7],
  [Perceptrón: neurona artificial como clasificador lineal. Redes prealimentadas (FFNN) I: capas ocultas, funciones de activación, pase hacia adelante (_forward pass_).],
  [Participación.
    Tarea 3.
  ],

  [8],
  [Redes prealimentadas (FFNN) II: regla de la cadena, cálculo de gradientes. Entrenamiento en la práctica: inicialización, regularización, sobreajuste.],
  [Participación.],

  [9],
  [Redes convolucionales (CNN): convolución, filtros, agrupación, arquitectura.],
  [Participación.
    Tarea 4.],
)

== Unidad 4: Aprendizaje no supervisado y aplicaciones del ML

#table(
  columns: (0.5fr, 2fr, 1fr),
  align: (center, left, center),
  [*Semana*], [*Contenido*], [*Evaluación*],

  [10],
  [K-Means: elección de k, inicialización. DBSCAN y agrupamiento jerárquico: densidad vs. particional.],
  [Participación.],

  [11],
  [Reducción de dimensionalidad (PCA). Aplicaciones del aprendizaje automático: Bases de datos vectoriales, búsqueda semántica, RAG. Ética e IA responsable.],
  [Participación.
    Tarea 5.
  ],

  [12], [Presentación del proyecto final.], [Participación. Presentación de proyecto final.],
)

= Evaluación

Todas las semanas se asignará material de estudio para la siguiente clase, que podrá incluir lecturas, videos y ejercicios prácticos. Se evaluará la participación en clase en base al contenido de estudio asignado. La intervención dentro de la clase configura el 15% de la nota final. 

A partir de la primera clase, se asignarán tareas cortas que deberán ser entregadas dos semanas después de su asignación. El contenido de las tareas incluirá aquello que se haya visto en la clase en la que se asignó la tarea, la clase posterior a esta, y cualquier clase anterior. Estas actividades serán evaluadas en clase y constituyen el 50% de la nota final. 

El 25% de la nota final corresponde a la presentación del proyecto final, que consiste en un proyecto de desarrollo de un modelo de aprendizaje automático o aprendizaje profundo. Los requerimientos específicos del proyecto se discutirán en clase. La entrega final consistirá en una presentación oral del proyecto en la última semana del curso. 

El resto de la nota final (10%) corresponde a apreciación del profesor, e incluye aspectos como asistencia, presentación, participación, puntualidad, actitud, y proactividad.

== Distribución de la nota final

- Tareas 1-5: 50% (10% c/u)
- Participación en clase: 15%
- Proyecto final: 25%
- Apreciación del profesor: 10%

= Honestidad académica

Todas las actividades de los estudiantes deben ser originales. De ocurrir una falta o plagio se recibirá automáticamente la nota de cero en dicha evaluación, se citará a los responsables a la Coordinación de Pregrado y se elevará el informe respectivo a la Facultad con copia a la Dirección Académica. El uso de herramientas de IA generativa, cuando se permita, debe ser declarado explícitamente.

= Política de entregas

== Para trabajos y tareas entregadas virtualmente

- Las tareas deben ser entregadas en la fecha y hora establecida, con un periodo de gracia de 4 horas sin penalización.
- Se aplicará una penalización del 10% entre las 4 y 24 horas siguientes a la fecha de entrega.
- Se aplicará una penalización del 30% entre las 24 y 48 horas siguientes a la fecha de entrega.
- Pasado este periodo, no se aceptarán entregas tardías y la nota será cero.

== Para trabajos y tareas entregadas en clase

- Las tareas y trabajos deben ser entregadas durante la clase a la que fueron asignadas.

== Para exposiciones y presentaciones

- Las exposiciones y presentaciones deben realizarse en la fecha y hora establecida.

== Comodín

Los estudiantes tendrán derecho a retrasar la fecha de entrega de una única tarea o trabajo, siguiendo las siguientes normas:
- Para entregas virtuales, el comodín puede ser utilizado para retrasar la entrega de una tarea o trabajo hasta 48 horas después de la fecha de entrega original, sin penalización. No se extenderá la fecha límite de entrega más allá de las 48 horas posteriores a la fecha original.
- Para entregas en clase, el comodín puede ser utilizado para retrasar la entrega de una tarea o trabajo hasta la clase siguiente, sin penalización. No será posible utilizar el comodín para retrasar la entrega en la última clase del periodo académico.
- Para exposiciones y presentaciones, el comodín puede ser utilizado para retrasar la presentación hasta la clase siguiente. Para este caso, el uso debe ser aprobado previamente por el profesor.
- Para trabajos en grupo, podrá aplicarse siempre que al menos un integrante del equipo lo tenga disponible. Su uso consumirá el comodín de todos los miembros del grupo que aún lo conserven.


== Sobre retrasos justificados

En caso de retrasos en la entrega debido a causas justificables, el estudiante debe comunicarse previamente (o en la brevedad posible una vez superada la eventualidad) con el profesor por correo electrónico. El profesor evaluará la situación y podrá otorgar una extensión de tiempo para la entrega sin penalización, según corresponda.

= Bibliografía recomendada

- James, G., Witten, D., Hastie, T., y Tibshirani, R. (2021). An Introduction to Statistical Learning (2ª ed.). Springer. https://www.statlearning.com/. 

- Goodfellow, I., Bengio, Y., y Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org.


== Otros recursos en línea

- Andriy Burkov (2019). The Hundred-Page Machine Learning Book. https://themlbook.com.
- MIT 6.390 Staff. Intro to Machine Learning lecture Notes. https://introml.mit.edu/notes/

