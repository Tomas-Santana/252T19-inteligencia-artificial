#import "@preview/rubber-article:0.5.2": *

#show: article
#show link: set text(fill: rgb("#0000ee"))
#show link: underline
#maketitle(
  title: "Inteligencia Artificial. Tarea 5",
  authors: ("Tomas Santana",),
  date: "Noviembre, 2026",
)

Esta tarea es opcional. Quienes no la entreguen recibirán como nota el promedio de sus otras tareas. La tarea se realiza en parejas.

1. Datos y análisis exploratorio (8 puntos). Descargar un dataset de datos reales y realizar un análisis exploratorio de los datos, limpiándolos según sea necesario.

\

2. Clustering (12 puntos). Aplicar clustering sobre los datos para descubrir patrones o agrupaciones que les interesen. Utilizar `KMeans` y otro algoritmo de clustering de su elección. Comparar los resultados y justificar la elección del número de clusters (ya sea a través de los datos o usando el método del codo). Analizar los resultados y extraer conclusiones.

\

#heading(numbering: none, level: 1)[Entrega y evaluación]

Cada pareja debe entregar un notebook (`.ipynb`) con la solución de todos los puntos y el dataset utilizado (o el código que lo descarga). La entrega es *en clase*. El notebook debe ejecutarse de principio a fin sin errores. La nota definitiva se calculará como:

$
"Nota" = "Implementación" times "Explicabilidad",
$

donde la implementación es la suma de los puntos anteriores (de 0 a 20) y la explicabilidad es un factor entre 0 y 1 que mide la capacidad de presentar y justificar la solución propia. Para medirla, cada pareja responderá 3 preguntas sobre su implementación, con su código abierto y ejecutable. Cualquiera de los dos integrantes puede ser llamado a responder. El factor se asigna según la siguiente tabla:

#align(center)[
  #table(
    columns: 2,
    align: (center, left),
    [*Factor*], [*Criterio*],
    [1.0], [Responde correctamente las 3 preguntas y ubica en su código lo que se le pregunta sin dudar.],
    [0.8], [Responde 2 preguntas de forma completa y 1 de forma parcial.],
    [0.6], [Responde correctamente 2 preguntas.],
    [0.3], [Responde correctamente 1 pregunta.],
    [0], [No responde correctamente ninguna pregunta, o no logra ubicar en su propio código lo que se le pregunta.],
  )
]

Una respuesta es *correcta* si contiene todos los puntos esperados y *parcial* si contiene la idea principal pero le falta alguno de ellos.
