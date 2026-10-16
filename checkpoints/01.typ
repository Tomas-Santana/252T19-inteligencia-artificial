#import "@preview/rubber-article:0.5.2": *

#show: article
#show link: set text(fill: rgb("#0000ee"))
#show link: underline
#maketitle(
  title: "Inteligencia Artificial. Guía de estudio 1",
  authors: ("Tomas Santana",),
  date: "Octubre 2026",
)

= Guía de estudio: clases 1 a 4

Esta guía reúne preguntas sobre los conceptos fundamentales estudiados en las primeras cuatro clases. Su propósito es orientar el estudio personal y ayudar a relacionar los conceptos vistos en clase. No es una actividad evaluada ni requiere entrega. Se recomienda intentar responder las preguntas con palabras propias, explicando el significado de los conceptos y de los símbolos matemáticos cuando corresponda.

== Parte 1. Fundamentos matemáticos

#enum(start: 1,
  [¿Qué es un vector y qué es una matriz? Explica cómo utilizarías un vector para representar las características de una casa y cómo organizarías los datos de varias casas en una matriz.],
  [¿Qué es el producto punto de dos vectores y qué tipo de resultado produce? Explica cómo aparece esta operación al calcular una predicción mediante $theta^T x$, y como se relaciona con la ecuación de la recta.],
  [¿Qué representa la derivada de una función en un punto? ¿Qué información proporciona el signo de la derivada y cómo puede ayudarnos a encontrar el mínimo de una función?],
)

== Parte 2. Aprendizaje supervisado y regresión lineal

#enum(start: 4,
  [¿Qué significa aprendizaje supervisado? Explica qué son una característica (feature) y una etiqueta (label), y cuál es la diferencia entre un problema de regresión y uno de clasificación. Da un ejemplo de cada uno.],
  [En un conjunto de entrenamiento de pares $(x^((i)), y^((i)))$, ¿qué representan $x^((i))$, $y^((i))$ y el superíndice $(i)$? ¿Qué diferencia existe entre el valor real $y^((i))$ y la predicción $h_theta(x^((i)))$?],
  [Considera el modelo $h_theta(x) = theta_0 + theta_1 x$. ¿Qué representan $x$, $h_theta(x)$, $theta_0$ y $theta_1$? Explica qué ocurriría con las predicciones si modificáramos uno de los parámetros, manteniendo el mismo valor de entrada.],
  [¿Qué significa entrenar un modelo de regresión lineal? ¿Cuáles son los valores que intentamos encontrar durante el entrenamiento y cuáles son los datos que mantenemos fijos?],
  [¿Qué es una función de costo y para qué se utiliza? Explica qué mide el error cuadrático y por qué buscamos parámetros que reduzcan ese error.],
  [¿En qué consiste el descenso por gradiente? Explica qué papel desempeñan el gradiente y la tasa de aprendizaje $alpha$, y por qué este procedimiento permite ajustar los parámetros de un modelo.],
  [¿Qué son las ecuaciones normales en regresión lineal? Explica qué problema resuelven y cuál es su principal diferencia con el descenso por gradiente.],
)

== Parte 3. Generalización, selección de modelos y regularización

#enum(start: 11,
  [¿Qué significa que un modelo generalice correctamente? Explica por qué obtener un error muy bajo sobre los datos de entrenamiento no garantiza que el modelo funcione bien con datos nuevos.],
  [¿Para qué sirven los conjuntos de entrenamiento, validación y prueba? Explica qué decisiones deberíamos tomar con cada uno y por qué no conviene utilizar el conjunto de prueba para escoger un modelo.],
  [¿Qué son el sobreajuste (overfitting) y el subajuste (underfitting)? Explica cómo se relacionan con la complejidad de un modelo y con sus errores de entrenamiento y validación.],
  [¿Qué es la regularización y qué problema busca resolver? Explica la diferencia principal entre Ridge ($L_2$) y Lasso ($L_1$), qué controla el hiperparámetro $lambda$ y en qué se diferencia un hiperparámetro de un parámetro aprendido.],
)

== Parte 4. Clasificación y regresión logística

#enum(start: 15,
  [¿Por qué no siempre resulta conveniente usar regresión lineal para resolver un problema de clasificación binaria? Explica qué tipo de salida queremos obtener y qué dificultades puede presentar una predicción lineal sin restricciones.],
  [En un clasificador lineal, ¿qué representa el score $theta^T x$? ¿Qué es la frontera de decisión y cómo determina el modelo a qué clase asignar una observación?],
  [¿Qué es la función sigmoide y para qué se utiliza en regresión logística? Explica por qué aplicarla al score permite interpretar la salida como una probabilidad.],
  [Si un modelo de regresión logística devuelve $h_theta(x) = 0.8$, ¿cómo interpretas ese resultado? ¿Qué clase predecirías utilizando un umbral de $0.5$? Explica también, en términos generales, qué buscamos al entrenar este modelo mediante máxima verosimilitud.],
)