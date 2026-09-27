#import "@preview/rubber-article:0.5.2": *

#show: article
#maketitle(
  title: "Inteligencia Artificial. Tarea 1",
  authors: ("Tomas Santana",),
  date: "September 26, 2026",
)



1. Sean los vectores:
$
y = mat(x_1 x_2; x_2^2; x_3 x_1)^T, quad a = mat(5; -2; 3)^T, quad x = mat(x_1; x_2; x_3)^T
$

Evalúa las siguientes derivadas matriciales mostrando todo el procedimiento analítico (en la notación de disposición de denominadores):

- $frac(partial, partial x) y$
- $frac(partial, partial x) a^T y$
- $frac(partial, partial x) x^T y$
- $frac(partial, partial x) y^T y$

\

2. Descargar un dataset en Kaggle (o cualquier otra fuente) con al menos 500 ejemplos y al menos 2 características predictoras continuas y una variable objetivo continua. Realizar un análisis exploratorio de los datos, y limpiar los datos según sea necesario. Para ello, pueden utilizar `pandas`, `numpy`, y `matplotlib` o `seaborn`.  

\

3. Implementar en Python una clase que implemente la solución analítica de la regresión lineal multivariable, siguiendo la ecuación normal:
$
theta = (X^T X)^(-1) X^T y 
$
La clase debe tener los siguientes métodos:
- `fit(X, y)`: Ajusta el modelo a los datos de entrenamiento.
- `predict(X)`: Realiza predicciones sobre nuevos datos.
- `mse(y_true, y_pred)`: Calcula el error cuadrático medio entre los valores verdaderos y las predicciones.
- `r2_score(y_true, y_pred)`: Calcula el coeficiente de determinación entre los valores verdaderos y las predicciones.
Como atributos de la clase, deben incluirse los parámetros del modelo $theta$ (usar `b_` siguiendo la convención de `sklearn`) y el intercepto $theta_0$ (usar `intercept_` siguiendo la convención de `sklearn`).

\

4. Utilizar la clase implementada para entrenar un modelo de regresión lineal multivariable con el dataset descargado en el punto 2. Entrenar con los mismos datos la clase `LinearRegression` de `sklearn` y comparar los resultados obtenidos con ambas implementaciones. Comparar las métricas de desempeño (MSE y R²).