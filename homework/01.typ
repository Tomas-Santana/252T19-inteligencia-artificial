#import "@preview/rubber-article:0.5.2": *

#show: article
#show link: set text(fill: rgb("#0000ee"))
#show link: underline
#maketitle(
  title: "Inteligencia Artificial. Tarea 1",
  authors: ("Tomas Santana",),
  date: "September 26, 2026",
)

1. Descargar un dataset en #link("https://www.kaggle.com/datasets", "Kaggle.com") (o cualquier otra fuente) con al menos 500 ejemplos y al menos 2 características predictoras continuas y una variable objetivo continua. Realizar un análisis exploratorio de los datos, y limpiar los datos según sea necesario. Para ello, pueden utilizar `pandas`, `numpy`, y `matplotlib` o `seaborn`. Importante: el dataset a utilizar debe ser de datos reales, no generado artificialmente. La descripción del dataset usualmente especifica el origen de los datos.

\

2. Implementar en Python una clase que implemente la solución analítica de la regresión lineal multivariable, siguiendo la ecuación normal:
$
theta = (X^T X)^(-1) X^T y 
$
La clase debe tener los siguientes métodos:
- `fit(X, y)`: Ajusta el modelo a los datos de entrenamiento.
- `predict(X)`: Realiza predicciones sobre nuevos datos.
- `mse(y_true, y_pred)`: Calcula el error cuadrático medio entre los valores verdaderos y las predicciones.
- `r2_score(y_true, y_pred)`: Calcula el coeficiente de determinación R² entre los valores verdaderos y las predicciones. 

Les recomiendo ver #link("https://www.youtube.com/watch?v=bMccdk8EdGo", "este video de StatQuest sobre el coeficiente de determinación"). Su implementación puede estar basada en la función `r2_score` de `sklearn` Pueden leer también #link("https://www.youtube.com/watch?v=bMccdk8EdGo", "la documentación del método").

Como atributos de la clase, deben incluirse los parámetros del modelo $theta$ (usar `b_` siguiendo la convención de `sklearn`) y el intercepto $theta_0$ (usar `intercept_` siguiendo la convención de `sklearn`).

\ 
Ejemplo de uso de la clase:

```python
from my_linear_regression import MyLinearRegression
import numpy as np

# Crear datos de ejemplo
df = pd.read_csv("ruta_al_dataset.csv")
X = df[["feature1", "feature2"]].values
y = df["target"].values

# Crear instancia de la clase
model = MyLinearRegression()

# Entrenar el modelo
model.fit(X, y)

# Realizar predicciones
predictions = model.predict(X)
```

\

3. Utilizar la clase implementada para entrenar un modelo de regresión lineal multivariable con el dataset descargado en el punto 1. Entrenar con los mismos datos la clase `LinearRegression` de `sklearn` y comparar los resultados obtenidos con ambas implementaciones. Comparar las métricas de desempeño (MSE y R²). Comparar los valores de los parámetros (las propiedades `coef_` y `intercept_`) del modelo obtenidos con ambas implementaciones. 

\

4. Preguntas de análisis conceptual y discusión:

  4.1. Partición de datos y validación: Por qué es fundamental evaluar el modelo sobre un conjunto de prueba que no haya sido utilizado durante el ajuste (`fit`)? Explique qué riesgo se corre al reportar únicamente las métricas calculadas sobre los datos de entrenamiento y cómo se relaciona esto con el dilema sesgo-varianza.

  4.2. Diagnóstico de sesgo y varianza: Analice el desempeño obtenido ($R^2$ y MSE) tanto en el conjunto de entrenamiento como en el de prueba. ¿Considera que su modelo sufre de *alto sesgo* (underfitting) o *alta varianza* (overfitting)? Justifique su respuesta teniendo en cuenta la complejidad del modelo (lineal) frente a la naturaleza del dataset elegido.