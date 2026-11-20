#import "@preview/rubber-article:0.5.2": *

#show: article
#show link: set text(fill: rgb("#0000ee"))
#show link: underline
#set heading(numbering: none)
#maketitle(
  title: "Inteligencia Artificial. Proyecto final",
  authors: ("Tomas Santana",),
  date: "Noviembre, 2026",
)

El proyecto final se hace en grupos de 2 a 4 estudiantes. El objetivo es hacer algo interesante con machine learning que vaya bastante más allá de lo que se hizo en las tareas. Se pueden usar los algoritmos vistos en clase (modelos clásicos, redes neuronales, redes convolucionales) o técnicas que se construyen sobre ellos y que habrá que estudiar por cuenta propia.

La dedicación esperada es de unas 20 horas por persona.

= Requisitos mínimos

Sin importar el tema, todo proyecto debe incluir:

+ Una pregunta o problema claro: qué se quiere lograr y cómo saber si se logró.
+ Datos adecuados: un dataset distinto a los de las tareas, o uno conocido usado de una forma nueva. Explicar de dónde viene y cómo se preparó.
+ Una evaluación correcta: métricas apropiadas para el problema y una partición de datos sin fuga de información (entrenamiento, validación y prueba, o validación cruzada).
+ Análisis de errores: dónde falla el mejor modelo, con ejemplos concretos, y una hipótesis de por qué.
+ Conclusiones honestas: qué funcionó, qué no y qué se haría con más tiempo. Un resultado negativo bien analizado es válido.

Para proyectos de implementación desde cero (por ejemplo, una SVM propia) o de fine-tuning, debe haber una comparación contra una implementación de referencia o contra el modelo sin ajustar.

= Ideas de proyecto

Estas son solo ideas; se puede hacer otra cosa.

- SVM desde cero: una librería propia con kernels lineal y RBF que haga clasificación (SVC) y regresión (SVR), comparada con scikit-learn en precisión y tiempo.
- Clasificación de imágenes: binaria o multiclase, con un dataset propio o poco trabajado. Comparar una CNN entrenada desde cero contra transfer learning (por ejemplo, ResNet o EfficientNet), con data augmentation y visualización de lo que mira el modelo (Grad-CAM).
- Detección de objetos: usar YOLO u otra librería preentrenada y agregarle clases nuevas con imágenes etiquetadas por el grupo.
- Fine-tuning con LoRA: ajustar un modelo de lenguaje pequeño (o un modelo de visión) a una tarea específica usando LoRA, y comparar contra el modelo base y contra un fine-tuning tradicional o un modelo clásico.
- Modelos generativos: un autoencoder variacional o una GAN entrenados sobre Fashion-MNIST, KMNIST u otro dataset similar (no los dígitos), comparando la calidad de lo generado.
- Series de tiempo: pronóstico sobre varios datasets con varios algoritmos, comparando modelos clásicos o gradient boosting contra redes recurrentes (LSTM/GRU) o convoluciones 1D.
- Texto, embeddings y tokenización: implementar word2vec o BPE, o comparar TF-IDF con un modelo clásico contra una red neuronal y contra embeddings preentrenados en una tarea de clasificación.
- Aprendizaje por refuerzo: implementar Q-Learning para un juego sencillo y, si alcanza, extenderlo a Deep Q-Learning.
- Otra arquitectura: por ejemplo, un transformer pequeño desde cero que genere texto a nivel de caracteres.

Sobre la GPU: entrenar redes grandes probablemente requiere una GPU NVIDIA (CUDA). Si no se tiene una, usar Google Colab o Kaggle, que ofrecen GPU gratis por tiempo limitado, y ajustar el alcance a esos límites.

= Entregables

== Repositorio en GitHub

El README es el informe del proyecto, debe poder leerse de principio a fin y entender qué se hizo y qué se obtuvo, sin abrir el código. Debe incluir:

+ Descripción del problema y del dataset (con enlace para descargarlo).
+ Métodos.
+ Resultados bien presentados: tablas comparativas, gráficas (curvas de entrenamiento, matrices de confusión, ejemplos de predicciones) e imágenes embebidas en el README.
+ Análisis de errores y conclusiones.
+ Cómo instalar las dependencias y reproducir los resultados principales.

Además, el repositorio debe tener un archivo de dependencias (`requirements.txt` o equivalente) y el código y los notebooks ordenados, con resultados visibles. No subir datasets pesados.

== Presentación

12 minutos para explicar el problema, el trabajo realizado, la arquitectura del modelo, los resultados y lo aprendido, seguidos de 3 minutos de preguntas. Todos los integrantes deben estar presentes.
