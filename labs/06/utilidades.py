"""Clase 8, laboratorio: funciones de apoyo. No hace falta modificar este archivo.

- load_mnist: lee MNIST desde la carpeta data/ (no necesita internet).
- train, accuracy: el ciclo de entrenamiento por mini-batches de las notas.
  En cada mini-batch, train llama a train_step de modelos.py.
- check_train_step: verifica train_step.
- shape_table: un forward de prueba que imprime la forma de la salida y los
  parámetros de cada capa.
- plot_histories, plot_filters, show_predictions: gráficas.
- summary: un resumen de la red para compartir.
"""

import copy
import gzip
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).parent / "data"


def verify(condition, success_message, error_message):
    print(("✅ " + success_message) if condition else ("❌ " + error_message))


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def _read(filename, offset):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. La carpeta data/ debe estar junto a este archivo, "
            "con los cuatro archivos .gz de MNIST."
        )
    return np.frombuffer(gzip.open(path).read(), np.uint8, offset=offset)


def load_mnist(validation_size=10000, seed=0):
    """Devuelve X_train, y_train, X_val, y_val, X_test, y_test.

    Las imágenes son tensores de forma (N, 1, 28, 28) con valores entre 0 y 1,
    y las etiquetas, tensores de forma (N,) con el dígito (0 a 9).
    Las 60000 imágenes de entrenamiento de MNIST se separan al azar en
    entrenamiento y validación. La prueba son las 10000 imágenes de prueba de MNIST.
    """
    X = _read("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
    y = _read("train-labels-idx1-ubyte.gz", 8)
    X_test = _read("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
    y_test = _read("t10k-labels-idx1-ubyte.gz", 8)

    order = np.random.default_rng(seed).permutation(len(y))
    i_val, i_train = order[:validation_size], order[validation_size:]

    def to_tensors(X, y):
        return torch.tensor(X / 255, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    return (*to_tensors(X[i_train], y[i_train]), *to_tensors(X[i_val], y[i_val]), *to_tensors(X_test, y_test))


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _validate_model(model, name="la red"):
    """Revisa que la red tenga capas y un forward completo antes de usarla."""
    if model is None:
        print("❌ Primero hay que completar y entrenar la red de la celda anterior.")
        return False
    if count_parameters(model) == 0:
        print(f"❌ {name} no tiene parámetros: falta declarar sus capas en __init__ (en modelos.py).")
        return False
    try:
        with torch.no_grad():
            model(torch.rand(2, 1, 28, 28))
    except NotImplementedError as e:
        print(f"❌ {e}")
        return False
    except RuntimeError as e:
        print(f"❌ Error en el forward: {e}\n   Usar shape_table para encontrar la capa con el problema.")
        return False
    return True


def accuracy(model, X, y):
    model.eval()                 # modo evaluación
    with torch.no_grad():         # sin registrar operaciones
        prediction = model(X).argmax(dim=1)
    return (prediction == y).float().mean().item()


def train(model, X_train, y_train, X_val, y_val, num_epochs=3, lr=0.1, batch_size=64, seed=0):
    """Entrena con descenso por gradiente con mini-batches, como en las notas.
    Cada paso lo da la función train_step de modelos.py.

    Devuelve un diccionario con la pérdida de entrenamiento promedio y la
    exactitud de validación de cada época.
    """
    import modelos  # aquí y no arriba, para usar siempre la versión más reciente de train_step

    if not _validate_model(model):
        return None
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {"loss": [], "val_accuracy": [], "steps": 0, "num_epochs": num_epochs, "batch_size": batch_size}
    start = time.time()
    for epoch in range(num_epochs):
        model.train()            # modo entrenamiento
        total = 0.0
        for X_batch, y_batch in loader:      # un paso por mini-batch
            try:
                L = modelos.train_step(model, X_batch, y_batch, loss_fn, optimizer)
            except NotImplementedError as e:
                print(f"❌ {e}")
                return None
            total += float(L) * len(y_batch)
            history["steps"] += 1
        history["loss"].append(total / len(y_train))
        history["val_accuracy"].append(accuracy(model, X_val, y_val))
        print(
            f"época {epoch + 1}/{num_epochs}: pérdida {history['loss'][-1]:.3f}, "
            f"exactitud de validación {history['val_accuracy'][-1]:.3f} "
            f"({time.time() - start:.0f} s)"
        )
    history["seconds"] = time.time() - start
    print(f"{history['steps']} pasos en total ({len(loader)} por época)")
    return history


def check_train_step(train_step, model, X, y):
    """Prueba train_step con unos pasos sobre un mini-batch (X, y)."""
    if not _validate_model(model):
        return
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    try:
        L = train_step(model, X, y, loss_fn, optimizer)
    except NotImplementedError as e:
        print(f"❌ {e}")
        return
    if not isinstance(L, float):
        print(f"❌ train_step debería devolver la pérdida como float (con .item()), y devuelve {type(L).__name__}")
        return
    print("✅ devuelve la pérdida como float")

    changed = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    if not changed:
        print("❌ los parámetros no cambiaron: ¿faltan loss.backward() u optimizer.step()?")
        return
    print("✅ los parámetros cambian: el optimizador da el paso")

    # Si falta zero_grad, los gradientes de este paso se suman a los del anterior.
    reference = copy.deepcopy(model)
    reference.zero_grad()
    loss_fn(reference(X), y).backward()
    train_step(model, X, y, loss_fn, optimizer)
    fresh = all(
        torch.allclose(p.grad, r.grad, atol=1e-6)
        for p, r in zip(model.parameters(), reference.parameters())
    )
    verify(fresh, "los gradientes se borran antes de cada paso",
           "los gradientes se acumulan de un paso a otro: ¿falta optimizer.zero_grad()?")

    losses = [train_step(model, X, y, loss_fn, optimizer) for _ in range(10)]
    verify(losses[-1] < L, f"la pérdida baja: {L:.3f} -> {losses[-1]:.3f}",
           f"la pérdida no baja ({L:.3f} -> {losses[-1]:.3f}). Revisar el orden de los pasos.")


# ---------------------------------------------------------------------------
# Forward de prueba
# ---------------------------------------------------------------------------


def _hint(message):
    """Traduce los errores de dimensiones más comunes."""
    m = re.search(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)", message)
    if m:
        received, declared = m.group(2), m.group(3)
        return (
            f"Una capa nn.Linear recibe vectores de {received} números, pero está declarada "
            f"con {declared} entradas. Revisar el primer argumento de esa nn.Linear "
            "(normalmente la primera capa densa después de aplanar)."
        )
    m = re.search(r"expected input\[(.*?)\] to have (\d+) channels, but got (\d+) channels", message)
    if m:
        return (
            f"Una capa nn.Conv2d espera {m.group(2)} canales de entrada pero recibe {m.group(3)}. "
            "El primer argumento de cada Conv2d debe ser el número de filtros de la capa anterior."
        )
    return None


def shape_table(model, input_shape=(1, 1, 28, 28)):
    """Pasa una imagen inventada por la red e imprime, para cada capa, la forma de
    su entrada y de su salida y su número de parámetros. Devuelve la forma de la
    salida de la red, o None si hubo un error."""
    rows = []
    seen = set()

    def hook(name):
        def callback(layer, inputs, output):
            params = 0
            if name not in seen:
                params = sum(p.numel() for p in layer.parameters())
                seen.add(name)
            rows.append((name, type(layer).__name__, tuple(inputs[0].shape), tuple(output.shape), params))

        return callback

    hooks = [
        layer.register_forward_hook(hook(name))
        for name, layer in model.named_modules()
        if name and len(list(layer.children())) == 0
    ]
    output = None
    try:
        with torch.no_grad():
            output = model(torch.rand(*input_shape))
    except NotImplementedError as e:
        print(f"❌ {e}")
    except RuntimeError as e:
        print(f"❌ Error en el forward de prueba:\n   {e}")
        hint = _hint(str(e))
        if hint:
            print(f"   Pista: {hint}")
    finally:
        for h in hooks:
            h.remove()

    print(f"\n{'capa':10} {'tipo':10} {'entrada':18} {'salida':18} {'parámetros':>10}")
    print("-" * 70)
    for name, layer_type, input_shape, output_shape, params in rows:
        print(f"{name:10} {layer_type:10} {str(input_shape):18} {str(output_shape):18} {params:>10}")
    print("-" * 70)
    print(f"{'total':58} {count_parameters(model):>10}")
    print("(ReLU y flatten no aparecen porque no son capas declaradas en __init__.)")
    return None if output is None else tuple(output.shape)


def implemented(model_class):
    """True si la clase ya tiene __init__ y forward completos."""
    try:
        model = model_class()
        with torch.no_grad():
            model(torch.rand(2, 1, 28, 28))
        return count_parameters(model) > 0
    except (NotImplementedError, RuntimeError, AttributeError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def plot_histories(histories):
    """histories: diccionario {name: history devuelta por train}."""
    histories = {k: v for k, v in histories.items() if v is not None}
    if not histories:
        print("❌ Todavía no hay entrenamientos que graficar.")
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, history in histories.items():
        epochs = range(1, len(history["loss"]) + 1)
        axs[0].plot(epochs, history["loss"], "o-", label=name)
        axs[1].plot(epochs, history["val_accuracy"], "o-", label=name)
    axs[0].set(xlabel="época", ylabel="pérdida", title="Pérdida de entrenamiento")
    axs[1].set(xlabel="época", ylabel="exactitud", title="Exactitud de validación")
    for ax in axs:
        ax.legend()
        ax.xaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.show()


def plot_filters(model, image):
    """Los filtros de model.conv1 y sus mapas de activación (después de ReLU) para una imagen."""
    if not _validate_model(model):
        return
    weights = model.conv1.weight.detach()[:, 0]  # (filtros, k, k)
    with torch.no_grad():
        maps = torch.relu(model.conv1(image.reshape(1, 1, 28, 28)))[0]
    n = min(len(weights), 16)
    fig, axs = plt.subplots(2, n + 1, figsize=(1.3 * (n + 1), 2.9))
    axs[0, 0].imshow(image.reshape(28, 28), cmap="gray_r")
    axs[0, 0].set_title("imagen", fontsize=9)
    axs[1, 0].axis("off")
    for i in range(n):
        r = weights[i].abs().max()
        axs[0, i + 1].imshow(weights[i], cmap="gray_r", vmin=-r, vmax=r)
        axs[0, i + 1].set_title(f"filtro {i + 1}", fontsize=9)
        axs[1, i + 1].imshow(maps[i], cmap="gray_r")
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[1, 1].set_ylabel("mapa", fontsize=9)
    plt.tight_layout()
    plt.show()


def show_predictions(model, X, y, n=12, errors=False):
    """Muestra n imágenes con la predicción de la red y su probabilidad (softmax).
    Con errors=True, muestra solo imágenes mal clasificadas."""
    if not _validate_model(model):
        return
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)
    pred = probs.argmax(dim=1)
    indices = torch.nonzero(pred != y).ravel() if errors else torch.arange(len(y))
    indices = indices[:n]
    fig, axs = plt.subplots(1, len(indices), figsize=(1.2 * len(indices), 1.9))
    for ax, i in zip(np.atleast_1d(axs), indices):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        correct = pred[i] == y[i]
        text = f"{pred[i].item()} ({probs[i, pred[i]].item():.0%})"
        if not correct:
            text += f"\nreal: {y[i].item()}"
        ax.set_title(text, fontsize=9, color="black" if correct else "#d62728")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def summary(name, model, history):
    """Un resumen de la red para compartir (por ejemplo, en Classroom)."""
    if history is None or not _validate_model(model):
        print("❌ Primero hay que completar y entrenar la red.")
        return
    print(f"===== {name} =====")
    print(model)
    print(f"Parámetros: {count_parameters(model)}")
    print(
        f"Entrenamiento: {history['num_epochs']} épocas, mini-batches de {history['batch_size']}, "
        f"{history['steps']} pasos, {history['seconds']:.0f} s"
    )
    print(f"Exactitud de validación final: {history['val_accuracy'][-1]:.3f}")
