import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray


def sigmoid(x: ndarray) -> ndarray:
    # Sigmoid: f(x) = 1 / (1 + exp(-x))
    return 1 / (1 + np.exp(-x))


def relu(x: ndarray) -> ndarray:
    # ReLU: f(x) = max(0, x)
    return np.maximum(0, x)


def gelu(x: ndarray) -> ndarray:
    # GeLU (Gaussian Error Linear Unit approximation):
    # f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def leaky_relu(x: ndarray, alpha: float = 0.1) -> ndarray:
    # LeakyReLU: f(x) = alpha * x if x < 0 else x
    return np.where(x > 0, x, x * alpha)


def silu(x: ndarray) -> ndarray:
    # SiLU (Sigmoid Linear Unit, also known as Swish):
    # f(x) = x * sigmoid(x)
    return x * sigmoid(x)


max_absolute_value = 3.5
x_values = np.linspace(-max_absolute_value, max_absolute_value, 400)

y_sigmoid = sigmoid(x_values)
y_relu = relu(x_values)
y_gelu = gelu(x_values)
y_leaky_relu = leaky_relu(x_values)
y_silu = silu(x_values)

plt.figure(figsize=(12, 8))

line_width = 3
plt.plot(x_values, y_sigmoid, label='Sigmoid', linewidth=line_width, color='blue')
plt.plot(x_values, y_relu, label='ReLU', linewidth=line_width, color='red')
plt.plot(x_values, y_gelu, label='GeLU', linewidth=line_width, color='green')
plt.plot(x_values, y_leaky_relu, label=r'LeakyReLU ($\alpha=0.1$)', linewidth=line_width, color='orange')
plt.plot(x_values, y_silu, label=r'SiLU / Swish ($x\sigma(x)$)', linewidth=line_width, color='purple')

plt.title('Activation functions')
plt.xlabel('x')
plt.ylabel('f(x)')

plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
plt.show()
