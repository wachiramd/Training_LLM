import numpy as np
import matplotlib.pyplot as plt

block_size = 128
embedding_size = 64
dimensions_to_plot = 16
plot_density = 10

if embedding_size % 2 != 0:
    raise ValueError("embedding_size must be an even number.")

if dimensions_to_plot > embedding_size:
    dimensions_to_plot = embedding_size
elif dimensions_to_plot <= 0:
    raise ValueError("dimensions_to_plot must be a positive number.")

# Corresponds to 1 / (10000^(2i / embedding_size)) or exp(-(2i / embedding_size) * log(10000)).
even_numbers = np.arange(0, embedding_size, 2, dtype=np.float32)
denominator = np.exp(even_numbers * -(np.log(10000.0) / embedding_size))

# Generate a denser range of positions for smoother plotting
positions_smooth = np.linspace(
    0,
    block_size - 1,
    int(block_size * plot_density)
)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except IOError:
    print("Style 'seaborn-v0_8-whitegrid' not found, using default.")

figure, axis = plt.subplots(figsize=(15, 10))
figure.patch.set_facecolor('white')
axis.set_facecolor('white')

colors = ['#0077be', '#d95319']
vertical_offset_factor = 2.5

for i in range(0, dimensions_to_plot):
    color_index = i % 2
    denominator_value = denominator[i // 2]

    if i % 2 == 0:  # Even dimension index (0, 2, 4...) -> Sine
        pe_smooth = np.sin(positions_smooth * denominator_value)
    else:  # Odd dimension index (1, 3, 5...) -> Cosine
        pe_smooth = np.cos(positions_smooth * denominator_value)

    # Calculate the vertical offset for stacking lines visually
    offset = (dimensions_to_plot - 1 - i) * vertical_offset_factor
    axis.plot(
        positions_smooth,
        pe_smooth + offset,
        color=colors[color_index],
        linewidth=1.5,
    )

axis.set_title(
    f'Sinusoidal positional encoding (First {dimensions_to_plot} dimensions)',
    fontsize=20
)
axis.set_xlabel('Position in sequence', fontsize=18)
axis.set_ylabel('Dimension index', fontsize=18)

axis.set_yticks([])
axis.set_yticklabels([])

axis.set_xlim(0, block_size - 1)
axis.set_ylim(
    -vertical_offset_factor,
    (dimensions_to_plot + 0.5) * vertical_offset_factor
)

plt.tight_layout()
plt.savefig(
    "../images/sinusoidal_positional_encoding_smooth.svg",
    format="svg",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
