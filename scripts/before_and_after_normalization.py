import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
feature1_original = np.random.normal(loc=50, scale=10, size=100)
feature2_original = 0.05 * feature1_original + np.random.normal(loc=3, scale=0.8, size=100)
original_data = np.column_stack((feature1_original, feature2_original))

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(original_data)

marker_size = 50
alpha_value = 0.7
plt.figure(figsize=(12, 6))

# Plot 1: Before Normalization
plt.subplot(1, 2, 1)
plt.scatter(original_data[:, 0], original_data[:, 1], color='red', s=marker_size)
plt.title('Before normalization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, linestyle='--', alpha=alpha_value)

# Plot 2: After Normalization
plt.subplot(1, 2, 2)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], color='blue', s=marker_size)
plt.title('After normalization')
plt.xlabel('Feature 1 (Normalized)')
plt.ylabel('Feature 2 (Normalized)')
plt.grid(True, linestyle='--', alpha=alpha_value)

plt.tight_layout()
plt.savefig('before_and_after_normalization.png', dpi=300, bbox_inches='tight')
plt.show()
