import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for height, weight, and age
np.random.seed(0)
num_samples = 100
height = np.random.normal(170, 10, num_samples)  # Mean height: 170 cm, Standard deviation: 10 cm
weight = height * 0.6 + np.random.normal(0, 5, num_samples)  # Weight is 60% of height with some noise
age = np.random.normal(33, 10, num_samples)  # Mean age: 30 years, Standard deviation: 5 years

# Stack height, weight, and age into a single data matrix
data = np.column_stack((height, weight, age))

# Compute covariance matrix
cov_matrix = np.cov(data, rowvar=False)

# Plot the data and covariance matrix
plt.figure(figsize=(12, 4))

# Scatter plot of height, weight, and age
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.scatter(height, weight, age, c='blue', alpha=0.6)
ax1.set_xlabel('Height (cm)')
ax1.set_ylabel('Weight (kg)')
ax1.set_zlabel('Age (years)')
ax1.set_title('Physical Attributes')

# Plot covariance matrix as heatmap
plt.subplot(1, 2, 2)
plt.imshow(cov_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix (Physical Attributes)')
plt.xticks(range(len(cov_matrix)), ['Height', 'Weight', 'Age'])
plt.yticks(range(len(cov_matrix)), ['Height', 'Weight', 'Age'])
plt.xlabel('Variables')
plt.ylabel('Variables')

plt.tight_layout()
plt.show()
