import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

# Sample data for properties
locations = ['A', 'B', 'C', 'D', 'E']
property_values = [100000, 200000, 500000, 1000000, 2000000]
mean_flood_damage = [5000, 10000, 25000, 35000, 39000]
std_dev_damage = [1, 1, 1, 1, 1]
return_period = 100 

# Define distributions for flood damage and frequency parameters
damage_distribution = gamma(1, scale=1000)
frequency_distrubution = gamma(2, scale=1)

# Bayesian estimation of flood losses for each location
average_loss_ratio_samples = []
num_samples = 10

for i in range(len(locations)):
    sample_damages = damage_distribution.rvs(size=num_samples)
    sample_frequencies = frequency_distrubution.rvs(size=num_samples)
    sample_ratios = (sample_damages + norm.rvs(scale=std_dev_damage[i])) / (property_values[i] * (1 / return_period))
    average_loss_ratio_samples.append(np.mean(sample_ratios))

average_loss_ratio_mean = np.mean(average_loss_ratio_samples)
average_loss_ratio_std = np.std(average_loss_ratio_samples)

# Plotting the average flood loss ratios for different locations
plt.bar(locations, average_loss_ratio_samples)
plt.axhline(average_loss_ratio_mean, color='r', linestyle='--', label='Mean')
plt.xlabel('Locations')
plt.ylabel('Average Flood Loss Ratio')
plt.title('Average Flood Loss Ratio for Different Locations')
plt.legend()
plt.show()

print('Probabilistic Average Flood Loss Ratio for Different Locations:')
print('Mean:', average_loss_ratio_mean)
print('Standard Deviation:', average_loss_ratio_std)

