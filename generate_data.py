"""
Generate synthetic post-silicon semiconductor test data
This script creates a large dataset simulating post-silicon validation measurements
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_CHIPS = 50000  # Number of chips tested
NUM_FEATURES = 100  # Number of test parameters per chip

print(f"Generating synthetic post-silicon semiconductor data...")
print(f"Number of chips: {NUM_CHIPS}")
print(f"Number of features per chip: {NUM_FEATURES}")

# Generate chip IDs
chip_ids = [f"CHIP_{i:06d}" for i in range(NUM_CHIPS)]

# Generate feature names (typical post-silicon test parameters)
feature_names = []
feature_categories = {
    'voltage': 20,
    'current': 15,
    'frequency': 15,
    'temperature': 10,
    'power': 10,
    'timing': 15,
    'leakage': 10,
    'noise': 5
}

for category, count in feature_categories.items():
    for i in range(count):
        feature_names.append(f"{category}_{i+1}")

# Generate synthetic data with realistic distributions
data = {}
data['chip_id'] = chip_ids

# Voltage measurements (mV) - normally distributed around 1200mV
for i in range(feature_categories['voltage']):
    data[f'voltage_{i+1}'] = np.random.normal(1200, 50, NUM_CHIPS)

# Current measurements (mA) - log-normal distribution
for i in range(feature_categories['current']):
    data[f'current_{i+1}'] = np.random.lognormal(3, 0.5, NUM_CHIPS)

# Frequency measurements (MHz) - normally distributed
for i in range(feature_categories['frequency']):
    base_freq = 2000 + i * 100
    data[f'frequency_{i+1}'] = np.random.normal(base_freq, 50, NUM_CHIPS)

# Temperature measurements (°C) - normally distributed
for i in range(feature_categories['temperature']):
    data[f'temperature_{i+1}'] = np.random.normal(65, 10, NUM_CHIPS)

# Power measurements (W) - gamma distribution
for i in range(feature_categories['power']):
    data[f'power_{i+1}'] = np.random.gamma(2, 15, NUM_CHIPS)

# Timing measurements (ns) - exponential distribution
for i in range(feature_categories['timing']):
    data[f'timing_{i+1}'] = np.random.exponential(5, NUM_CHIPS)

# Leakage current (nA) - log-normal distribution
for i in range(feature_categories['leakage']):
    data[f'leakage_{i+1}'] = np.random.lognormal(2, 1, NUM_CHIPS)

# Noise measurements (dB) - normally distributed
for i in range(feature_categories['noise']):
    data[f'noise_{i+1}'] = np.random.normal(-80, 5, NUM_CHIPS)

# Create DataFrame
df = pd.DataFrame(data)

# Add pass/fail classification based on multiple criteria
# A chip fails if any critical parameter is out of spec
df['pass_fail'] = 'PASS'

# Voltage out of range (1100-1300 mV)
voltage_fail = (df['voltage_1'] < 1100) | (df['voltage_1'] > 1300)

# Temperature too high (> 85°C)
temp_fail = df['temperature_1'] > 85

# Frequency too low (< 1900 MHz)
freq_fail = df['frequency_1'] < 1900

# Power too high (> 50W)
power_fail = df['power_1'] > 50

# Mark failures
df.loc[voltage_fail | temp_fail | freq_fail | power_fail, 'pass_fail'] = 'FAIL'

# Add lot and wafer information
df['lot_id'] = [f"LOT_{i//1000:03d}" for i in range(NUM_CHIPS)]
df['wafer_id'] = [f"WAFER_{i//200:03d}" for i in range(NUM_CHIPS)]

# Reorder columns
cols = ['chip_id', 'lot_id', 'wafer_id'] + feature_names + ['pass_fail']
df = df[cols]

# Save to CSV
output_file = 'data/semiconductor_test_data.csv'
os.makedirs('data', exist_ok=True)
df.to_csv(output_file, index=False)

print(f"\nData generation complete!")
print(f"Output file: {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
print(f"\nDataset summary:")
print(f"  Total chips: {len(df)}")
print(f"  Passed: {(df['pass_fail'] == 'PASS').sum()} ({(df['pass_fail'] == 'PASS').sum()/len(df)*100:.1f}%)")
print(f"  Failed: {(df['pass_fail'] == 'FAIL').sum()} ({(df['pass_fail'] == 'FAIL').sum()/len(df)*100:.1f}%)")
print(f"  Features: {len(feature_names)}")
