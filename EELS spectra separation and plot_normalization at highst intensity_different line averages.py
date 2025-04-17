# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 02:04:27 2024

@author: panay
"""

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.ndimage import gaussian_filter1d
import os

# Ask user to select a file
Tk().withdraw()  # Hide the root window
filename = askopenfilename()  # Show the file selection dialog
data = np.loadtxt(filename)  # Load data from the selected file

# Determine the number of rows and columns in the data
num_rows, num_cols = data.shape

# Ask user for the number of sections to plot
num_sections = int(input("Enter the number of sections to plot: "))

# Ask user to select start and end rows for each section
section_starts = []
section_ends = []
for i in range(num_sections):
    start_row = int(input(f"Enter the starting row for section {i+1} (0-{num_rows-1}(top-bottom)): "))
    end_row = int(input(f"Enter the ending row for section {i+1} (0-{num_rows-1}(top-bottom)): "))
    section_starts.append(start_row)
    section_ends.append(end_row)

# Ask user for the range of columns to plot
start_col = 0
end_col = num_cols

# Select the subset of data based on the user's input
selected_data = data[:, start_col:end_col]

# Calculate the number of averages based on the section starts and ends
num_averages = num_sections

# Create an array to store the averages
averages = np.zeros((num_averages, end_col - start_col))

# Calculate the averages and store them in the array
avg_idx = 0
labels = []

for start_row, end_row in zip(section_starts, section_ends):
    # Average the rows between start_row and end_row for each section
    avg_data = np.mean(selected_data[start_row:end_row], axis=0)
    averages[avg_idx] = avg_data
    labels.append(f"Spectrum {start_row} to {end_row}")
    avg_idx += 1

# Ask user for the distance between the plotted averages
distance = float(input("Enter the intensity distance between the spectra (100-300, or 0-10 if normalization is used): "))

# Ask user for the value to add and multiply to the x-axis
add_value = float(input("Enter the value to add to the x-axis (e.g., 700-850 for Ni): "))
mult_value = float(input("Enter the value to multiply the x-axis by (resolution): "))

# Ask user for the standard deviation of the Gaussian kernel
gaussian_sigma = float(input("Enter the standard deviation of the Gaussian kernel (1 or 2): "))

# Smooth the averaged data using Gaussian filter
smoothed_averages = np.zeros_like(averages)
for i in range(num_averages):
    smoothed_averages[i] = gaussian_filter1d(averages[i], gaussian_sigma)

### Normalize each spectrum by its own maximum peak ###
normalized_averages = np.zeros_like(smoothed_averages)

for i in range(num_averages):
    local_max_peak = np.max(smoothed_averages[i])  # Find the local maximum of each spectrum
    normalized_averages[i] = smoothed_averages[i] / local_max_peak  # Normalize by its own peak

# Ask user for the x-range to show on the plot
x_start = float(input("Enter the starting value of x-axis range: "))
x_end = float(input("Enter the ending value of x-axis range: "))

# Ask user for the filename to save the results
save_filename = input("Enter the filename to save the results: ")

# Save the normalized and smoothed averages to a text file in the directory of the initial file
save_path = os.path.join(os.path.dirname(filename), save_filename + ".txt")
with open(save_path, "w") as f:
    for i in range(num_averages):
        x_vals = (np.arange(end_col - start_col) * mult_value) + add_value
        y_vals = normalized_averages[i] + i * distance
        data = np.column_stack((x_vals, y_vals))
        np.savetxt(f, data, fmt="%f", delimiter="\t")

# Plot the normalized, smoothed averages on top of each other with the specified distance
fig, ax = plt.subplots()
for i in range(num_averages):
    x_vals = (np.arange(end_col - start_col) * mult_value) + add_value
    ax.plot(x_vals, normalized_averages[i] + i * distance, label=labels[i])

ax.set_xlim(x_start, x_end)
ax.set_ylabel("Intensity (a.u.)", fontsize=14)
ax.set_xlabel("Energy-loss (eV)", fontsize=14)
ax.set_yticklabels([])  # Remove y-axis labels since they are not meaningful
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(loc="upper right", fontsize=10)
plt.show()
