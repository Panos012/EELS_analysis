# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 13:06:55 2023

@author: panay
"""

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.ndimage import gaussian_filter1d
import math
import os

# define a function to calculate the mean of a list of numbers
def mean(data):
    return sum(data) / len(data)

# define a function to calculate the standard deviation of a list of numbers
def stdev(data):
    data_mean = mean(data)
    variance = sum((x - data_mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

# define a function to normalize a list of numbers using z-score normalization
def zscore_normalization(data):
    data_mean = mean(data)
    data_stdev = stdev(data)
    normalized_data = [(x - data_mean) / data_stdev for x in data]
    return normalized_data

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

# Ask user how many rows to average together
avg_rows = int(input("Enter the number of rows to average together: "))

# Calculate the number of averages to be plotted
num_averages = sum([(end_row - start_row) // avg_rows for start_row, end_row in zip(section_starts, section_ends)])

# Create an array to store the averages
averages = np.zeros((num_averages, end_col - start_col))

# Calculate the averages and store them in the array
avg_idx = 0
labels = []

for start_row, end_row in zip(section_starts, section_ends):
    for i in range((end_row - start_row) // avg_rows):
        avg_start_row = start_row + i * avg_rows
        avg_end_row = start_row + (i + 1) * avg_rows
        averages[avg_idx] = np.mean(selected_data[avg_start_row:avg_end_row], axis=0)
        labels.append(f"Spectrum {start_row} until {end_row}")
        avg_idx += 1

# Ask user for the distance between the plotted averages
distance = int(input("Enter the Intensity distance between the spectra (100-300,0-10 if normalization is used): "))

# Ask user for the value to add and multiply to the x-axis
add_value = float(input("Enter the value to add to the x-axis (ex.700-850 for Ni): "))
mult_value = float(input("Enter the value to multiply the x-axis by (resolution): "))

# Ask user for the standard deviation of the Gaussian kernel
gaussian_sigma = int(input("Enter the standard deviation of the Gaussian kernel (1 or 2): "))

# Smooth the averaged data using Gaussian filter
smoothed_averages = np.zeros_like(averages)
for i in range(num_averages):
    smoothed_averages[i] = gaussian_filter1d(averages[i], gaussian_sigma)

# Normalize the smoothed averages
normalized_averages = np.zeros_like(smoothed_averages)
for i in range(num_averages):
    normalized_averages[i] = zscore_normalization(smoothed_averages[i])

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
        # y_vals = smoothed_averages[i] + i * distance # Enable this to use avoid normalization
        # y_vals = averages[i] + i * distance # Enable this to use the raw data and not the Gaussian smooth 
        data = np.column_stack((x_vals, y_vals))
        np.savetxt(f, data, fmt="%f", delimiter="\t")

# Plot the smoothed averages on top of each other with the specified distance
fig, ax = plt.subplots()
for i in range(num_averages):
    x_vals = (np.arange(end_col - start_col) * mult_value) + add_value
    #ax.plot(x_vals, normalized_averages[i] + i * distance)
    ax.plot(x_vals, smoothed_averages[i] + i * distance) # Enable this to use avoid normalization
    #ax.plot(x_vals, averages[i] + i * distance) # Enable this to use the raw data and not the Gaussian smooth 

ax.set_xlim(x_start, x_end)
ax.set_ylabel("Intensity (a.u.)", fontsize=14)
ax.set_xlabel("Energy-loss (eV)", fontsize=14)
ax.set_yticklabels([])  # Remove y-axis labels since they are not meaningful
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(loc="upper right", fontsize=10)
plt.show()
