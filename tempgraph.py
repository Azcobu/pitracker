import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('testdata.csv', parse_dates=['timestamp'])

plt.figure(figsize=(10, 6), dpi=80)
plt.style.use('dark_background')

# Define the updated temperature range and colors
temperature_range = [0, 15, 25, 30, 40, 45]  
colors = ['blue', 'green', 'yellow', 'orange', 'red', 'red']

cmap = mcolors.LinearSegmentedColormap.from_list("temperature_gradient", colors)
norm = mcolors.Normalize(vmin=min(temperature_range), vmax=max(temperature_range))
timestamps = mdates.date2num(df['timestamp'])  # Convert timestamps to numeric format
temperatures = df['temperature'].to_numpy()  # Get temperatures as a numpy array
X, Y = np.meshgrid(timestamps, np.linspace(0, 45, 500))  # Extend range to 45 for gradient
Z = norm(Y)  # Normalize the vertical gradient
gradient_colors = cmap(Z)  # Map normalized values to colors
mask = Y > np.interp(X[0], timestamps, temperatures)  # Mask above the curve
gradient_colors[mask] = (0, 0, 0, 0)  # Make masked areas transparent

plt.imshow(gradient_colors, extent=(timestamps[0], timestamps[-1], 0, 45), aspect='auto', origin='lower')

plt.plot(df['timestamp'], df['temperature'], color='white', linewidth=2)

plt.grid(visible=True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.gca().xaxis.set_major_locator(mdates.HourLocator())  
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5)) 

plt.tick_params(axis='both', which='major', labelsize=8, color='lightgray')

plt.gca().set_xlabel('')
plt.gca().set_ylabel('')
plt.gca().set_title('')

current_temp = df['temperature'].iloc[-1]
plt.text(0.2, 0.85, f'{current_temp:.1f}°',  
         transform=plt.gca().transAxes, 
         fontsize=80,  
         fontweight='bold', 
         fontfamily='Arial',
         color='cyan',
         ha='center',  
         va='center')    

plt.tight_layout()
plt.savefig('temperature_graph.png', facecolor='black', edgecolor='none')
plt.close()
