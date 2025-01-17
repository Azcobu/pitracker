import time
import csv
import os
from datetime import datetime, timedelta
import threading
import pygame
import serial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE = 9600

CSV_FILE = "temperature_log.csv"
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
UPDATE_INTERVAL = 5
CSV_WRITE_INTERVAL = 3600  
HOURS_TO_KEEP = 24

pygame.init()
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("Temperature Display")
font = pygame.font.SysFont(None, 96)

# In-memory buffer for temperature readings
temp_buffer = []

def read_sensor():
    """Reads the current data from the sensor via the serial connection."""
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
    
        # Read a line from the serial port
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                sernum, temp, humid, touch = line.split(',')
                return float(temp), float(humid), float(touch)
            except Exception as err:
                print(f'{err} - returned output was {line}')
                return None
            
def write_csv_from_buffer():
    """Writes buffered temperatures to the CSV file and prunes old data."""
    global temp_buffer

    # Write buffered data to CSV
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(temp_buffer)

    # Clear the buffer
    temp_buffer = []

    # Prune entries older than 24 hours
    prune_csv()

def prune_csv():
    """Keeps only the last 24 hours of data in the CSV file."""
    cutoff = datetime.now() - timedelta(hours=HOURS_TO_KEEP)
    rows = []

    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        rows = [row for row in reader if datetime.fromisoformat(row[0]) > cutoff]

    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def display_temperature(current_temp, graph_path):
    """Updates the Pygame display with the current temperature and graph."""
    screen.fill(BACKGROUND_COLOR)

    # Display graph
    if os.path.exists(graph_path):
        graph_image = pygame.image.load(graph_path)
        graph_rect = graph_image.get_rect(center=(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2 + 40))
        screen.blit(graph_image, graph_rect)
    else:
        placeholder_text = font.render("Graph not available", True, TEXT_COLOR)
        screen.blit(placeholder_text, (DISPLAY_WIDTH // 2 - 100, DISPLAY_HEIGHT // 2))

    # Display current temperature
    temp_text = font.render(f"{current_temp:.2f} °C", True, TEXT_COLOR)
    screen.blit(temp_text, (20, 20))

    pygame.display.flip()

def generate_graph():
    """Generates a graph from the last 24 hours of temperature data."""
    timestamps, temperatures = [], []

    # Read data from CSV
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row[0]))
            temperatures.append(float(row[1]))

    # Add data from the in-memory buffer
    for timestamp, temp in temp_buffer:
        timestamps.append(datetime.fromisoformat(timestamp))
        temperatures.append(temp)

    # Sort the data by timestamp
    combined_data = sorted(zip(timestamps, temperatures), key=lambda x: x[0])
    timestamps, temperatures = zip(*combined_data)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures
    })

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

def main():
    pygame.mouse.set_visible(False)
    threading.Thread(target=csv_writer_thread, daemon=True).start()

    graph_path = "temperature_graph.png"
    last_graph_update = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Read current temperature
        sensor_return = read_sensor()
        if sensor_return and len(sensor_return) == 3:
            current_temp, current_humid, current_touch = sensor_return
        else:
            current_temp, current_humid, current_touch = None, None, None

        if current_temp is not None:
            # Buffer the current temperature with a timestamp
            temp_buffer.append([datetime.now().isoformat(), current_temp])

        # Generate graph every 5 minutes
        if time.time() - last_graph_update > 20: #QQQQ
            generate_graph()
            last_graph_update = time.time()

        # Display updates
        display_temperature(current_temp if current_temp is not None else 0.0, graph_path)
        time.sleep(UPDATE_INTERVAL)

def csv_writer_thread():
    while True:
        # Write buffered data to CSV every hour
        write_csv_from_buffer()
        time.sleep(CSV_WRITE_INTERVAL)

if __name__ == "__main__":
    main()