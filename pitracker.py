import time
import csv
import os
from datetime import datetime, timedelta
from io import BytesIO
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
GRAPH_UPDATE_INTERVAL = 60
CSV_WRITE_INTERVAL = 300  
HOURS_TO_KEEP = 24

pygame.init()
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
font = pygame.font.SysFont(None, 112)
temp_graph = BytesIO() 

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

def is_png(data: BytesIO) -> bool:
    try:
        png_signature = b'\x89PNG\r\n\x1a\n'
        data.seek(0)
        header = data.read(8)  
        data.seek(0) 
        return header == png_signature
    except Exception as err:
        print(f'Error checking BytesIO object - {err}')
        return False

def display_temperature(current_temp, graph_path):
    """Updates the Pygame display with the current temperature and graph."""
    global temp_graph
    screen.fill(BACKGROUND_COLOR)

    # Display graph
    if temp_graph and is_png(temp_graph):
        try:
            temp_graph.seek(0)
            graph_image = pygame.image.load(temp_graph, 'png')
            graph_rect = graph_image.get_rect(center=(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2))
            screen.blit(graph_image, graph_rect)
        except Exception as err:
            print(f'Error loading graph: {err}')
            print(type(temp_graph))
    else:
        placeholder_text = font.render("Graph not available", True, TEXT_COLOR)
        screen.blit(placeholder_text, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2))

    # Display current temperature
    temp_text = font.render(f"{current_temp:.1f}°", True, TEXT_COLOR)
    screen.blit(temp_text, (32, 32))

    pygame.display.flip()

def generate_graph():
    """Generates a graph from the last 24 hours of temperature data."""
    global temp_graph

    timestamps, temperatures = [], []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                timestamps.append(datetime.fromisoformat(row[0]))
                temperatures.append(float(row[1]))

    # Add data from the in-memory buffer
    for timestamp, temp in temp_buffer:
        timestamps.append(datetime.fromisoformat(timestamp))
        temperatures.append(temp)

    if not timestamps or not temperatures:
        print("Can't generate graph - no data available")
        return

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
    plt.gca().set_frame_on(False)

    plt.tick_params(axis='both', which='major', labelsize=8, color='lightgray')

    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    plt.gca().set_title('')


    plt.tight_layout()
    #plt.savefig('temperature_graph.png', facecolor='black', edgecolor='none')

    temp_graph.seek(0)
    temp_graph.truncate(0)
    plt.savefig(temp_graph, format='png', facecolor='black', edgecolor='none', bbox_inches='tight')
    temp_graph.seek(0)
    plt.close()

def main():
    # 2 timers - 5 minutes for graph updates, and 1 hour for CSV updates
    last_graph_time = 0
    last_csv_time = 0
    
    pygame.mouse.set_visible(False)
    graph_path = "temperature_graph.png"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        now_timestamp = int(time.time())
        current_temp, current_humid, current_touch = None, None, None

        if datetime.now().second % 5 == 0:
            # Read current temperature
            sensor_return = read_sensor()
            if sensor_return and len(sensor_return) == 3:
                current_temp, current_humid, current_touch = sensor_return

            display_temperature(current_temp if current_temp is not None else 0.0, graph_path)

        if now_timestamp - last_graph_time >= GRAPH_UPDATE_INTERVAL:
            if current_temp is not None:
                
                print(f'{datetime.now().strftime("%H:%M:%S")} - generating new graph...')
                temp_buffer.append([datetime.now().isoformat(), current_temp])
                generate_graph()
                last_graph_time = time.time()

        if now_timestamp - last_csv_time >= CSV_WRITE_INTERVAL:
            print(f'{datetime.now().strftime("%H:%M:%S")} - writing to CSV...')
            write_csv_from_buffer()
            prune_csv()
            last_csv_time = time.time()

        time.sleep(1)

if __name__ == "__main__":
    main()