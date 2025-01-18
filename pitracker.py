import time
import csv
import os
from datetime import datetime, timedelta
from io import BytesIO
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
BACKGROUND_COLOUR = (0, 0, 0)
TEXT_COLOUR = (255, 255, 255)
TEXT2_COLOUR = (128,128, 255)
UPDATE_INTERVAL = 5
GRAPH_UPDATE_INTERVAL = 300
CSV_WRITE_INTERVAL = 3600
HOURS_TO_KEEP = 24

pygame.init()
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
font = pygame.font.SysFont(None, 112)
font2 = pygame.font.SysFont(None, 48)
temp_graph = BytesIO() 

# In-memory buffer for temperature readings
temp_buffer = []

def read_sensor():
    """Reads the current data from the sensor via the serial connection."""
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
    
        # Read a line from the serial port
        line = ser.readline().decode('utf-8').strip()
        #print(f'{datetime.now().strftime("%H:%M:%S")} - Sensor returned: {line}')
        if line:
            try:
                _, temp, humid, touch = line.split(',')
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

def get_temp_graph():
    global temp_graph
    if temp_graph is None or temp_graph.closed:
        temp_graph = BytesIO()
    return temp_graph

def display_temperature(current_temp, current_humid):
    """Updates the Pygame display with the current temperature and graph."""
    global temp_graph
    screen.fill(BACKGROUND_COLOUR)

    # Display graph
    if temp_graph and is_png(temp_graph):
        try:
            temp_graph.seek(0)
            # Create a copy of the data for pygame to use
            temp_data = BytesIO(temp_graph.getvalue())
            graph_image = pygame.image.load(temp_data, 'png')
            graph_rect = graph_image.get_rect(center=(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2))
            screen.blit(graph_image, graph_rect)
        except Exception as err:
            print(f'Error loading graph: {err}')
            print(type(temp_graph))
    else:
        placeholder_text = font.render("Graph not available", True, TEXT_COLOUR)
        screen.blit(placeholder_text, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2))

    # Display current temperature
    if isinstance(current_temp, (float, int)):
        temp_text = font.render(f"{current_temp:.1f}°", True, TEXT_COLOUR)
    else:
        temp_text = font.render("N/A", True, TEXT_COLOUR)
    screen.blit(temp_text, (42, 32))

    # Display current humidity

    if isinstance(current_humid, (float, int)):
        temp_humid = font2.render(f"{current_humid:.1f}%", True, TEXT2_COLOUR)
    else:
        temp_humid = font2.render("N/A", True, TEXT2_COLOUR)
    screen.blit(temp_humid, (42, 115))

    pygame.display.flip()

def generate_graph():
    """Generates a graph from the last 24 hours of temperature data."""
    global temp_graph

    timestamps, temperatures, humidities = [], [], []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                timestamps.append(datetime.fromisoformat(row[0]))
                temperatures.append(float(row[1]))
                humidities.append(float(row[2]))

    # Add data from the in-memory buffer
    for timestamp, temp, humid in temp_buffer:
        timestamps.append(datetime.fromisoformat(timestamp))
        temperatures.append(temp)
        humidities.append(humid)

    if not timestamps or not temperatures:
        print("Can't generate graph - no data available")
        return

    # Sort the data by timestamp
    combined_data = sorted(zip(timestamps, temperatures, humidities), key=lambda x: x[0])
    timestamps, temperatures, humidities = zip(*combined_data)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'humidity': humidities
    })

    fig = plot_temp_humidity(df)
    temp_graph = get_temp_graph()
    temp_graph.seek(0)
    temp_graph.truncate(0)
    plt.savefig(temp_graph, format='png', facecolor='black', edgecolor='none', bbox_inches='tight')
    temp_graph.seek(0)
    plt.close()

def plot_temp_humidity(df): 
    plt.figure(figsize=(10, 6), dpi=80)
    plt.style.use('dark_background')

    # Define the temperature range and colours
    temperature_range = [0, 15, 25, 30, 40, 45]  
    colours = ['blue', 'green', 'yellow', 'orange', 'red', 'red']

    cmap = mcolors.LinearSegmentedColormap.from_list("temperature_gradient", colours)
    norm = mcolors.Normalize(vmin=min(temperature_range), vmax=max(temperature_range))
    
    # Convert timestamps to numerical values for interpolation
    timestamps_num = mdates.date2num(df['timestamp'])
    temperatures = df['temperature'].to_numpy()
    humidities = df['humidity'].to_numpy()
    
    # Create a regular grid of x (time) points
    x_points = np.linspace(timestamps_num[0], timestamps_num[-1], 200)  # Increase resolution
    y_points = np.linspace(0, 45, 500)
    
    # Create the mesh grid using the regular x points
    X, Y = np.meshgrid(x_points, y_points)
    
    # Interpolate temperatures for each x point in the grid
    temp_interpolated = np.interp(x_points, timestamps_num, temperatures)
    
    # Create mask for each vertical slice
    mask = Y > np.repeat(temp_interpolated[np.newaxis, :], len(y_points), axis=0)
    
    # Create gradient colors
    Z = norm(Y)
    gradient_colours = cmap(Z)
    gradient_colours[mask] = (0, 0, 0, 0)  # Make masked areas transparent

    # Plot gradient using the numerical timestamps
    plt.imshow(gradient_colours, 
              extent=(timestamps_num[0], timestamps_num[-1], 0, 45), 
              aspect='auto', 
              origin='lower')

    # Plot temperature line
    plt.plot(df['timestamp'], temperatures, color='white', linewidth=2, label='Temperature')
    
    # Plot scaled humidity line
    scaled_humidity = (humidities / 100) * 45  # Scale humidity to match temperature range
    plt.plot(df['timestamp'], scaled_humidity, color='blue', linewidth=2, 
            label='Humidity', alpha=0.8)

    # Grid and formatting
    plt.grid(visible=True, which='major', color='gray', linestyle='--', 
            linewidth=0.5, alpha=0.5)
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
    plt.gca().set_frame_on(False)

    # Add second y-axis for humidity percentage
    ax2 = plt.gca().twinx()
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2.set_ylabel('Humidity %', color='skyblue')

    # Formatting
    plt.tick_params(axis='both', which='major', labelsize=8, color='lightgray')
    plt.legend(loc='upper right')
    plt.tight_layout()

    return plt.gcf()

def main():
    # 2 timers - 5 minutes for graph updates, and 1 hour for CSV updates
    last_graph_time = 0
    last_csv_time = 0
    
    pygame.mouse.set_visible(False)

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

            display_temperature(current_temp if current_temp is not None else 'N/A',\
                                current_humid if current_humid is not None else 'N/A')

        if now_timestamp - last_graph_time >= GRAPH_UPDATE_INTERVAL:
            if current_temp is not None:
                
                print(f'{datetime.now().strftime("%H:%M:%S")} - generating new graph...')
                temp_buffer.append([datetime.now().isoformat(), current_temp, current_humid])
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