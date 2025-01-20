# To do:
# Avg and hottest temp last 24 hours? - done
# outline temp numbers for better visibility
# cache rendered font outlines in dict to avoid re-rendering - use function attribute
# at midnight get stats for the day - high, avg, for longterm use?
# make graph properly 24 hrs, not 24 hrs + up to another hr in temp buffer - done
# get display timeout working with touch sensor

import time
import subprocess
import csv
import os
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional, Tuple, Union, Dict
import pygame
import serial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

class SensorReadError(Exception):
    """Custom exception for sensor read failures"""

class DataValidationError(Exception):
    """Custom exception for data validation failures"""

class PiTracker:
    SERIAL_PORT = '/dev/ttyACM0'
    BAUD_RATE = 9600
    CSV_FILE = "temperature_log.csv"
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 480
    BACKGROUND_COLOUR = (0, 0, 0)
    TEXT_COLOUR = (255, 255, 255)
    TEXT2_COLOUR = (128, 128, 255)
    UPDATE_INTERVAL = 5
    GRAPH_UPDATE_INTERVAL = 300
    CSV_WRITE_INTERVAL = 3600
    HOURS_TO_KEEP = 24
    SCREEN_TIMEOUT = 60

    def __init__(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pi_tracker.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        pygame.init()
        self.screen = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.temp_font = pygame.font.SysFont(None, 112)
        self.humid_font = pygame.font.SysFont(None, 48)
        self.small_font = pygame.font.SysFont(None, 28)
        self.temp_graph = BytesIO()
        self.temp_buffer = []
        self.max_temp_past24 = 0.0
        self.avg_temp_past24 = 0.0
        self.min_temp_past24 = 0.0

        # Cache for rendered text
        self._text_cache: Dict[str, pygame.Surface] = {}

        pygame.mouse.set_visible(False)
        self.logger.info("PiTracker initialized")

    def read_sensor(self) -> Optional[Tuple[float, float, float]]:
        """
        Reads the current data from the sensor via the serial connection.
        Returns tuple of (temperature, humidity, touch) or raises SensorReadError
        """
        try:
            with serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1) as ser:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    raise SensorReadError("No data received from sensor")

                _, temp, humid, touch = line.split(',')
                return float(temp), float(humid), float(touch)

        except (serial.SerialException, serial.SerialTimeoutException) as e:
            self.logger.error("Serial communication error: %s", e)
            raise SensorReadError(f"Serial communication failed: {e}")
        except ValueError as e:
            self.logger.error("Invalid sensor data format: %s", line)
            raise SensorReadError(f"Invalid sensor data: {e}")

    def write_csv_from_buffer(self) -> None:
        """Writes buffered temperatures to the CSV file and prunes old data."""
        if not self.temp_buffer:
            self.logger.info("No data in buffer to write")
            return

        try:
            with open(self.CSV_FILE, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(self.temp_buffer)

            self.temp_buffer = []
            self.prune_csv()
            self.logger.info("Successfully wrote buffer to CSV and pruned old data")

        except IOError as e:
            self.logger.error("Failed to write to CSV file: %s", e)
            raise

    def prune_csv(self) -> None:
        """Keeps only the last 24 hours of data in the CSV file."""
        cutoff = datetime.now() - timedelta(hours=self.HOURS_TO_KEEP)

        try:
            # Read existing data
            with open(self.CSV_FILE, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                rows = [row for row in reader if datetime.fromisoformat(row[0]) > cutoff]

            # Write filtered data back
            with open(self.CSV_FILE, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(rows)

            self.logger.info("Pruned CSV file to last %s hours", self.HOURS_TO_KEEP)

        except IOError as e:
            self.logger.error("Failed to prune CSV file: %s", e)
            raise

    @staticmethod
    def is_png(data: BytesIO) -> bool:
        """Verify if BytesIO contains PNG data."""
        try:
            png_signature = b'\x89PNG\r\n\x1a\n'
            data.seek(0)
            header = data.read(8)
            data.seek(0)
            return header == png_signature
        except Exception as e:
            logging.error("Error checking PNG format: %s", e)
            return False

    def get_temp_graph(self) -> BytesIO:
        """Get the temperature graph BytesIO object, creating if necessary."""
        if self.temp_graph is None or self.temp_graph.closed:
            self.temp_graph = BytesIO()
        return self.temp_graph

    @staticmethod
    def nice_round(innum: float) -> str:
        """Format number to string, removing trailing .0 if present."""
        innum = str(round(innum, 1))
        return innum[:-2] if innum.endswith('.0') else innum

    def get_cached_text(self, text: str, font: pygame.font.Font, color: Tuple[int, int, int]) -> pygame.Surface:
        """Get cached text surface or render and cache if not exists."""
        cache_key = f"{text}{font.get_height()}{color}"
        if cache_key not in self._text_cache:
            self._text_cache[cache_key] = font.render(text, True, color)
        return self._text_cache[cache_key]

    def display_temperature(self, current_temp: Union[float, str], current_humid: Union[float, str]) -> None:
        """Updates the Pygame display with the current temperature and graph."""
        try:
            left_margin = 50
            self.screen.fill(self.BACKGROUND_COLOUR)

            # Display graph
            if self.temp_graph and self.is_png(self.temp_graph):
                try:
                    self.temp_graph.seek(0)
                    temp_data = BytesIO(self.temp_graph.getvalue())
                    graph_image = pygame.image.load(temp_data, 'png')
                    graph_rect = graph_image.get_rect(center=(self.DISPLAY_WIDTH // 2, self.DISPLAY_HEIGHT // 2))
                    self.screen.blit(graph_image, graph_rect)
                except Exception as e:
                    self.logger.error("Error loading graph: %s", e)
                    placeholder_text = self.get_cached_text("Graph load error", self.temp_font, self.TEXT_COLOUR)
                    self.screen.blit(placeholder_text, (20, self.DISPLAY_HEIGHT // 2))
            else:
                placeholder_text = self.get_cached_text("Rendering graph...", self.temp_font, self.TEXT_COLOUR)
                self.screen.blit(placeholder_text, (20, self.DISPLAY_HEIGHT // 2))

            # Display temperature
            if isinstance(current_temp, (float, int)):
                formatted_temp = self.nice_round(current_temp)
                temp_text = self.temp_font.render(f"{formatted_temp}°", True, self.TEXT_COLOUR)
            else:
                temp_text = self.get_cached_text("N/A", self.temp_font, self.TEXT_COLOUR)
            self.screen.blit(temp_text, (left_margin, 32))

            # Display humidity
            if isinstance(current_humid, (float, int)):
                formatted_humid = self.nice_round(current_humid)
                humid_text = self.humid_font.render(f"{formatted_humid}%", True, self.TEXT2_COLOUR)
            else:
                humid_text = self.get_cached_text("N/A", self.humid_font, self.TEXT2_COLOUR)
            self.screen.blit(humid_text, (left_margin, 110))

            # Display smaller stats
            xpad, ypad = 50, 30
            line1 = "Last 24 hours:"
            max_t = self.nice_round(self.max_temp_past24)
            avg_t = self.nice_round(self.avg_temp_past24)
            min_t = self.nice_round(self.min_temp_past24)
            line2 = f"Max: {max_t}°, Avg: {avg_t}°, Min: {min_t}°"
            line1_surface = self.small_font.render(line1, True, self.TEXT_COLOUR)
            line2_surface = self.small_font.render(line2, True, self.TEXT_COLOUR)

            # Get text sizes
            line1_width, line1_height = line1_surface.get_size()
            line2_width, line2_height = line2_surface.get_size()

            # Calculate positions for right-justified text
            line1_x = self.DISPLAY_WIDTH - xpad - line1_width
            line1_y = ypad
            line2_x = self.DISPLAY_WIDTH - xpad - line2_width
            line2_y = line1_y + line1_height + 5  # Slight gap between lines

            self.screen.blit(line1_surface, (line1_x, line1_y))
            self.screen.blit(line2_surface, (line2_x, line2_y))

            pygame.display.flip()
        except pygame.error as e:
            self.logger.error("Error updating display: %s", e)

    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate dataframe structure and content."""
        required_columns = {'timestamp', 'temperature', 'humidity'}
        if not all(col in df.columns for col in required_columns):
            raise DataValidationError("Missing required columns in dataframe")

        if df.empty:
            raise DataValidationError("No data available for plotting")

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise DataValidationError("Timestamp column must be datetime type")

    def generate_stats(self, temps: list) -> None:
        self.max_temp_past24 = max(temps)
        self.avg_temp_past24 = sum(temps) / len(temps)
        self.min_temp_past24 = min(temps)

    def generate_graph(self) -> None:
        """Generates a graph from the last 24 hours of temperature data."""
        try:
            timestamps, temperatures, humidities = [], [], []
            cutoff = datetime.now() - timedelta(hours=self.HOURS_TO_KEEP)

            if os.path.exists(self.CSV_FILE):
                with open(self.CSV_FILE, "r", encoding="utf-8") as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if datetime.fromisoformat(row[0]) > cutoff:
                            timestamps.append(datetime.fromisoformat(row[0]))
                            temperatures.append(float(row[1]))
                            humidities.append(float(row[2]))

            # Add data from the in-memory buffer
            for timestamp, temp, humid in self.temp_buffer:
                timestamps.append(datetime.fromisoformat(timestamp))
                temperatures.append(temp)
                humidities.append(humid)

            if not timestamps or not temperatures:
                self.logger.warning("No data available for graph generation")
                return

            if temperatures:
                self.generate_stats(temperatures)

            # Sort the data by timestamp
            combined_data = sorted(zip(timestamps, temperatures, humidities), key=lambda x: x[0])
            timestamps, temperatures, humidities = zip(*combined_data)

            df = pd.DataFrame({
                'timestamp': timestamps,
                'temperature': temperatures,
                'humidity': humidities
            })

            self.validate_data(df)
            self.plot_temp_humidity(df)

            # Update the graph buffer
            self.temp_graph = self.get_temp_graph()
            self.temp_graph.seek(0)
            self.temp_graph.truncate(0)
            plt.savefig(self.temp_graph, format='png', facecolor='black', edgecolor='none', bbox_inches='tight')
            self.temp_graph.seek(0)
            plt.close()

            self.logger.info("Graph generated successfully")

        except (IOError, DataValidationError) as e:
            self.logger.error("Error generating graph: %s", e)
            raise

    def create_temperature_gradient(self, df: pd.DataFrame, timestamps_num: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create the temperature gradient for the plot."""
        temperatures = df['temperature'].to_numpy()
        x_points = np.linspace(timestamps_num[0], timestamps_num[-1], 200)
        y_points = np.linspace(0, 45, 500)

        X, Y = np.meshgrid(x_points, y_points)
        temp_interpolated = np.interp(x_points, timestamps_num, temperatures)
        mask = Y > np.repeat(temp_interpolated[np.newaxis, :], len(y_points), axis=0)

        return X, Y, mask, temp_interpolated

    def plot_temp_humidity(self, df: pd.DataFrame) -> None:
        """Plot temperature and humidity data with gradient background."""
        try:
            fig = plt.figure(figsize=(10, 6), dpi=80)
            plt.style.use('dark_background')

            ax1 = plt.gca()
            temperature_range = [0, 15, 25, 30, 40, 45]
            colours = ['blue', 'green', 'yellow', 'orange', 'red', 'red']

            cmap = mcolors.LinearSegmentedColormap.from_list("temperature_gradient", colours)
            norm = mcolors.Normalize(vmin=min(temperature_range), vmax=max(temperature_range))

            timestamps_num = mdates.date2num(df['timestamp'])
            _, Y, mask, _ = self.create_temperature_gradient(df, timestamps_num)

            Z = norm(Y)
            gradient_colours = cmap(Z)
            gradient_colours[mask] = (0, 0, 0, 0)

            ax1.imshow(gradient_colours,
                      extent=(timestamps_num[0], timestamps_num[-1], 0, 45),
                      aspect='auto',
                      origin='lower')

            # Plot temperature line
            ax1.plot(df['timestamp'], df['temperature'], color='white',
                    linewidth=2, label='Temperature')

            # Add second y-axis for humidity
            ax2 = ax1.twinx()
            ax2.plot(df['timestamp'], df['humidity'], color='blue',
                    linewidth=2) # label='Humidity', alpha=0.8

            # Configure axes
            ax1.grid(visible=True, which='major', color='black', linestyle='-',
                    linewidth=1, alpha=1)
            ax1.xaxis.set_major_locator(mdates.HourLocator())
            ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
            ax1.set_frame_on(False)

            ax1.set_ylim(0, 45)
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='y', labelcolor='blue')
            # ax2.set_ylabel('Humidity %', color='blue')

            plt.tick_params(axis='both', which='major', labelsize=8, color='lightgray')
            plt.tight_layout()

        except Exception as e:
            self.logger.error("Error plotting temperature/humidity: %s", e)
            raise

    def cleanup(self) -> None:
        """Clean up resources before shutdown"""
        try:
            pygame.quit()
            # Don't close temp_graph here as it might be in use
            self._text_cache.clear()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error("Error during cleanup: %s", e)

    def run(self) -> None:
        last_graph_time = 0
        last_csv_time = 0

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                now_timestamp = int(time.time())
                current_temp, current_humid, current_touch = None, None, None

                if datetime.now().second % 5 == 0:
                    try:
                        sensor_data = self.read_sensor()
                        if sensor_data:
                            current_temp, current_humid, current_touch = sensor_data
                    except SensorReadError as e:
                        self.logger.warning("Failed to read sensor: %s", e)

                    self.display_temperature(
                        current_temp if current_temp is not None else 'N/A',
                        current_humid if current_humid is not None else 'N/A'
                    )

                    if current_temp:
                        if current_temp > self.max_temp_past24:
                            self.max_temp_past24 = current_temp
                            self.temp_buffer.append([datetime.now().isoformat(), current_temp, current_humid])
                        if current_temp < self.min_temp_past24:
                            self.min_temp_past24 = current_temp
                            self.temp_buffer.append([datetime.now().isoformat(), current_temp, current_humid])

                if now_timestamp - last_graph_time >= self.GRAPH_UPDATE_INTERVAL:
                    if current_temp is not None:
                        self.logger.info("Generating new graph...")
                        self.temp_buffer.append([datetime.now().isoformat(), current_temp, current_humid])
                        self.generate_graph()
                        last_graph_time = time.time()

                if now_timestamp - last_csv_time >= self.CSV_WRITE_INTERVAL:
                    self.logger.info("Writing to CSV...")
                    self.write_csv_from_buffer()
                    last_csv_time = time.time()

                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error("Unexpected error in main loop: %s", e)
        finally:
            self.cleanup()

if __name__ == "__main__":
    tracker = PiTracker()
    tracker.run()
