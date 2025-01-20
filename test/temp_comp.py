# temperature comparison
import serial
import adafruit_dht
import board
import time

class SensorReadError(Exception):
    """Custom exception for sensor read failures"""

def read_sht41() -> Optional[Tuple[float, float, float]]:
    serial_port = '/dev/ttyACM0'
    baud_rate = 9600

    try:
        with serial.Serial(serial_port,baud_rate, timeout=1) as ser:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                raise SensorReadError("No data received from sensor")

            _, temp, humid, touch = line.split(',')
            return float(temp) #, float(humid), float(touch)

    except (serial.SerialException, serial.SerialTimeoutException) as e:
        print("Serial communication error: %s", e)
        #raise SensorReadError(f"Serial communication failed: {e}")
    except ValueError as e:
        print("Invalid sensor data format: %s", line)
        #raise SensorReadError(f"Invalid sensor data: {e}")

def read_dht22():
    dht_sensor = adafruit_dht.DHT22(board.D4)
    try:
        temperature = dht_sensor.temperature
        #humidity = dht_sensor.humidity

        if temperature is not None # and humidity is not None:
            return temperature
        else:
            print("Failed to retrieve data from sensor. Trying again...")
    except RuntimeError as error:
        # Handle occasional sensor read errors
        print(f"Sensor error: {error}")


def main():

    while True:
        trinkey_temp = read_sht41()
        dht22_temp = read_dht22()
        print(f'Trinkey: {trinkey_temp}, DHT22: {dht22_temp}')
        time.sleep(2)

if __name__ == '__main__':
    main()