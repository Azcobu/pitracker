import time
import board
import adafruit_dht
import serial

# Initialize DHT22 (GPIO4, pin 7)
dht22 = adafruit_dht.DHT22(board.D4)  # Ensure GPIO4 is correct for your setup

# Initialize serial connection for SHT41
sht41_serial = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)

def read_sensors():
    while True:
        try:
            # Read DHT22 data
            dht_temp = dht22.temperature
            dht_humidity = dht22.humidity

            # Read SHT41 data via serial
            sht41_serial.write(b"read\n")  # Command to read data (check your device's manual)
            response = sht41_serial.readline().decode().strip()

            if response:
                sht_temp, sht_humidity = map(float, response.split(','))  # Adjust parsing as needed
            else:
                sht_temp = sht_humidity = None

            # Print readings
            print(f"DHT22: Temp={dht_temp:.1f}°C, Humidity={dht_humidity:.1f}%")
            if sht_temp is not None and sht_humidity is not None:
                print(f"SHT41: Temp={sht_temp:.1f}°C, Humidity={sht_humidity:.1f}%")
            else:
                print("SHT41: No data received")

        except RuntimeError as e:
            print(f"DHT22 RuntimeError: {e}. Retrying...")

        except Exception as e:
            print(f"Unexpected error: {e}")
            break

        time.sleep(2)

if __name__ == "__main__":
    try:
        read_sensors()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        dht22.exit()  # Clean up DHT22 resources
        sht41_serial.close()  # Close serial connection
