# temperature comparison
import serial
import adafruit_dht
import board
import time

dht22 = adafruit_dht.DHT22(board.D4)  


def read_sht41():
    serial_port = '/dev/ttyACM0'
    baud_rate = 9600

    try:
        with serial.Serial(serial_port,baud_rate, timeout=1) as ser:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                print("SHT41: no data received from sensor")
                return 'N/A'

            _, temp, humid, touch = line.split(',')
            return float(temp)

    except Exception as err:
        print(f'SHT41 error:{err}')
        return 'N/A'

def read_dht22():
    try:
        temperature = dht_sensor.temperature

        if temperature is not None:
            return temperature
        else:
            print("DHT22: failed to retrieve data from sensor.")
            return 'N/A'
    except Exception as err:
        print(f'DHT22: sensor error: {err}')
        return 'N/A'

def main():

    while True:
    try:
        trinkey_temp = read_sht41()
        dht22_temp = read_dht22()
        print(f'Trinkey: {trinkey_temp}, DHT22: {dht22_temp}')
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        dht22.exit()  # Clean up DHT22 resources
        sht41_serial.close()  # Close serial connection

if __name__ == '__main__':
    main()
