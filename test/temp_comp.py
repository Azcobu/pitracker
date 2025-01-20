import time
import adafruit_dht
import board
import serial

dht_device = adafruit_dht.DHT22(board.D4)

def read_sht41():
    try:
        with serial.Serial('/dev/ttyACM0', 9600, timeout=2) as ser:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                print("SHT41: no data received from sensor")

            _, temp, humid, touch = line.split(',')
            return float(temp)
    except Exception as err:
        print(f'SHT41 error:{err}')

def read_dht22():
    try:
        temperature_c = dht_device.temperature
        if temperature_c is not None:
            return temperature_c
    except Exception as err:
        print(f'DHT22 error:{err}')

def main():
    while True:
        try:
            dht_temp = read_dht22()
            sht_temp = read_sht41()
            if dht_temp and sht_temp:
                diff = round(abs(dht_temp - sht_temp), 2)
                print(f'DHT22: {dht_temp}, SHT41: {sht_temp}, difference= {diff}')
        except Exception as err:
            print(f"Error: {err}")
        time.sleep(2.0)

if __name__ == '__main__':
    main()