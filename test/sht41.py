import time
import serial
    
def read_sht41():

    try:
        with serial.Serial('/dev/ttyACM0', 9600, timeout=2) as ser:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                print("SHT41: no data received from sensor")
                return 'N/A'

            _, temp, humid, touch = line.split(',')
            return float(temp)

    except Exception as err:
        print(f'SHT41 error:{err}')
        return 'N/A'

def main():
    while True:
        try:
            temp = read_sht41()
            print(f'{temp}')
        except Exception as err:
            print(f'SHT41 error:{err}')

if __name__ == '__main__':
    main()