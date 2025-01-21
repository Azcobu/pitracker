import time
import serial
    
def read_sht41():
    try:
        with serial.Serial('COM5', 9600, timeout=2) as ser:
            ser.reset_input_buffer()
            ser.write(b"get_data\n")
            ser.flush()  
            
            # Wait a bit for the sensor to take measurements
            time.sleep(0.1)
            response = ser.readline().decode('utf-8').strip()
            
            if not response:
                print("SHT41: no data received from sensor")
                return None
                
            try:
                temp, humid = map(float, response.split(','))
                return temp, humid
            except ValueError:
                print(f"SHT41: unexpected response format: {response}")
                return None
                
    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
        return None

def main():
    while True:
        try:
            temp = read_sht41()
            print(f'{temp}')
        except Exception as err:
            print(f'SHT41 error:{err}')

if __name__ == '__main__':
    main()