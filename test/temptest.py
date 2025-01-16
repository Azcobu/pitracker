import serial

PORT = '/dev/ttyACM0'
BAUD_RATE = 9600  # Default baud rate for many devices

try:
    # Open the serial port
    with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser:
        print(f"Connected to {PORT}")
        
        while True:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            if line:  # If the line isn't empty, print it
                print(line)
except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("Program terminated.")
