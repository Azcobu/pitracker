from datetime import datetime, timedelta
import random

def write_testfile(outstrs):
    with open('testdata.csv', 'w', encoding='utf-8') as outfile:
        outfile.write('timestamp,temperature\n')
        for x in outstrs:
            outfile.write(x + '\n')

def main():
    temp = 25
    temp_variation = 0.5

    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = [(start_time + timedelta(minutes=i)).isoformat(sep=' ') for i in range(1440)]
    outstrs = []

    # Print the timestamps
    for timestamp in timestamps:
        tempvar = random.randint(0, 2)
        if tempvar == 0:
            temp -= temp_variation
        if tempvar == 2:
            temp += temp_variation

        temp = round(temp, 1)
        outstrs.append(f'{timestamp},{temp}')

    write_testfile(outstrs)

if __name__ == '__main__':
    main()