import serial
import time
import csv

# === SETTINGS ===
port = 'COM16'            # Replace with your COM port
baudrate = 115200
input_csv = 'command_test.csv'
output_csv = 'received_data_test.csv'

# === INIT SERIAL ===
ser = serial.Serial(port, baudrate, timeout=1)
time.sleep(2)
ser.reset_input_buffer() 

# === Wait for Arduino "ready" ===
while True:
    if ser.in_waiting:
        line = ser.readline().decode().strip()
        print("Arduino:", line)
        if "ready" in line.lower():
            break

# === Open input/output files ===
with open(input_csv, 'r') as fin, open(output_csv, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    for row in reader:
        # ——1）发送当前行——
        data_str = ",".join(row)
        print("Sending:", data_str)
        ser.write((data_str + "\n").encode())

        # ——2）在这里不断 read，直到拿到 4 段合法数据才 break——
        while True:
            response = ser.readline().decode().strip()
            if not response:
                # 超时或空行，继续等
                continue

            print("Raw response:", repr(response))
            parts = response.split(',')

            if len(parts) == 4 and all(p != "" for p in parts):
                # ✔️ 收到合法回应，写入 CSV
                writer.writerow(parts)
                break
            else:
                # 🛑 收到非目标数据，丢弃并继续等
                print("→ 忽略这一行，不发送下一条命令")

        # ——此处才会进入下一次 for loop，发送下一行——
