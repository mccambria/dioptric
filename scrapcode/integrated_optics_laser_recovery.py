import serial

laser = serial.Serial("COM10", 115200)
# laser = serial.Serial(
#     "COM13",
#     9600,
#     serial.EIGHTBITS,
#     serial.PARITY_NONE,
#     serial.STOPBITS_ONE,
# )

try:
    # print(laser.write(b"e 1"))
    # print(laser.write(b"r i"))
    # print(laser.read_all())
    # print(laser.write(b"NM?"))
    # print(laser.read_all())
    # print(laser.write("ID?".encode()))
    # cmd = "0s1".encode()
    cmd = "r i".encode()
    # cmd = "ID?".encode()
    # cmd = "NM?".encode()
    laser.write(cmd)
    print(laser.readline())

finally:
    laser.close()
