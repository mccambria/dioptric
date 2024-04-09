import serial.tools.list_ports
import serial
import logging
import time

class SLMServer:
    def __init__(self, vid, pid, serial_number):
        self.vid = vid
        self.pid = pid
        self.serial_number = serial_number
        self.controller = None

    def find_serial_port(self):
        # Iterate over available serial ports to find the one matching the provided USB device
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.vid == self.vid and port.pid == self.pid and port.serial_number == self.serial_number:
                return port.device
        return None

    def init_server(self):
        serial_port = self.find_serial_port()
        if serial_port:
            try:
                self.controller = serial.Serial(
                    serial_port,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=2,
                )
                time.sleep(0.1)
                self.controller.flush()
                time.sleep(0.1)
                logging.info("Initialization complete")
                return "Initialization successful"
            except Exception as e:
                logging.error("Failed to initialize serial device: {}".format(e))
                return "Initialization failed: {}".format(e)
        else:
            logging.error("Serial port not found for the specified USB device.")
            return "Serial port not found"

    def close_server(self):
        if self.controller:
            self.controller.close()
            logging.info("Serial device closed")

if __name__ == "__main__":
    # USB device identifier components
    vid = 0x10C4  # Vendor ID
    pid = 0xEA60  # Product ID
    serial_number = '00429430'  # Serial number

    # Initialize the server
    server = SLMServer(vid, pid, serial_number)
    init_result = server.init_server()
    print(init_result)  # Print the initialization result

    # Perform other operations with the serial device if needed

    # Close the server when done
    # server.close_server()

# import serial
# import logging
# import time

# class SLMServer:
#     def __init__(self, device_id):
#         self.device_id = device_id
#         self.controller = None

#     def init_server(self):
#         try:
#             self.controller = serial.Serial(
#                 self.device_id,
#                 # baudrate=115200,
#                 bytesize=serial.EIGHTBITS,
#                 parity=serial.PARITY_NONE,
#                 stopbits=serial.STOPBITS_ONE,
#                 timeout=2,
#             )
#             time.sleep(0.1)
#             self.controller.flush()
#             time.sleep(0.1)
#             logging.info("Initialization complete")
#             return "Initialization successful"
#         except Exception as e:
#             logging.error("Failed to initialize serial device: {}".format(e))
#             return "Initialization failed: {}".format(e)

#     def close_server(self):
#         if self.controller:
#             self.controller.close()
#             logging.info("Serial device closed")

# if __name__ == "__main__":
#     # Set the path of the serial device
#     device_id = 429430

#     # device_id = '/dev/ttyUSB0'  # Adjust this according to your device
#     print(device_id)
#     # Initialize the server
#     server = SLMServer(device_id)
#     init_result = server.init_server()
#     print(init_result)  # Print the initialization result

#     # Perform other operations with the serial device if needed

#     # Close the server when done
#     # server.close_server()
