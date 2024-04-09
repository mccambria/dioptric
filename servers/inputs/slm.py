# -*- coding: utf-8 -*-
"""
Input server for the amplified photodiode. Communicates via the DAQ.

Created on Thu Mar 20 08:52:34 2020

@author: mccambria

### BEGIN NODE INFO
[info]
name = photodiode
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

# from labrad.server import LabradServer
# from labrad.server import setting
# from twisted.internet.defer import ensureDeferred
# import numpy
# import nidaqmx
# import socket
# import logging
# from utils import common


import socket
import struct
import numpy as np
import bz2
import zlib
import gzip

class SLM_Server:
    def __init__(self, host='', port=9999):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.connected = False
    
    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")
    
    def accept_connection(self):
        print("Waiting for a connection...")
        self.client_socket, client_address = self.server_socket.accept()
        print(f"Connected to {client_address}")
        self.connected = True
    
    def receive_image(self):
        if not self.connected:
            print("No client connected.")
            return
        
        payload_size = struct.calcsize("i")
        data = b''
        
        # Receiving the message size
        while len(data) < payload_size:
            data += self.client_socket.recv(4096)
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("i", packed_msg_size)[0]
        
        # Receiving the image data
        while len(data) < msg_size:
            data += self.client_socket.recv(4096)
        
        frame_data = data[:msg_size]
        
        return frame_data
    
    def decompress_image(self, frame_data, compression):
        if compression == 'bz2':
            frame_data = bz2.decompress(frame_data)
        elif compression == 'zlib':
            frame_data = zlib.decompress(frame_data)
        elif compression == 'gzip':
            frame_data = gzip.decompress(frame_data)
        
        return frame_data
    
    def update_slm(self, image_data):
        # Replace this with your SLM update logic
        print("Updating SLM with received image data.")
        image = np.frombuffer(image_data, dtype=np.uint8)
        # Perform operations to display the image on the SLM
        
    def close(self):
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()

# Example usage:
if __name__ == "__main__":
    server = SLM_Server()
    server.start()
    
    try:
        server.accept_connection()
        while True:
            image_data = server.receive_image()
            if image_data:
                decompressed_data = server.decompress_image(image_data, compression='zlib')  # Adjust compression as needed
                server.update_slm(decompressed_data)
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        server.close()


# __server__ = SLM()

# if __name__ == "__main__":
#     from labrad import util

#     util.runServer(__server__)
