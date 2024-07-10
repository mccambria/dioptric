import numpy as np
from utils import tool_belt as tb
from labrad.types import Value
import matplotlib.pyplot as plt

def cam_plot(cam):
    cam.set_exposure(Value(0.0001, 's'))
    img = cam.get_image()
    if img is None:
        print("No image acquired")
        return

    # Ensure img is converted to the appropriate dtype if needed
    img = img.astype(np.uint8)  # Example conversion to uint8

    # Plot the result
    plt.figure(figsize=(12, 9))
    plt.imshow(img, cmap='gray')  # Added cmap for grayscale image
    plt.show()

try:
    cam = tb.get_server_thorcam()
    cam.info()
    print(f"Camera Properties: {cam.info()}")
    cam_plot(cam)
finally:
    print("Closing")
    # Close the camera
    cam.close()
    # Then close the SDK
    cam.close_sdk() 

output = f"Camera Properties: {cam.info()}"
print(output)
