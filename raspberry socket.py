import socket
import io
from picamera2 import Picamera2

# Initialize camera
camera = Picamera2()
camera_config = camera.create_still_configuration(main={"size": (1920, 1080)})
camera.configure(camera_config)
camera.start()

# Setup socket on Raspberry Pi
raspi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
raspi_socket.bind(('0.0.0.0', 8000))  # Bind to port 8000
raspi_socket.listen(1)

def capture_image():
    stream = io.BytesIO()
    image = camera.capture_file(stream, format='jpeg')  # Capture image as JPEG
    stream.seek(0)
    return stream

print("Waiting for laptop connection...")

while True:
    conn, addr = raspi_socket.accept()  # Wait for connection from laptop
    print(f"Connected by {addr}")
   
    try:
        # Capture image from PiCamera2
        image_stream = capture_image()

        # Get image size and send to laptop
        image_size = len(image_stream.getvalue())
        conn.sendall(image_size.to_bytes(4, byteorder='big'))

        # Send image data in chunks
        conn.sendall(image_stream.read())
        print("Image sent!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()  # Close the connection
