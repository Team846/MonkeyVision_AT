import requests
import cv2
import numpy as np
import random

def download_and_open_image(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 

        image_data = np.asarray(bytearray(response.content), dtype=np.uint8)

        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        nm = "".join([str(random.randint(0, 9)) for i in range(9)])

        cv2.imwrite(f"compar/{nm}.jpg", image)

        if image is None:
            print("Failed to decode the image.")
            return

        cv2.imshow("Downloaded Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the image: {e}")


url = "http://10.8.46.7:5801/get_frame_stack"


download_and_open_image(url)
