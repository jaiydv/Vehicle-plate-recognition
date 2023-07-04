# Vehicle-plate-recognition


https://user-images.githubusercontent.com/85070747/204129962-2b0a9a9d-8440-4f75-99bd-9700f4f868e7.mp4


This project is a vehicle plate recognition system that utilizes the YOLO model for vehicle plate detection. It is built using OpenCV, Python, TensorFlow, and Keras. The system is designed to detect license plates on vehicles, enabling automated plate recognition for various applications.



## Features

- Utilizes the YOLO (You Only Look Once) model for accurate and efficient vehicle plate detection.
- Processes images or video streams to detect and extract license plate regions.
- Provides an easy-to-use interface to input images or video streams and view the detected license plates.

## Installation

To set up the environment for running this project, follow these steps:

1. Clone the repository to your local machine or download the ZIP file.
2. Install the required dependencies by running the following command:

   ```bash
   pip install opencv-python tensorflow keras pytesseract
   ```

3. The weights are stored in anpr_best.pt file
## Usage

To use the vehicle plate recognition system, follow these steps:

1. Ensure that you have the necessary dependencies and the pre-trained YOLO weights (see Installation section).
2. Prepare the input data:
   - For image-based recognition, place the input image in the project directory.
   - For video-based recognition, set the video source (file path or camera index) in the `vid.py` file.
3. Open the `vid.py` file in a text editor.
4. Adjust any necessary configurations such as confidence threshold, input image size, etc.
5. Run the script.
6. The script will process the input data, detect license plates using the YOLO model, extract plate regions, and perform OCR to recognize the characters.
7. The detected license plates along with the recognized characters will be displayed on the screen or saved as output, depending on the configuration.

## Contributing

Contributions are welcome! If you encounter any issues or have ideas for improvements, open an issue or submit a pull request. Please adhere to the existing code style and clearly describe your changes.

## Acknowledgements

This project relies on the following libraries and resources:

- [OpenCV](https://opencv.org/): Used for image and video processing.
- [TensorFlow](https://www.tensorflow.org/): Used for deep learning and model training.
- [Keras](https://keras.io/): Used for building and training neural networks.
- [PyTesseract](https://github.com/madmaze/pytesseract): Used for OCR.

Special thanks to the contributors of these libraries and the creators of the YOLO model for their valuable work.

## Contact

If you have any questions, suggestions, or feedback, you can reach out to the project maintainer:

- Name: Jaiparkash yadav
- Email: jaipywork@gmail.com

Feel free to get in touch!



