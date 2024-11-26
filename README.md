# Human Scream Detection ML Python

Human Scream Detection is a Python project that records audio using the PyAudio module and detects human screams in the audio. If a scream is detected, the program sends an alert email to a designated helpline. The project includes a graphical user interface (GUI) created with the Tkinter module, offering four main functionalities: recording audio, predicting screams in recorded audio, selecting audio from local files, and testing selected files.

![ScreenShot of GHI of this project](https://github.com/Kunal-kawate/Human_Scream_Detection_ML_Python/blob/main/ScreenShot/GUI.png)

## Features

- **Record Audio**: Capture audio in real-time using your computer's microphone.
- **Predict Scream in Recorded Audio**: Analyze recorded audio to detect human screams.
- **Select Audio from Local Files**: Choose pre-recorded audio files from local storage for analysis.
- **Test Selected File**: Test an audio file from local storage for the presence of human screams and trigger an alert if detected.
- NOTE- use Negative Folder sounds to testing scream , which are negative for scream detection.....

## Project Structure

- **main.py**: The main Python program that handles audio recording, scream detection, and the GUI.
- **requirements.txt**: A list of required Python modules for the project.
- **asset**: A folder containing human screaming sounds used to create the dataset and predict human voices.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Kunal-kawate/Human_Scream_Detection_ML_Python.git
    cd Human_Scream_Detection_ML_Python
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

4. Install the required modules:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main program:

    ```bash
    python main.py
    ```

2. Use the GUI to interact with the program:
    - **Record Audio**: Click "Record Audio" to start recording. Stop the recording to save the audio file.
    - **Predict Scream in Recorded Audio**: Click "Predict Scream" to analyze the recorded audio for screams.
    - **Select Audio from Local Files**: Click "Select Audio" to choose an audio file from your local storage.
    - **Test Selected File**: Click "Test File" to analyze the selected audio file for screams and send an alert if detected.

## Asset Folder

The `asset` folder contains human screaming sounds that are used to create the dataset and for predicting human screams in the audio. These sounds are essential for training and testing the scream detection algorithm.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

**Note**: Ensure you have an active internet connection and the necessary email configurations set up in `main.py` for the email alert feature to work.

---

**Author**: Kunal Kawate

**GitHub Repository**: [Human_Scream_Detection_ML_Python](https://github.com/Kunal-kawate/Human_Scream_Detection_ML_Python)
