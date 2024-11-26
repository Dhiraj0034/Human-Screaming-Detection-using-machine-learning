import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pyaudio
import wave

# SMTP Code
def SMTP(probability):
    import smtplib
    print('\nSMTP')
    HOST = "smtp.gmail.com"
    PORT = 587

    FROM_EMAIL = "abc9@gmai.com"
    PASSWORD = '**********' # go to your gmail setting and create app password and enter that password key here

    TO_EMAIL = 'j424@gmai.com'
    subject= 'Human Scream Detection Project'
    msg=f"Scream detected with probability: {probability:.2f}"

    MESSAGE = """Subject: {}
    {}""".format(subject,msg)

    smtp = smtplib.SMTP(HOST, PORT)

    status_code, response = smtp.ehlo()
    print(f"[*] Echoing the server: {status_code} {response}")

    status_code, response = smtp.starttls()
    print(f"[*] Starting TLS connection: {status_code} {response}")

    status_code, response = smtp.login(FROM_EMAIL, PASSWORD)
    print(f"[*] Logging in: {status_code} {response}")

    smtp.sendmail(FROM_EMAIL, TO_EMAIL, MESSAGE)
    print('\nEmail was successfully sent !')
    smtp.quit()



# Function to extract features from audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        messagebox.showerror("Error", f"Error encountered while parsing file: {file_name}")
        return None

# Function to load data
def load_data(data_dir):
    features, labels = [], []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            class_label = 1 if 'scream' in file.lower() else 0
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(class_label)
    return np.array(features), np.array(labels)

# Function to record audio
def record_audio(filename, duration=2, fs=22050):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    print("Recording...")
    frames = []
    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to predict scream
def predict_scream(model, file_name):
    features = extract_features(file_name)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        prob = model.predict_proba(features)
        return prediction[0], prob[0][1]
    else:
        return None, None

# Load data
data_dir = 'Assets/positive'
X, y = load_data(data_dir)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = SVC(kernel='linear', probability=True)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Tkinter GUI
class ScreamDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Scream Detection")
        self.master.geometry("800x600")

        # Load background image
        self.bg_image = tk.PhotoImage(file="1.png")  # Replace with the path to your image file

        # Create canvas for full window background
        self.canvas = tk.Canvas(self.master, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.background = self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

        # Add resizing functionality for the window background
        self.master.bind("<Configure>", self.resize_background)

        # Frame for widgets with a background image
        self.frame = tk.Frame(self.master, relief="groove", bd=5)
        self.frame.place(relx=0.5, rely=0.5, relwidth=0.8, relheight=0.8, anchor="center")

        # Canvas for frame background
        self.frame_canvas = tk.Canvas(self.frame, width=1000, height=1000)
        self.frame_canvas.pack(fill="both", expand=True)

        # Load and set frame background image
        self.frame_bg_image = tk.PhotoImage(file="1.png")  # Replace with the path to your image
        self.frame_background = self.frame_canvas.create_image(0, 0, image=self.frame_bg_image, anchor="nw")

        # Add widgets to the frame
        self.title_label = ttk.Label(self.frame_canvas, text="Scream Detection App", font=("Helvetica", 18, "bold"), foreground="darkblue")
        self.title_label.pack(pady=20)

        # Record Button
        self.record_button = ttk.Button(self.frame_canvas, text="Record Audio", command=self.record_audio, width=20)
        self.record_button.pack(pady=10)

        # Predict Button
        self.predict_button = ttk.Button(self.frame_canvas, text="Predict Scream", command=self.predict_scream, width=20)
        self.predict_button.pack(pady=10)

        # Select Audio Button
        self.select_button = ttk.Button(self.frame_canvas, text="Select Audio", command=self.select_audio, width=20)
        self.select_button.pack(pady=10)

        # Test Selected Audio Button
        self.test_button = ttk.Button(self.frame_canvas, text="Test Selected Audio", command=self.test_selected_audio, width=20)
        self.test_button.pack(pady=10)

        # Result Label
        self.result_label = ttk.Label(self.frame_canvas, text="", font=("Helvetica", 14, "italic"), foreground="green")
        self.result_label.pack(pady=20)

        # SMTP Label
        self.smtp_label = ttk.Label(self.frame_canvas, text="", font=("Helvetica", 14, "italic"), foreground="red")
        self.smtp_label.pack(pady=20)

        # Accuracy Label
        self.accuracy_label = ttk.Label(self.frame_canvas, text=f"Accuracy: {accuracy:.2f}", font=("Helvetica", 14), foreground="purple")
        self.accuracy_label.pack(pady=10)

        # Report Label (Text box)
        self.report_label = tk.Text(self.frame_canvas, height=10, width=80, font=("Courier", 10), bg="lightgrey", relief="sunken", bd=2)
        self.report_label.pack(pady=10)
        self.report_label.insert(tk.END, report)
        self.report_label.config(state=tk.DISABLED)

        # Selected File
        self.selected_file = None

    def resize_background(self, event):
        """Resize the window and frame backgrounds dynamically."""
        width = event.width
        height = event.height
        self.canvas.config(width=width, height=height)
        self.bg_resized = self.bg_image.subsample(2, 2)  # Adjust scaling as needed
        self.canvas.itemconfig(self.background, image=self.bg_image)

    def record_audio(self):
        filename = 'output.wav'
        record_audio(filename)
        messagebox.showinfo("Record Audio", "Audio recorded successfully.")

    def predict_scream(self):
        filename = 'output.wav'
        prediction, probability = predict_scream(model, filename)
        if prediction is not None:
            if prediction == 1:
                self.result_label.config(text=f"Scream detected with probability: {probability:.2f}")
            else:
                self.result_label.config(text=f"No scream detected. Probability of scream: {probability:.2f}")
        else:
            messagebox.showerror("Predict Scream", "Error occurred during prediction.")

    def select_audio(self):
        self.selected_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.selected_file:
            messagebox.showinfo("Select Audio", f"Selected file: {self.selected_file}")

    def test_selected_audio(self):
        if self.selected_file:
            prediction, probability = predict_scream(model, self.selected_file)
            if prediction is not None:
                if prediction == 1:
                    self.result_label.config(text=f"Scream detected with probability: {probability:.2f}")
                    SMTP(probability)
                    self.smtp_label.config(text=f"Email was successfully sent!")
                else:
                    self.result_label.config(text=f"No scream detected. Probability of scream: {probability:.2f}")
            else:
                messagebox.showerror("Test Selected Audio", "Error occurred during prediction.")
        else:
            messagebox.showwarning("Test Selected Audio", "No audio file selected. Please select an audio file first.")

def main():
    root = tk.Tk()
    app = ScreamDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
