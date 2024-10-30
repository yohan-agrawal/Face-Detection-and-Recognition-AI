import cv2
import face_recognition
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        self.root.geometry("800x600")

        self.video_capture = None

        self.label = Label(root)
        self.label.pack(padx=10, pady=10)
        
        Button(root, text="Upload Image", command=self.upload_image).pack(side=LEFT, padx=5)
        Button(root, text="Start Video", command=self.start_video).pack(side=LEFT, padx=5)
        Button(root, text="Stop Video", command=self.stop_video).pack(side=LEFT, padx=5)

        self.known_face_encodings = []
        self.known_face_names = []
        
        self.load_sample_face_data()

    def load_sample_face_data(self):
        """Load a sample face encoding for face recognition testing."""
        try:
            image = face_recognition.load_image_file("sample_image.jpg")
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append("Detected person")
        except Exception as e:
            messagebox.showerror("Error", f"Sample face image could not be loaded: {e}")

    def upload_image(self):
        """Handle the uploading and recognition of a single image."""
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        try:
            image = cv2.imread("C:\AI PROJECTS\sample_image.jpg")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_image)
            encodings = face_recognition.face_encodings(rgb_image, faces)

            for (top, right, bottom, left), face_encoding in zip(faces, encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    name = self.known_face_names[matches.index(True)]

                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.display_image(img)
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {e}")

    def start_video(self):
        """Start real-time face recognition via webcam."""
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not access the camera.")
                self.video_capture = None
                return
        self.detect_faces_in_video()

    def stop_video(self):
        """Stop the webcam feed and clear the display."""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.label.config(image="")

    def detect_faces_in_video(self):
        """Capture video frames and perform face recognition."""
        if self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret:
                self.stop_video()
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    name = self.known_face_names[matches.index(True)]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.display_image(img)
            
            self.root.after(10, self.detect_faces_in_video)

    def display_image(self, img):
        """Display an image in the Tkinter label."""
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()