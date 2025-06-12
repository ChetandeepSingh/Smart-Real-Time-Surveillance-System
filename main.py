import cv2
import numpy as np
import imutils
import datetime
from model import Model
from send_mail_custom_lib import EmailSender
import time
import os
import threading
import serial

from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# Serial communication setup
SERIAL_PORT = "COM7"  # Change this to your serial port
BAUD_RATE = 115200
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
except Exception as e:
    print(e)


def send_serial_data(msg):
    try:
        ser.write(msg.encode("utf-8"))
        print("Data Sent Serially")
    except Exception as e:
        print(e)


# Load the face detector and mask detector models
faceNet = cv2.dnn.readNet(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
)
maskNet = load_model("mask_detector.h5")


def detect_and_predict_mask(frame, faceNet, maskNet, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        preds = maskNet.predict(np.vstack(faces))
        return (locs, preds)
    else:
        return ([], [])

def process_video(input_video_path):

    recording_time = 30
    violence_detected_count = 0
    mask_detected_count = 0
    suspicious_count = 0

    violence_video_count = 1
    theft_video_count = 1
    gun_count = 1

    recording = False
    last_recording_time = 0
    recording_start_time = 0

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Could not open video cam")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fps = 7

    firstFrame = None
    out = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame")
            break

        # Resize frame for gun detection
        # resized_frame = imutils.resize(frame, width=500)
        confidence_threshold = 0.45 
        gun_exist = False
        results = yolo_model(frame)[0]  # Get the first result

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = results.names[cls]

            if label.lower() == "guns"and conf > confidence_threshold:  # match your training label
                gun_exist = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Weapon {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Violence detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(rgb_frame)
        label = prediction["label"]
        cv2.putText(frame, f"Scene: {label}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        conf = prediction["confidence"]
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        violence_detected = "fight" in label.title().lower()

        if violence_detected:
            violence_detected_count += 1
            threading.Thread(target=send_serial_data, args=("V\n",)).start()
        else:
            violence_detected_count = 0

        # Mask detection
        locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

        mask_detected = False  # Track if any mask is found

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            mask_detected = mask_detected or (label == "Mask")

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        if mask_detected:
            mask_detected_count += 1
            threading.Thread(target=send_serial_data, args=("T\n",)).start()
        else:
            mask_detected_count = 0

        if gun_exist:
            gun_count += 1
            threading.Thread(target=send_serial_data, args=("G\n",)).start()
        else:
            gun_count = 0

        # Recording logic
        current_time = time.time()

        if (
            not recording
            and (violence_detected_count > 10 or mask_detected_count > 10)
            and (current_time - last_recording_time) >= 10
        ):
            os.makedirs("Output Video", exist_ok=True)
            # Start recording
            recording = True
            recording_start_time = current_time

            if violence_detected_count > 10:
                output_filename = (
                    f"Output Video/violence_video_{violence_video_count}.avi"
                )
                violence_recording = True
                mask_recording = False
                # Use the same fps for recording as the input video
                out = cv2.VideoWriter(
                    output_filename,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    fps,
                    (width, height),
                )
                print(f"Recording started: {output_filename}")
                violence_detected_count > 0
            if mask_detected_count > 10:
                output_filename = f"Output Video/theft_video_{theft_video_count}.avi"
                violence_recording = False
                mask_recording = True
                # Use the same fps for recording as the input video
                out = cv2.VideoWriter(
                    output_filename,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    fps,
                    (width, height),
                )
                print(f"Recording started: {output_filename}")
                mask_detected_count = 0
            
            if gun_count > 5:
                output_filename = f"Output Video/suspicious_person_with_weapon{theft_video_count}.avi"
                violence_recording = False
                mask_recording = True
                # Use the same fps for recording as the input video
                out = cv2.VideoWriter(
                    output_filename,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    fps,
                    (width, height),
                )
                print(f"Recording started: {output_filename}")
                gun_count = 0

        if recording:

            out.write(frame)

            if (current_time - recording_start_time) >= recording_time:
                # Stop recording
                recording = False
                if violence_recording:
                    violence_video_count += 1
                    last_recording_time = current_time
                    out.release()
                    print(f"Recording stopped: {output_filename}")
                    print(f"Waiting for 10 seconds before next possible recording...")
                    print("Sending Output Video on Mail")
                    threading.Thread(
                        target=sendMail, args=("Violence Detected", output_filename)
                    ).start()
                elif mask_recording:
                    theft_video_count += 1
                    last_recording_time = current_time
                    out.release()
                    print(f"Recording stopped: {output_filename}")
                    print(f"Waiting for 10 seconds before next possible recording...")
                    print("Sending Output Video on Mail")
                    threading.Thread(
                        target=sendMail, args=("Theft Detected", output_filename)
                    ).start()

        # Draw the text and timestamp on the frame
        cv2.putText(
            frame,
            datetime.datetime.now().strftime("%m-%d-%Y %H:%M%p"),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )

        if gun_exist:
            cv2.putText(
                frame,
                "Suspicious Person with Weapon",
                (0, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame, "Normal", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        # Add text based on detection
        if violence_detected:
            cv2.putText(
                frame,
                "Violence Detected",
                (0, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame, "Normal", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        label = "Theft Activity" if mask_detected else "Normal"
        color = (0, 0, 255) if mask_detected else (0, 255, 0)
        cv2.putText(frame, label, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Recording...", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    ser.close()


def sendMail(body, attachment_filename):
    email_sender = EmailSender(smtp_server, smtp_port, email_from, email_pass)
    start = time.time()
    email_sender.send_email(email_to, subject, body, attachment_filename)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed Time: {elapsed_time} seconds")


model = Model()
yolo_model = YOLO("best1.pt")

smtp_server = "smtp.gmail.com"
smtp_port = 587
email_from = "meharchetandeepsingh@gmail.com"
email_pass = ""

email_to = ["iotnc4@gmail.com"]  # Can be a string or a list
subject = "New email from Violence and Theft Detection Module"

process_video(0)