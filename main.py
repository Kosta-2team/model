import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
import serial
import cv2
from ultralytics import YOLO
import easyocr
import pymongo
import time
import numpy as np
import torch
from datetime import datetime
import re
import requests

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variables
mongodb_uri = os.getenv("MONGODB_URI")

# Initialize serial connection
try:
    ser = serial.Serial('COM3', 9600)  
    print("Serial connection established")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

# Check if GPU is available
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

# Initialize external webcam
def initialize_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("External camera failed. Trying default camera (index 0).")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open any camera")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# Initialize MongoDB connection
client = pymongo.MongoClient(mongodb_uri)
db = client["carParking"]
collection = db["entries"]

# Load YOLO models
try:
    print("Loading YOLO models...")
    coco_model = YOLO('yolov8s.pt')
    np_model = YOLO('best.pt')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    coco_model, np_model = None, None

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=gpu_available)

# Initialize column counter
column_counter = 1

def detect_vehicles(frame):
    detections = coco_model(frame)[0]
    vehicle_bounding_boxes = []
    vehicles = [2, 3, 5, 7]
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles and score > 0.5:
            vehicle_bounding_boxes.append([x1, y1, x2, y2, score])
    return vehicle_bounding_boxes

def detect_license_plates(roi):
    try:
        height, width = roi.shape[:2]
        aspect_ratio = width / height
        new_width = 640
        new_height = int(new_width / aspect_ratio)
        roi_resized = cv2.resize(roi, (new_width, new_height))

        license_plates = np_model(roi_resized)[0]
        plates = []
        for license_plate in license_plates.boxes.data.tolist():
            plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
            scale_x = width / new_width
            scale_y = height / new_height
            orig_plate_x1 = int(plate_x1 * scale_x)
            orig_plate_y1 = int(plate_y1 * scale_y)
            orig_plate_x2 = int(plate_x2 * scale_x)
            orig_plate_y2 = int(plate_y2 * scale_y)
            plate = roi[orig_plate_y1:orig_plate_y2, orig_plate_x1:orig_plate_x2]
            plates.append((plate, orig_plate_x1, orig_plate_y1, orig_plate_x2, orig_plate_y2, plate_score))
        return plates
    except Exception as e:
        print(f"Error in license plate detection: {e}")
        return []

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    resized = cv2.resize(morph, (300, 100))
    return resized

def read_license_plate(license_plate_crop):
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None
    
    # Apply multiple preprocessing techniques
    methods = [
        lambda x: x,  # Original image
        lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
        lambda x: cv2.threshold(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        lambda x: cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
    ]
    
    for method in methods:
        processed_plate = method(license_plate_crop)
        detections = reader.readtext(processed_plate)
        
        for detection in detections:
            bbox, text, score = detection
            if score > 0.5:
                # Filter out non-alphanumeric characters
                text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
                return text, score
    
    return None, None

def capture_and_process_image(cap):
    global column_counter
    print("Capturing image...")
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return

    cv2.imwrite("debug_raw_frame.jpg", frame)
    vehicle_bounding_boxes = detect_vehicles(frame)
    if not vehicle_bounding_boxes:
        print("No vehicles detected")
        return

    for bbox in vehicle_bounding_boxes:
        x1, y1, x2, y2, score = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        plates = detect_license_plates(roi)
        if not plates:
            print("No license plates detected in this vehicle")
            continue

        for plate, plate_x1, plate_y1, plate_x2, plate_y2, plate_score in plates:
            cv2.rectangle(frame, (int(x1 + plate_x1), int(y1 + plate_y1)), (int(x1 + plate_x2), int(y1 + plate_y2)), (255, 0, 0), 2)
            np_text, np_score = read_license_plate(plate)
            if np_text:
                print(f"License Plate: {np_text} (Confidence: {np_score})")
                cv2.putText(frame, np_text, (int(x1 + plate_x1), int(y1 + plate_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"car_plate_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved as {filename}")
                existing_entry = collection.find_one({"numPlate": np_text, "outTime": '0'})
                if existing_entry:
                    try:
                        collection.update_one(
                            {"_id": existing_entry["_id"]},
                            {"$set": {"outTime": timestamp, "rate": 10000, "etc": None}}
                        )
                        print(f"Updated outTime for license plate {np_text}")
                        
                        api_url = "http://192.168.0.18:7003/api/notify"  # API 서버 URL
                        payload = {
                            "id": str(existing_entry["_id"]),
                            "numPlate": np_text,
                            "inTime": existing_entry["inTime"],
                            "outTime": timestamp,
                            "rate": 10000,
                        }
                        response = requests.post(api_url, json=payload)
                        if response.status_code == 200:
                            print("Notified C# API server successfully.")
                        else:
                            print(f"Failed to notify C# API server: {response.status_code}, {response.text}")
                                        
                    except Exception as e:
   
                        print(f"Error updating exit time: {e}")
                else:                    
                    data = {
                        "column": column_counter,
                        "numPlate": np_text,
                        "inTime": timestamp,
                        "outTime": '0',
                        "rate": 10000,
                        "totalCost": None,
                        "minsParked": None,
                        "etc": None,
                    }
                try:
                        result = collection.insert_one(data)
                        print(f"Inserted new license plate entry with ID: {result.inserted_id}")
                        column_counter += 1  # Increment column counter
                except Exception as e:
                        print(f"Error inserting to MongoDB: {e}")

                # Send data to Arduino
                try:
                    ser.write(f"{np_text}\n".encode())
                    print(f"Sent to Arduino: {np_text}")
                except Exception as e:
                    print(f"Error sending data to Arduino: {e}")

def main():
    cap = initialize_camera()
    if not cap:
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            vehicle_bounding_boxes = detect_vehicles(frame)
            for bbox in vehicle_bounding_boxes:
                x1, y1, x2, y2, score = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, "Vehicle", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Vehicle Detection", frame)

            # Check for Arduino input
            if ser.in_waiting > 0:
                arduino_input = ser.readline().decode().strip()
                print(f"Received from Arduino: {arduino_input}")  # Debugging info
                if arduino_input == "ENTRY":
                    print("Arduino triggered capture")  # Debugging info
                    capture_and_process_image(cap)

            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                capture_and_process_image(cap)
            elif key == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()