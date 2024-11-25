import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from keras.models import load_model

# Load pre-trained model for number plate recognition (use a suitable model)
plate_model = load_model('your_plate_recognition_model.h5')  # Replace with your model

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="parking_db"
)
cursor = db.cursor()

# Function to recognize number plate from the frame
def recognize_number_plate(frame):
    # Preprocess the frame for the model (e.g., resize, normalize)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize image

    # Predict number plate using TensorFlow model
    number_plate = plate_model.predict(image)
    return number_plate

# Function to check parking space availability from the database
def check_parking_space():
    cursor.execute("SELECT * FROM parking_spaces WHERE status='available' LIMIT 1")
    parking_space = cursor.fetchone()
    return parking_space

# Function to update parking space status after exit
def update_parking_space(parking_id):
    cursor.execute("UPDATE parking_spaces SET status='available' WHERE id=%s", (parking_id,))
    db.commit()

# OpenCV setup for video capture (camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame for visualization
    cv2.imshow("Parking System", frame)

    # Simulate entering the parking lot
    if cv2.waitKey(1) & 0xFF == ord('e'):  # Press 'e' to simulate entry
        number_plate = recognize_number_plate(frame)
        print(f"Detected Number Plate: {number_plate}")

        parking_space = check_parking_space()
        if parking_space:
            print(f"Parking Available at Location: {parking_space[1]}")
            cursor.execute("UPDATE parking_spaces SET status='occupied' WHERE id=%s", (parking_space[0],))
            db.commit()
        else:
            print("No Parking Space Available")

    # Simulate exiting the parking lot
    elif cv2.waitKey(1) & 0xFF == ord('x'):  # Press 'x' to simulate exit
        number_plate = recognize_number_plate(frame)
        print(f"Detected Number Plate: {number_plate}")

        cursor.execute("SELECT * FROM parking_spaces WHERE status='occupied' LIMIT 1")
        occupied_space = cursor.fetchone()
        if occupied_space:
            update_parking_space(occupied_space[0])
            print(f"Parking Space at Location {occupied_space[1]} is now Available")

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
db.close()
