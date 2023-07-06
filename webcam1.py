from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import csv

# Loading the YOLOV8 model for license plate detection
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Accessing the  webcam
cap = cv2.VideoCapture(0)

# Initializing the EasyOCR reader
reader = easyocr.Reader(['en'],gpu=False)

# To store results in  list
results = []

# Minimum confidence score for displaying the OCR result to improve accuracy
min_confidence = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting the license plates
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        if class_id == 0:  # Checking if the detected object is a license plate

            # Cropping the  license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            # Preprocessing the license plate image for EasyOCR
            license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_thresh = cv2.threshold(license_plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            license_plate_preprocessed = cv2.cvtColor(license_plate_thresh, cv2.COLOR_GRAY2BGR)

            # Performing  OCR on license plate
            ocr_results = reader.readtext(license_plate_preprocessed)

            if len(ocr_results) > 0:
                confidence = ocr_results[0][2]
                license_plate_text = ocr_results[0][1]

                if confidence > min_confidence and len(license_plate_text) >= 2:
                    # Drawing bounding box and text on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"License Plate: {license_plate_text}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Appending the results to list if license plate is recognized
                    results.append([license_plate_text, x1, y1, x2, y2])

    # Displaying the frame
    cv2.imshow("Webcam", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Save results to CSV file if any license plates are recognized
if len(results) > 0:
    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['License Plate', 'X1', 'Y1', 'X2', 'Y2'])
        writer.writerows(results)
