import cv2
import numpy as np
import pytesseract
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('plates.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''CREATE TABLE IF NOT EXISTS plates
             (plate_number TEXT)''')
conn.commit()

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to capture the frame and perform number plate detection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save the captured frame
        cv2.imwrite('captured_frame.jpg', frame)

        # Perform number plate detection
        plate_img = cv2.imread('captured_frame.jpg')
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        edges = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area (assuming it's the number plate)
        if contours:
            areas = [cv2.contourArea(contour) for contour in contours]
            max_index = np.argmax(areas)
            max_contour = contours[max_index]

            # Get the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(max_contour)

            # Crop the number plate region
            plate_crop = plate_img[y:y + h, x:x + w]

            # Use Tesseract OCR to extract text from the number plate
            custom_config = r'--oem 3 --psm 6'
            plate_number = pytesseract.image_to_string(plate_crop, config=custom_config)

            # Store the plate number in the database
            cursor.execute("INSERT INTO plates (plate_number) VALUES (?)", (plate_number,))
            conn.commit()

            print("Number Plate Detected:", plate_number)

        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()
