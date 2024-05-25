import os
import cv2
import numpy as np
import time
import pyttsx3
from tensorflow.keras.models import load_model
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Function to map gesture codes to gesture names
def mapper(gesture_code):
    
    alphabet = "ABCDEFGHIJKLMNNOPQRSTUVWXYZ"
    if 0 <= gesture_code < len(alphabet):
        return alphabet[gesture_code]
    else:
        return None

# Function to open application based on gesture
def open_application(gesture):
    if gesture =="A":
        os.startfile("C:\\Program Files\\Mozilla Firefox\\firefox.exe")
        return "Mozilla Firefox"
    
    elif gesture == "D":      
        os.system("TASKKILL /F /IM firefox.exe")
        return "Closing Firefox"
    
    elif gesture == "B":
        os.startfile(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
        return "Google Chrome"
    
    elif gesture == "F":
        os.system("TASKKILL /F /IM chrome.exe")
        return "Closing Chrome"
    
    elif gesture == "W":
        os.startfile(r"C:\Windows\notepad.exe")
        return "Opening notepad"

    elif gesture == "N":
         os.system("TASKKILL /F /IM notepad.exe")
         return "Closing notepad"
    
    elif gesture == "C":
        os.startfile(r"C:\Program Files\Git\git-bash.exe")
        return "Opening git-bash"

    elif gesture == "K":
         os.system("TASKKILL /F /IM POWERPNT.EXE")
         return "Closing powerpoint"
    
    elif gesture == "L":
        os.startfile(r"C:\Program Files (x86)\Microsoft Office\Office14\POWERPNT.EXE")
        return "Opening powerpoint"

    elif gesture == "G":
        change_volume("increase")
        return "Increasing volume"
    
    elif gesture == "H":
        change_volume("decrease")
        return "Decreasing volume"
    
    elif gesture == "R":
        toggle_mute()
        return "Toggling mute/unmute"
    

def change_volume(direction):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    # Get the current volume range
    current_volume = volume.GetMasterVolumeLevelScalar()

    # Increase or decrease the volume based on the direction
    if direction == "increase":
        volume.SetMasterVolumeLevelScalar(min(1.0, current_volume + 0.6), None)
    elif direction == "decrease":
        volume.SetMasterVolumeLevelScalar(max(0.0, current_volume - 0.6), None)

def toggle_mute():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Toggle mute/unmute
    is_muted = volume.GetMute()
    volume.SetMute(not is_muted, None)


# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
engine.say("Welcome! Let's get started.")
engine.say("Show gestures to Control computer.")
engine.runAndWait()

# Loading the model
model = load_model("hack36.h5")

# Open webcam
video = cv2.VideoCapture(0)

# Constants for gesture recognition
start_time = time.time()
stabilization_delay = 0  # 5 seconds
count_threshold = 15 ### Adjust this threshold as needed, 0 means immediate recognition
app_opening_alphabets = "ADBFGHR"

# Variables to track continuous predictions
current_prediction = None
current_prediction_count = 0
app_opened = False
confirmation_requested = False
s_u_count = 0

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    # If an application is open and confirmation is requested, wait for gesture input from the user
    if confirmation_requested:
        ret, frame = video.read()
        key = cv2.waitKey(1)

        if elapsed_time >= stabilization_delay:
            start_time = time.time()  # Reset start time

            # Capture frame
            ret, frame = video.read()
            if not ret:
                continue

            # Determine the center coordinates for the ROI
            frame_height, frame_width, _ = frame.shape
            center_x = frame_width // 2
            center_y = frame_height // 2
            roi_size = 200  # Size of the ROI
            roi_half_size = roi_size // 2
            roi_x1 = center_x - roi_half_size
            roi_y1 = center_y - roi_half_size
            roi_x2 = center_x + roi_half_size
            roi_y2 = center_y + roi_half_size
            roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 200
            # Crop region of interest (ROI)
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            #cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.putText(frame, "Predicting...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Webcam", frame)

            cv2.imshow("ROI", roi)
            ###Show the ROI in a separate frame

            bi=cv2.GaussianBlur(roi, (1, 1), 0)
            # Convert the image to RGB
            rgb_image = cv2.cvtColor(bi, cv2.COLOR_BGR2RGB)
            # Split the RGB channels
            r, g, b = cv2.split(rgb_image)   
            # Apply adaptive thresholding to each channel
            _, thresholded_r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresholded_g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresholded_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ti = cv2.merge([thresholded_r, thresholded_g, thresholded_b])
            ###resize roi
            
            roi_resized = cv2.resize(ti, (224, 224))

            # Normalize pixel values
            roi_normalized = roi_resized / 255.0

            # Expand dimensions to match model input shape
            roi_final = np.expand_dims(roi_normalized, axis=0)

            # Make prediction using the model
            prediction = model.predict(roi_final)
            gesture_code = np.argmax(prediction[0])
            user_gesture = mapper(gesture_code)

            # Check if the user wants to stay on the current application or find another one
            if user_gesture == "S":
                # User wants to stay on the current application
                engine.say("Staying on the current application. Thank you!")
                engine.runAndWait()
                video.release()
                cv2.destroyAllWindows()
                exit()
            elif user_gesture == "U":
                # User wants to find another application
                engine.say("Let's do another action  for you.")
                engine.runAndWait()
                app_opened = False
                confirmation_requested = False
                continue

    # If enough time has passed for stabilization, proceed with prediction
    if elapsed_time >= stabilization_delay:
        start_time = time.time()  # Reset start time

        # Capture frame
        ret, frame = video.read()
        if not ret:
            continue

        # Determine the center coordinates for the ROI
        frame_height, frame_width, _ = frame.shape
        center_x = frame_width // 2
        center_y = frame_height // 2
        roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 200

        # Crop region of interest (ROI)
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Draw a rectangle around the ROI
        #cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

        # Display "Predicting..." text on the frame
        cv2.putText(frame, "Predicting...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Webcam", frame)

        ###Show the ROI in a separate frame
        cv2.imshow("ROI", roi)

        bi=cv2.GaussianBlur(roi, (1, 1), 0)
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(bi, cv2.COLOR_BGR2RGB)
        # Split the RGB channels
        r, g, b = cv2.split(rgb_image)   
        # Apply adaptive thresholding to each channel
        _, thresholded_r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresholded_g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresholded_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ti = cv2.merge([thresholded_r, thresholded_g, thresholded_b])

        ###resize roi      
        roi_resized = cv2.resize(ti, (224, 224))

        # Normalize pixel values
        roi_normalized = roi_resized / 255.0

         # Expand dimensions to match model input shape
        roi_final = np.expand_dims(roi_normalized, axis=0)

        # Make prediction using the model
        prediction = model.predict(roi_final)
        gesture_code = np.argmax(prediction[0])
        user_gesture = mapper(gesture_code)

        # Check if the predicted gesture indicates an application opening
        if user_gesture and user_gesture in app_opening_alphabets:
            # If threshold is zero, open the application immediately
            if count_threshold == 0:
                app_name = open_application(user_gesture)
                engine.say(f"As gesture detected {app_name}.")
                engine.runAndWait()

                # Prompt for confirmation
                engine.say("Do you want to continue with this action? Show 'Y' to stay or 'V' to find another one.")
                engine.runAndWait()
                confirmation_requested = True

            else:
                # Increment the count for the current predicted gesture
                if user_gesture == current_prediction:
                    current_prediction_count += 1
                else:
                    current_prediction = user_gesture
                    current_prediction_count = 1
                
                # Check if the count exceeds the threshold for opening applications
                if current_prediction_count >= count_threshold:
                    # Open the application
                    app_name = open_application(user_gesture)
                    engine.say(f"Opening {app_name}.")
                    engine.runAndWait()

                    # Prompt for confirmation
                    engine.say("Do you want to continue with this action? Show 'Y' to stay  or 'V' to find another one.")
                    engine.runAndWait()
                    confirmation_requested = True
                    s_u_count = 0  

    
    if app_opened and (user_gesture == "Y" or user_gesture == "V"):
        s_u_count += 1
        if s_u_count >= count_threshold:
            if user_gesture == "Y":
                # User wants to stay on the current application
                engine.say("Staying on the current application. Thank you!")
                engine.runAndWait()
                video.release()
                cv2.destroyAllWindows()
                exit()
            elif user_gesture == "V":
                # User wants to find another application
                engine.say("Let's do another action for you.")
                engine.runAndWait()
                app_opened = False
                s_u_count = 0  # Reset the count for 'Y' and 'V' gestures
                continue

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
