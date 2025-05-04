import cv2
import sys
from deepface import DeepFace
import webbrowser as wb


# Set up video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video device.")
    sys.exit()

# Define colors for rectangles (BGR format for OpenCV)
COLOR_ANGRY = (0, 0, 255)  # Red
COLOR_OTHER = (0, 255, 0)  # Green
RECT_THICKNESS = 2
url = 'https://youtu.be/9Fle2CP8gR0?si=FFB0UfU2DB70vd2P' #your fav calming music

# --- Main Loop ---
while True:
    angry_time =0
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False, # Don't crash if no face found
            silent=True # Suppress internal DeepFace logs for cleaner output
        )

        if isinstance(results, list) and results:
            for result in results:
                 # Check if 'region' and 'dominant_emotion' keys exist
                 # Sometimes if detection confidence is low, keys might be missing
                 if 'region' in result and 'dominant_emotion' in result:
                    face_region = result['region']
                    emotions = result['emotion'] # Get the dictionary of probabilities
                    dominant_emotion = result['dominant_emotion'] # Still useful]

                    # Get bounding box coordinates
                    x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                        # --- Custom Logic based on probabilities ---
                    angry_confidence_threshold = 50 # Example: Require > 50% confidence for 'angry'
                    is_confidently_angry = emotions.get('angry', 0) > angry_confidence_threshold

                    if is_confidently_angry:
                        angry_time +=1
                        rect_color = COLOR_ANGRY
                        emotion_text = f"Angry ({emotions.get('angry', 0):.0f}%)"

                        if angry_time >=3:
                            wb.open_new_tab(url)
                        
                    else:
                        angry_time = 0
                        rect_color = COLOR_OTHER
                        # Show dominant emotion if not confidently angry
                        emotion_text = f"{dominant_emotion} ({emotions.get(dominant_emotion, 0):.0f})"

                    # Draw the rectangle on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, RECT_THICKNESS)

                    # Put the emotion text above the rectangle
                    cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, RECT_THICKNESS)
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        pass

    cv2.imshow('Camera Feed with Emotion', frame) # Display the frame

    # --- Wait for Key Press ---
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera feed stopped.")