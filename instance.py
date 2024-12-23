import pixellib
from pixellib.instance import instance_segmentation
import cv2
import time
import numpy as np

# Initialize the segmentation model
segmentation_model = instance_segmentation()

# Load the pre-trained Mask R-CNN model
segmentation_model.load_model('mask_rcnn_coco.h5')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# For calculating FPS
prev_time = time.time()

# Toggle flag to switch between background modes
background_mode = False

while cap.isOpened():
    ret, frame = cap.read() # Capture a frame from the webcam
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Apply Instance Segmentation on the captured frame
    result = segmentation_model.segmentFrame(frame, show_bboxes=True)

    # The segmented frame with bounding boxes is stored in result[1]
    segmented_frame = result[1]

    # Create a black background
    black_background = np.zeros_like(frame)

    # Check if background_mode is enabled (True means black background mode)
    if background_mode:
        # Create the mask from the segmentation result
        mask = result[0]["masks"]
        if mask.shape[2] > 0:  # Check if any object is detected 
            combined_mask = np.any(mask, axis=2).astype(np.uint8) * 255  # Combine all masks and create a single mask
            inverted_mask = cv2.bitwise_not(combined_mask) # Invert the mask to separate foreground (objects) from background

            # Apply the mask on the frame to extract the foreground (segmented objects)
            foreground = cv2.bitwise_and(frame, frame, mask=combined_mask)
            # Apply the inverted mask on the black background to keep the background black
            background_area = cv2.bitwise_and(black_background, black_background, mask=inverted_mask)
            # Combine foreground and background areas to get the final frame with black background
            frame_with_background = cv2.add(foreground, background_area)
        else:
            frame_with_background = black_background  # If no objects, just black background
    else:
        frame_with_background = segmented_frame  # Normal view

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Overlay FPS on the frame
    cv2.putText(frame_with_background, f"FPS: {fps:.2f}", (10, 30), # Position of text (top-left)
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # Text settings (font, color, thickness)

    # Display the frame
    cv2.imshow('Instance Segmentation', frame_with_background)  

    # Keyboard controls
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('t'):  # Toggle background mode
        background_mode = not background_mode
        print("Toggled background mode:", "Black Background" if background_mode else "Normal View")
    elif key == ord('s'):  # Save frame as an image
        cv2.imwrite('segmented_frame.jpg', frame_with_background)
        print("Frame saved as segmented_frame.jpg")
    elif key == ord('p'):  # Pause
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)

# Release the video capture and close all OpenCV windows when done
cap.release()
cv2.destroyAllWindows()
