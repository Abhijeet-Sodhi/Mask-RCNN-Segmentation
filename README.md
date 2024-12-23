# Mask-RCNN-Segmentation 🎥🎨
This project uses Mask R-CNN for real-time instance segmentation from webcam video, allowing users to toggle between two background modes: the original background or a solid black background, offering an interactive video processing experience.

## Credits 🤖
[![Instance Segmentation using Mask-RCNN with PixelLib and Python](https://www.youtube.com/watch?v=i_-ud01wFhc.jpg)](https://www.youtube.com/watch?v=i_-ud01wFhc) - 
**Nicholas Renotte**.
The base code for this project was adapted from Nicholas Renotte. While the original concept and code were used as a foundation, modifications were made to suit the features of this R-CNN Instance Segmentation Simulation.

## Demo 🎬
Check out how the real-time segmentation and background switching work:

https://github.com/user-attachments/assets/f2a49dab-4fe6-42d2-98ad-c6370c9f2365

## The Code files: 📄
**instance.py:** Contains the core logic for instance segmentation using Mask R-CNN, including background toggling functionality.

## Functionality ⚙️

**Real-Time Segmentation:** The Mask R-CNN model segments objects in the webcam feed, drawing bounding boxes around detected objects.

**Background Mode Toggle:**
- **Normal View:** Displays the original webcam feed with bounding boxes around detected objects.
- **Black Background:** Replaces the background with a solid black background while keeping the segmented objects.

**FPS Overlay:** The frames per second (FPS) is displayed in the top-left corner of the screen to monitor the performance of real-time processing.

## Installation 💻
Make sure to have the following dependencies installed:

*python version == 3.8*

*pip install tensorflow == 2.4.1*

*pip install numpy==1.26.1*

*pip install tensorflow-gpu==2.4.1*

*pip install opencv-python==4.10.0.84*

*pip install pixellib*

## Usage 📌
Clone the repository and navigate to the project folder.

*python instance.py*

## Key Controls: 🕹️
**'t':** Toggle between normal and black background modes.

**'q':** Quit the program.

**'s':** Save the current frame as an image.

**'p':** Pause the video feed (press any key to resume).

## Theory Insight 💡
**Mask R-CNN Overview:**
Mask R-CNN is a state-of-the-art **pre-trained deep learning model** for instance segmentation. It is capable of not only detecting objects in an image but also generating pixel-wise segmentation masks for each detected object. 

The model was trained on a large dataset **(COCO dataset)** and is pre-configured to recognize various objects with high accuracy. When applied to new input data, it outputs bounding boxes around detected objects, along with the pixel-level segmentation masks for each individual object.

### Workflow:
**Input:** The input is a live feed from the webcam (video frames).

**Segmentation:** For each frame captured from the webcam, the pre-trained Mask R-CNN model segments detected objects and generates corresponding segmentation masks.

**Background Toggle:** The user can toggle between two background modes:

**Normal View:** The original webcam feed with bounding boxes around the detected objects.

**Black Background:** The background is replaced with a solid black background, while the segmented objects are retained.

**FPS Calculation:** The Frames Per Second (FPS) is calculated in real-time to monitor the performance of the segmentation process. It is displayed in the top-left corner of the video feed, providing a measure of how efficiently the system is processing the video stream.
