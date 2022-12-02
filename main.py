# Import packages
from detector import *
import os

# Create the main program loop
def main():
    # Determine the input video source
    videoPath = None

    video_title = input("\nName of .mp4 file (or enter 0 for webcam capture)\n\t> ")

    if video_title == "0":
        videoPath = 0
    else:
        videoPath = f"./videos/{video_title}"

    # Setup paths
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")
    
    # Create a detector object
    # Initialize with the paths specified above
    detector = Detector(videoPath, configPath, modelPath, classesPath)
    
    # Start the detector's video analysis
    detector.onVideo()

if __name__ == "__main__":
    main()