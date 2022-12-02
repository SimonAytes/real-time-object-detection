# Import libraries
import cv2
import numpy as np
import time
from collections import Counter

# consistent coloring
np.random.seed(20)

# Create the detector class
class Detector:
    """Object detector class"""
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        """Initialize detector object."""
        # Set paths
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Configure detector
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        
        # Import class labels and weights
        self.readClasses()

    def readClasses(self):
        """Read the compatible classes from the locally stored configuration file."""
        # Open the configuration file
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        
        # Rename the first class to avoid errors
        self.classesList.insert(0, "__Background__")

        # Create color list for classes using random seed for consistency
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def classCount(self, classLabelIds):
        """Get a text string with the number of classes to display on the frame."""
        # Get the number of occurences for each id
        objects = Counter(list(classLabelIds)).most_common()

        # Seed the display text
        displayText = "-- OBJECTS --\n"

        # Loop through each object tuple --> format = (ID, num occurences)
        for object in objects:
            # Get the name and number of occurences
            obj_name = self.classesList[object[0]]
            num_occurences = object[1]
            
            # Create temporary text for object
            temp_text = "{} (Count: {})\n".format(obj_name.upper(), num_occurences)
            
            # Append temp text to the display text
            displayText += temp_text
        
        # Return text containing all objects and occurences
        return displayText

    def onVideo(self):
        """Play the video and detect in real-time during playback (video) or capture (webcam)"""
        # Open a video stream
        cap = cv2.VideoCapture(self.videoPath)

        # Check that the video stream is active (present error if the file cannot be loaded)
        if (cap.isOpened() == False):
            print("Error opening file...")
            return

        # Extract boolean representing presence -- success
        # Extract current frame from video stream -- image
        # success is set by this initial expression and is then set at the end of each loop.
        # -- this controls the flow and shows that there is a frame next in the sequence
        (success, image) = cap.read()

        # While the frame is present
        while success:
            # Detect predefined classes in frame
            # Confidence is set at 50%
            classLabelIds, confidences, bboxs = self.net.detect(image, confThreshold = 0.5)
            
            # Get all coordinates of objects
            bboxs = list(bboxs)
            
            # Get the list of all confidence scores for the objects
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            # Retrieve the IDs of the objects found in the image
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)

            # If there are objets present, execute the following...
            if len(bboxIdx) != 0:
                # For all objects present...
                for i in range(0, len(bboxIdx)):
                    # Get the object
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    
                    # Get variables related to the detection
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIds[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    ### DRAW BOXES ###
                    # Get the vertices of the box
                    x, y, w, h = bbox

                    # Create a box at the given vertices
                    cv2.rectangle(image, (x, y), (x+w, y+h), color=classColor, thickness=1)
                    
                    # Create a text with the class label and confidence (rounded to 2 decimals)
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    ### CREATE CORNER BOUNDARIES ###
                    # Define the line width as 30% of the width or height (which ever is smaller)
                    lineWidth = min(int(w * .3), int(h * .3))

                    # Bottom Left
                    cv2.line(image, (x, y), (x + lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x, y), (x, y + lineWidth), classColor, thickness=5)
                    
                    # Bottom Right
                    cv2.line(image, (x + w, y), (x + w - lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x + w, y), (x + w, y + lineWidth), classColor, thickness=5)
                    
                    # Top Left
                    cv2.line(image, (x, y + h), (x + lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x, y + h), (x, y + h - lineWidth), classColor, thickness=5)
                    
                    # Top Right
                    cv2.line(image, (x + w, y + h), (x + w - lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w, y + h -  lineWidth), classColor, thickness=5)

            ### DISPLAY LABELS OF PRESENT OBJECTS ###
            # Get class distribution
            displayText = self.classCount(classLabelIds).split('\n')

            # Loop through each line in the display text
            for i in range(0, len(displayText)):
                # Draw a rectangle background for display text (OPTIONAL)
                #text_size, _ = cv2.getTextSize(displayText[i], cv2.FONT_HERSHEY_PLAIN, 1, 2)
                #text_w, text_h = text_size
                #cv2.rectangle(image, (50, 50 + (i*20)), (50 + text_w, 50 +(i*20) + text_h), (0, 0, 0), -1)

                # Display the text on a new line (20 pixels barrier)
                cv2.putText(image, displayText[i], (50, 50 + (i*20)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                

            
            # Display the labeled image
            cv2.imshow("Result", image)

            # Check if the quit key has been pressed (set as "q")
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Check that there is a frame after the current one
            # If there is, success = True, and the loop executes again
            (success, image) = cap.read()

        # When (A) the exit key is pressed or (B) there are no more frames, exit program.
        cv2.destroyAllWindows()