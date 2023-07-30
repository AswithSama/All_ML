#MY FIRST ML PROJECT

ENSURE YOU UPLOAD "yolov3.weights" FROM INTERNET AND USE AN APPROPRIATE VIDEO FILE IN PLACE OF ".mp4" FILE I MENTIONED

This ML project aims to perform multiple functions on a video, such as vehicle detection, classification (car, bike, truck, or bus), vehicle tracking, unique vehicle counting, frame counting, and traffic congestion detection.To achieve this, the project uses the YOLOv3 model, and a video input is processed as a series of frames using OpenCV. The frames per second (fps) ratio is typically set to 30, but it can be adjusted for optimal performance.

The vehicle detection and classification process identifies different types of vehicles in each frame. By tracking vehicles across consecutive frames, the system ensures a reliable count of unique vehicles passing a specified line. The additional functionality involves predicting traffic congestion. If the number of vehicles exceeds a certain threshold (e.g., 25) and remains at that level for a specified duration (e.g., 2 minutes), the system signals traffic congestion. Based on this detection, an alert message is sent to the traffic police to address the situation promptly.

NOTE: It's important to note that the values for the number of vehicles and threshold time are arbitrary and subject to adjustment based on road width and road type, respectively.

In summary, the project provides two broad and  five main functionalities: detecting, classifying, and tracking vehicles, along with unique vehicle counting and frame vehicle counting. The crucial sixth functionality is detecting traffic congestion, which aids in alerting the traffic police to address potential issues on the road. 
