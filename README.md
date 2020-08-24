# Social-Distance-Feedback-For-The-Blind
A social distancing feedback system for the blind using the OAK-D camera. This project is part of the [OpenCV Spatial AI Competition](https://opencv.org/opencv-spatial-ai-competition/).

![Proposed System Output Illustration](https://github.com/ibaiGorordo/Social-Distance-Feedback/blob/master/docs/images/proposed%20system%20illustration.jpg)

# Project Idea

Due to current worldwide outbreak of COVID-19, **social distancing** has become a new social norm as a measure to prevent the widespread of the pandemic. However, **visually impaired people** are struggling to keep independence in the new socially distanced normal. For blind people, it is not possible to easily confirm if they are keeping the social distance with the people around them. 

The objective of my project is to develop a feedback system for the blind that informs about the distance to other people around and whether someone is not wearing a mask. The system uses the [**OpenCV AI Kit with Depth (OAK-D)**](https://www.kickstarter.com/projects/opencv/opencv-ai-kit) due to the capability to run Deep Learning models in real time in the edge in combination with depth estimation from the estereo cameras.

# Medium Posts

I will be publishing a series of Medium posts where I write about the journey to develop the proposed system.

- [**Part 1: Initial Setup + Depth**](https://medium.com/@ibaiGorordo/opencv-spatial-ai-competition-journey-part-1-e76593d456fe?source=friends_link&sk=c2f0617e6d940c74a97fbf471f61f2db): Introduction about my project and initial setup of the OAK-D board. I also compare the estimated depth with the one from the Azure Kinect DK.

![Estimated Depth using the OAK-D](https://github.com/ibaiGorordo/Social-Distance-Feedback/blob/master/docs/images/OAKD%20depth.png)

- [**Part 2: Yet another Face Mask Detector…**](https://medium.com/@ibaiGorordo/part-2-yet-another-face-mask-detector-opencv-spatial-ai-competition-journey-91dfaf96c6e8?source=friends_link&sk=4da5e426c723857b2c4ce56331426aa9): Comparison of different object detection models (SSD-MobileNetv2, YOLOv3-tiny and YOLOv4-tiny) for **Face Mask Detection** to find where the people around is.

![Face Mask Detection using YOLOv3-tiny](https://github.com/ibaiGorordo/Social-Distance-Feedback/blob/master/docs/images/Part%202%20-%20Face%20mask%20detection%20intro.gif)

