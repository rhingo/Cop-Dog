"""
Course: EECS 452
Project: Cop Dog
Team Members: Rahul Hingorani
              Julia Kerst
              Angela Brown
              Saud Alrufaydah

Project Description: 
"""

# Import modules
import numpy as np
import cv2
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import pygame
from random import randint
import wiringpi
import io

# Initialize Camera
cam = PiCamera()
rawCapture = PiRGBArray(cam, size=(640, 480))


def detectFace(img): 
    #Found to be best classifiers through testing
    faceCascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                        'haarcascades/haarcascade_frontalface_alt.xml')
    faceCascade2 = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                         'haarcascades/haarcascade_frontalface_default.xml')
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayImg, 1.3, 3) #Found to be best parameters

    if len(faces) == 1:
            for x,y,w,h in faces:
                    detectedFace = grayImg[y:y+h, x:x+w]
    else:   #Run second check
            facecheck = faceCascade2.detectMultiScale(grayImg, 1.3, 3)
            if len(facecheck) == 1:
                for x,y,w,h, in facecheck:
                    detectedFace = grayImg[y:y+h, x:x+w]
            else: 
                return None, (None, None, None, None)

    #need to set all faces to be same size for recognition
    detectedFace = cv2.resize(detectedFace, (150, 150))
    
    return detectedFace, (x, y, w, h)

def getLocation()
    sensorDist = 0
    #gets the distance from the ultrasonic sensor
    return sensorDist


def takeMugshots():
    numPhotos = 3
    facePhotos = 0
    faceImages = []

    while facePhotos < numPhotos:
        cam.start_preview()
        time.sleep(3)

        cam.capture(rawCapture, format="bgr")
        fullImage = rawCapture.array

        faceImage, (x,y,w,h) = detectFace(fullImage)

        if faceImage is not None:
            faceImages.append(faceImage)
            facePhotos += 1
            print('Face found!')
        else:
            print('No face found :(')

        rawCapture.truncate(0)

    cam.stop_preview()
    return faceImages
    

def getTrainingData(numPlayers):
    faces = []
    labels = []
    
    for suspectID in range(1,numPlayers+1):
        waitString = 'Waiting for Player ' + str(suspectID) + ' to be ready...'
        input(waitString) # Will begin taking pictures on enter key push
        faceImages = takeMugshots()
        faces.extend(faceImages)
        labels.extend([suspectID]*len(faceImages))

    return faces,labels


def findNextSuspect():
    # Capture frames from the camera
    faceCentered = False
    for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # Grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        face, (xface, yface, wface, hface) = detectFace(image)
        h, w = image.shape[:2]

        if face is not None:
            print('Face found!')
            cv2.rectangle(image,(xface,yface),(xface+wface,yface+hface),(255,0,0),2)
            faceCentered = checkFaceCentered(xface, wface, w)
                
        # Show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
 
        # Clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        if faceCentered:
            # Stop moving robot
            stopbuf = bytes([ord('s')])
            retlen, retdata = wiringpi.wiringPiSPIDataRW(0, stopbuf)

            # Wait to stabilize
            time.sleep(2)

            #Get distance
            suspectLoc = getdistance()

            cam.capture(rawCapture, format="bgr")
            suspectImage = rawCapture.array

            rawCapture.truncate(0)
            return suspectImage, suspectLoc
 
        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


def scanSuspects(numPlayers):
    # Initialize test data
    testData = []
    locationData = []

    # Start moving robot
    forwardbuf = bytes([ord('f')])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, forwardbuf)
    
    suspectsFound = 0
    while suspectsFound < numPlayers:
        suspectImage, suspectLoc = findNextSuspect()
        testData.append(suspectImage)
        locationData.append(suspectLoc)
        suspectsFound += 1

        # Start moving robot
        forwardbuf = bytes([ord('f')])
        retlen, retdata = wiringpi.wiringPiSPIDataRW(0, forwardbuf)
        
        # Wait to avoid finding same suspect
        time.sleep(8)

    return testData, locationData
        

def checkFaceCentered(xface, wface, w):
    threshold = 30
    xcenter = xface + wface//2
    return ((xcenter < (w//2 + threshold)) and (xcenter > (w//2 - threshold)))


def predictCriminal(testData, recognizer, criminal):
    predictions = []
    confidences = []
    
    for i in range(len(testData)):
        predictedLabel, confidence = recognizer.predict(testData[i])
        predictions.append(predictedLabel)
        confidences.append(confidence)

    suspectedCriminals = [i for i,prediction in enumerate(predictions) if predictions[i] == criminal]

    if len(suspectedCriminals) == 1:
        predictedCriminal = suspectedCriminal[0]
    elif len(suspectedCriminals) > 1:
        highestConfidence = max([confidences[i] for i in range(len(suspectedCriminals))])
        predictedCriminal = confidences.index(highestConfidence)
    else:
        print('Could not find criminal')
        predictedCriminal = None

    print ('Predicted Criminal is Suspect {}'.format(predictedCriminal))
    return predictedCriminal


def moveToCriminal(predictedCriminal, locationData)
    # Start moving robot backwards
    backbuf = bytes([ord('b')])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, backbuf)

    currentLoc = getLocation()

    criminalLoc = locationData[predictedCriminal]

    while currentLoc < criminalLoc
        #keep moving
        time.sleep(1)
        currentLoc = getLocation()

    #now stop and bark
    stopbuf = bytes([ord('s')])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, stopbuf)

    #Initializing music
    pygame.init()
    pygame.mixer.music.load('bark.mp3')

    #Playing music
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    return()

""" Main function """
def main():
    try:
        # Initialize camera settings
        cam.resolution = (640, 480)
        cam.framerate = 32
        cam.vflip = True

        # Initialize recognizer
        #recognizer = cv2.face.createEigenFaceRecognizer()

        # Initialize SPI
        wiringpi.wiringPiSPISetup(0, 500000)

        # Initialize to a stop
        stopbuf = bytes([ord('s')])
        retlen, retdata = wiringpi.wiringPiSPIDataRW(0, stopbuf)

        # Get number of players
        numPlayers = input('How many people are playing? ')
        numPlayers = int(numPlayers)

        # Get training data
        faces, labels = getTrainingData(numPlayers)

        # Train model
        print('Training...')
        # recognizer.train(faces, np.array(labels)) #expects numpy array, not list

        # Select criminal
        input('Now everyone get in line!') # Waits for input so everyone can be ready
        criminal = randint(1, numPlayers) #Select Criminal
        crimstr = 'The Criminal is Suspect #' + str(criminal)

        # Scan suspects
        testData, locationData = scanSuspects(numPlayers)

        # Predict criminal
        #predictedCriminal = predictCriminal(testData, recognizer, criminal)

        if predictedCriminal is not None:
            moveToCriminal(predictedCriminal, locationData)

        return()
        
    except KeyboardInterrupt:
        print('Breaking!')
        stopbuf = bytes([ord('s')])
        retlen, retdata = wiringpi.wiringPiSPIDataRW(0, stopbuf)
        cv2.destroyAllWindows()
        return()

if __name__ == "__main__":
    main()
