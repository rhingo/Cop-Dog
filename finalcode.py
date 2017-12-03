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
import pygame
from random import randint
import wiringpi
import io
import traceback


# Initialize global dictionary to hold player_id:player_name pairs
playerIDName = {}

# Initialize recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()


"""
Define Robot Control State Machine

States:

1: robotForward
2: robotBackward
3: robotStop
4: robotRecord
5: robotSeek
"""
def robotForward():
    forwardbuf = bytes([1,0])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, forwardbuf)
    
def robotBackward():
    backbuf = bytes([2,0])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, backbuf)

def robotStop():
    stopbuf = bytes([3,0])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, stopbuf)
    
def robotRecord(playerID):
    recordbuf = bytes([4,playerID,0])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, recordbuf)

def robotSeek(criminalID):
    seekbuf = bytes([5,criminalID,0])
    retlen, retdata = wiringpi.wiringPiSPIDataRW(0, seekbuf)
    


def predictSuspect(suspectID):
    # Initialize camera
    cam = cv2.VideoCapture(0)

    # Initialize detector
    detector = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                     'haarcascades/haarcascade_frontalface_alt.xml')
    detector2 = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                         'haarcascades/haarcascade_frontalface_default.xml')
    # Found face in previous frame
    prevFrameFace = False

    # Record ROI
    ROI = -1

    numImages = 0
    predictions = []
    confidences = []
    while numImages <= 200: 
        # Get frame
        ret,img = cam.read()

        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Decision #1: Check if previous frame had a face
        if prevFrameFace:
            # Detect faces sized +/-20% off biggest face in previous search
            minSize = (int(ROI*8/10), int(ROI*8/10))
            maxSize = (int(ROI*12/10), int(ROI*12/10))
        else:
            # Minimum face size is 1/5th of screen width
            # Maximum face size is 2/3rds of screen width
            hImg, wImg = gray.shape[:2]
            minSize = (int(hImg/5),int(hImg/5))
            maxSize = (int(hImg*2/3),int(hImg*2/3))
            
        faces = detector.detectMultiScale(gray, 1.3, 3, 0, minSize, maxSize)

        # If no face found, try different classifier
        if len(faces) == 0:
            faces = detector2.detectMultiScale(gray, 1.3, 3, 0, minSize, maxSize)

        # If it still didn't work, no face
        if len(faces) == 0:
            prevFrameFace = False

        printString = ""
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            face = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
            prevFrameFace = True
            ROI = w
            
            face_centered = checkFaceCentered(x,w,wImg)
            if face_centered:
                # Stop when face is centered for the first time
                if numImages == 0:
                    robotStop()
                    robotRecord(suspectID)

                # Make prediction
                prediction,confidence = recognizer.predict(face)
                predictions.append(prediction)
                confidences.append(confidence)
                print("Prediction: {}, Confidence: {}".format(playerIDName[prediction],confidence))
                printString = '{} - {}'.format(playerIDName[prediction],confidence)
                numImages += 1

        cv2.putText(img,printString, (int(x+w*3/4),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
        cv2.imshow('frame',img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Majority pick vote
    # Find the indices of the 100 lowest confidence values (most certain)
    idx = np.argpartition(np.array(confidences),50)[:50]
    topPredictions = [predictions[i] for i in idx]
    lowestConfidences = [confidences[i] for i in idx]
    
    predictedPlayer = max(set(predictions), key=predictions.count)
    meanConfidence = np.mean(lowestConfidences)
    
    cam.release()
    cv2.destroyAllWindows()

    return predictedPlayer, meanConfidence


def checkFaceCentered(xface, wface, w):
    threshold = 30
    xcenter = xface + wface//2
    return ((xcenter < (w//2 + threshold)) and (xcenter > (w//2 - threshold)))


def scanSuspects(numPlayers):
    predictions = []
    confidences = []
    
    for suspectID in range(1,numPlayers+1):
        # Start moving robot
        robotForward()

        if suspectID > 1:
            time.sleep(4)
        
        prediction, meanConfidence = predictSuspect(suspectID)
        predictions.append(prediction)
        confidences.append(meanConfidence)
        print("\nFinal Prediction: {}, Final Confidence: {}".format(playerIDName[prediction],meanConfidence))

    # Make sure robot stopped
    robotStop()

    return predictions, confidences


"""
Given an image, try to extract the face region of the image using
two different Haar Cascade Classifiers. If no face detected, return None.

Parameters: img
Returns: detectedFace - cropped image of just face region
"""
def getFace(img):
    # Initialize classifiers - found to be best classifiers through testing
    detector = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                        'haarcascades/haarcascade_frontalface_alt.xml')
    detector2 = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                         'haarcascades/haarcascade_frontalface_default.xml')
    # Convert to gray scale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Minimum face size is 1/5th of screen width
    # Maximum face size is 2/3rds of screen width
    h, w = grayImg.shape[:2]
    minSize = (int(h/5),int(h/5))
    maxSize = (int(h*2/3),int(h*2/3))

    # Try first cascade classifier detection
    faces = detector.detectMultiScale(grayImg, 1.3, 3, 0, minSize, maxSize) 
            
    # Try again with different classifier
    if len(faces) != 1:
        faces = detector2.detectMultiScale(grayImg, 1.3, 3, 0, minSize, maxSize)

    # If it still didn't work, quit
    if len(faces) != 1:
        return None
    
    for x,y,w,h, in faces:
        detectedFace = grayImg[y:y+h, x:x+w]

    # Need to set all faces to be same size for recognition
    detectedFace = cv2.resize(detectedFace, (150, 150))
    
    return detectedFace


"""
Takes a photo and tries to detect a face. If a face was detected,
the image is added to the training set, otherwise keep trying.

Parameters: None
Returns: faceImages - list of face images
"""
def takeMugshots():
    numImages = 50
    faceImages = []
    
    # Initialize camera
    cam = cv2.VideoCapture(0)

    # Loop that goes on until we find enough face photos
    findingFaces = True
    while findingFaces:
        # Step 1: Gather images
        images = []
        photoID = 0
        print('Capturing Images...')
        for photoID in range(numImages):
            ret,img = cam.read()
            images.append(img)

            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        
        # Step 2: Detect faces in photos
        print('Detecting Faces...')
        for img in images:
            faceImage = getFace(img)
            if faceImage is not None:
                faceImages.append(faceImage)

            if len(faceImages) >= 10:
                findingFaces = False
                break
            
    return faceImages


"""
Each suspect sits in front of the robot and has their picture
taken. Get 10 face images of each person

Parameters: numPlayers
Returns: faces - list of face images
         labels - list of player IDs
"""
def getTrainingData(numPlayers):
    faces = []
    labels = []
    
    for playerID in range(1,numPlayers+1):
        waitString = ('{}, please get ready to take your '
                      'pictures '.format(playerIDName[playerID]))
        input(waitString) # Will begin taking pictures on enter key push
        faceImages = takeMugshots()
        faces.extend(faceImages)
        labels.extend([playerID]*len(faceImages))

    return faces,labels


def bark():
    time.sleep(2) #Gives robot time to reach final location
    
    #Initializing music
    pygame.init()
    pygame.mixer.music.load('bark.mp3')

    #Playing music
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue


"""
Initializes each player to have a name associated with their ID

Parameters: numPlayers
Returns: playerIDName - dictionary of form {playerID:playerName}
"""
def getPlayerNames(numPlayers):
    for playerID in range(1,numPlayers+1):
        playerName = input("What is Player {}'s name: ".format(playerID))
        playerIDName[playerID] = playerName
    


"""
Picks one of the suspects to be the criminal

Parameters: numPlayers
Returns: playerID of suspected criminal
"""
def selectCriminal(numPlayers):
    input('Now everyone get in line!') # Waits for input so everyone can be ready
    criminal = randint(1, numPlayers) #Select Criminal
    print('Selecting Criminal...')
    print('The Criminal is ' + str(playerIDName[criminal]))

    return criminal



def locateCriminal(predictions, confidences, criminal):
    
    potentialCriminals = [i for i in range(len(predictions)) if predictions[i] == criminal]

    numCrim = len(potentialCriminals)

    if numCrim == 0:
        print('No Criminal Located')
    else:
        criminalConfidence= min([confidences[i] for i in potentialCriminals])
        criminalIdx = confidences.index(criminalConfidence)
        robotSeek(criminalIdx+1)
        print(criminalIdx+1)
        bark()


""" Main function """
def main():
    try:
        # Initialize SPI
        wiringpi.wiringPiSPISetup(0, 500000)

        # Initialize to a stop
        robotStop()

        # Get number of players
        numPlayers = input('How many people are playing? ')
        numPlayers = int(numPlayers)

        # Get Player Names
        getPlayerNames(numPlayers)

        # Get training data
        faces, labels = getTrainingData(numPlayers)

        # Train model
        print('Training...')
        recognizer.train(faces, np.array(labels)) #expects numpy array, not list

        # Select criminal
        criminal = selectCriminal(numPlayers)

        # Scan suspects and Predict
        predictions, confidences = scanSuspects(numPlayers)

        # Move to Criminal Location
        locateCriminal(predictions, confidences, criminal)
    
    except Exception as e:
        print('Breaking!')
        print(traceback.format_exc())
        stopbuf = bytes([ord('s')])
        retlen, retdata = wiringpi.wiringPiSPIDataRW(0, stopbuf)
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

