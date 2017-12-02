import cv2
import numpy as np



def getTrainingData(people):
    facePhotos = 0
    faces = []
    labels = []

    for ID,person in enumerate(people):
        for i in range(1,10):
            filename = '/home/pi/Cop-Dog/Pi_Test_Photos/{}{}.jpg'.format(person,i)
            print(filename)
            img = cv2.imread(filename)
            faceimg = getFace(img)

            if faceimg is not None:
                facePhotos += 1
                faces.append(faceimg)
                labels.append(ID+1)
            print(facePhotos)
            
        facePhotos = 0
                
    return faces,labels

def getFace(img):
    #Found to be best classifiers through testing
    faceCascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                        'haarcascades/haarcascade_frontalface_alt.xml')
    faceCascade2 = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                         'haarcascades/haarcascade_frontalface_default.xml')
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, 1.3, 3) #Found to be best parameters

    # Try again with different classifier
    if len(faces) != 1:
        faces = faceCascade2.detectMultiScale(grayImg, 1.3, 3)

    # If it still didn't work, quit
    if len(faces) != 1:
        return None
    
    for x,y,w,h, in faces:
        detectedFace = grayImg[y:y+h, x:x+w]

    #need to set all faces to be same size for recognition
    detectedFace = cv2.resize(detectedFace, (150, 150))
    
    return detectedFace 

def detectFace(recognizer,people):
    # Initialize camera
    cam = cv2.VideoCapture(0)

    # Initialize detector
    detector = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                     'haarcascades/haarcascade_frontalface_default.xml')
    detector2 = cv2.CascadeClassifier('/home/pi/Downloads/opencv/data/'
                                         'haarcascades/haarcascade_frontalface_alt.xml')
    # Found face in previous frame
    prevFrameFace = False

    # Record ROI
    ROI = -1

    count = 0
    predictions = []
    confidences = []
    while True: 
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
            h, w = gray.shape[:2]
            minSize = (int(h/5),int(h/5))
            maxSize = (int(h*2/3),int(h*2/3))
            
        faces = detector.detectMultiScale(gray, 1.3, 3, 0, minSize, maxSize)

        # If no face found, try different classifier
        if len(faces) == 0:
            faces = detector2.detectMultiScale(gray, 1.3, 3, 0, minSize, maxSize)

        # If it still didn't work, no face
        if len(faces) == 0:
            prevFrameFace = False
            
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            face = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
            prediction,confidence = recognizer.predict(face)
            predictions.append(prediction)
            confidences.append(confidence)
            print("Prediction: {}, Confidence: {}".format(people[prediction-1],confidence))
            prevFrameFace = True
            ROI = w

        cv2.imshow('frame',img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        count += 1

    #prediction = max(predictions)
    #mean_conf = np.mean(confidences)
    #print("Prediction: {}, Confidence: {}".format(people[prediction-1],mean_conf))

    cam.release()
    cv2.destroyAllWindows()
    
def main():

    recognizer = cv2.face.createLBPHFaceRecognizer()

    people = ['Rahul', 'Julia','Angela','Miguel','Keanu ']
    faces,labels = getTrainingData(people)

    recognizer.train(faces,np.array(labels))
    #recognizer.load('trained_dataset.yml')
        
    detectFace(recognizer,people)

if __name__ == "__main__":
    main()
