import cv2

def detectFace():
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
            prevFrameFace = True
            ROI = w

        cv2.imshow('frame',img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    
def main():
    detectFace()

if __name__ == "__main__":
    main()
