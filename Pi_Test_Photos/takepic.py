import cv2
import time

# Initialize camera
cam = cv2.VideoCapture(0)

name =  input('What is your name? ')
numpics = input('How many photos? ')
numpics = int(numpics)

for i in range(0, numpics):
    time.sleep(.2)
    ret,img = cam.read()
            
    string = name + str(i+1) + '.jpg'
	
    cv2.imwrite(string,img)

    cv2.imshow('frame',img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
	


