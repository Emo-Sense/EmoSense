import cv2 as cv
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model("FER_custom.keras")  #tbd

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

video = cv.VideoCapture(0)  # Change the number for when we use diff webcam

# Face Detection function
# Using Haar Cascades for now.. might change later

def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier("haar_facedetect.xml")
    #scale factor : (the amount the image is scaled i.e. reduced ) lower =  more accurate but slwer processing
    #minNeighbors : (specifies how many neighbors each face rectangle should have for it to be a face) higher = more accurate
    faces = haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=3 )
    return faces



while True:
    ret, frame = video.read()
    if not ret:
        break
    # Grayscale conversion (check whats done during training)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Face Detection
    # DK if we will need it still keeping it. 
    faces = face_detect(frame)

    '''
    If we need to specify the number of faces to detect

    if len(faces) > 0:
        faces.sort(key=lambda f: f[2] * f[3], reverse) #sorts based on the size of face

    '''
    
    for (x, y, w, h) in faces: # can specify range

        #preprocess the face region for your model
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv.resize(face_roi, (48, 48)) #change size based on model
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1)) #change size based on model
        
        #PREDICT
        predictions = model.predict(reshaped_face)
        max_index = np.argmax(predictions)
        emotion = emotions[max_index]
        
        #display the emotion 
        cv.putText(frame, emotion, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        #draw rect around face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # display new frame
    cv.imshow('Real-time Emotion Detection', frame)
    
    # break loop if 1 q is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#release video capture and close the window
video.release()
cv.destroyAllWindows()



