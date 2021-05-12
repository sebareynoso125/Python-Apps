#Probamos el modelo

import cv2
import os

dataPath = 'C:/Users/sebar/Documents/Python-Apps/Facial-recognition/Face-data'
guestsPaths = os.listdir(dataPath)
print('Guests: ', guestsPaths)

'''
#method 1
face_recognizer = cv2.face.EigenFaceRecognizer_create()
#lee el modelo creado con trainingFR.py
face_recognizer.read('modelEigenFace.xml')
'''
'''
#method 2
face_recognizer = cv2.face.FisherFaceRecognizer_create()
#lee el modelo creado con trainingFR.py
face_recognizer.read('modelFisherFace.xml')
'''
'''
#method 2
face_recognizer = cv2.face.EigenFaceRecognizer_create()
#lee el modelo creado con trainingFR.py
face_recognizer.read('modelEigenFace.xml')
'''

#method 3
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#lee el modelo creado con trainingFR.py
face_recognizer.read('modelLBPHFace.xml')


#Seteamos la camara web para usar el modelo de reconocimiento facial
cap = cv2.VideoCapture(0)
faceClasif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClasif.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        #lafuncion predict predice, en base a los datos de entrada, una etiqueta [0] y un número de confianza [1]
        result = face_recognizer.predict(rostro)
        #Al rededor de la cara que se detecte en el video aparecerá el nombre de la persona reconocida
        cv2.putText(frame,'{}'.format(result),(x,y-25),1,1.3,(255,255,0),1,cv2.LINE_AA)

        '''
        #method 1
        if result[1] < 13000: #numero clave para saber cuando reconoce a alguien realmente
            cv2.putText(frame,'{}'.format(guestsPaths[result[0]]),(x,y-20),1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),1,1.3,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        '''
        '''
        #method 2
        if result[1] < 500: #numero clave para saber cuando reconoce a alguien realmente
            cv2.putText(frame,'{}'.format(guestsPaths[result[0]]),(x,y-20),1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),1,1.3,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        '''
        
        #method 3
        if result[1] < 90: #numero clave para saber cuando reconoce a alguien realmente
            cv2.putText(frame,'{}'.format(guestsPaths[result[0]]),(x,y-20),1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),1,1.3,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

#liberamos la cámara
cap.release()
cv2.destroyAllWindows()