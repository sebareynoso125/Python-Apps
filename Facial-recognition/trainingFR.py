#entrenamos el reconocimiento facial

import cv2
import os
import numpy as np
from numpy.core.numeric import count_nonzero

dataPath = 'C:/Users/sebar/Documents/Python-Apps/Facial-recognition/Face-data'

#lista las personas registradas en la app
guestsList = os.listdir(dataPath)
print('Guests: ', guestsList)

labels = []
facesData = []
label = 0

#se leen las carpetas que contienen los archivos de los invitados y se prepara el dataset
for nameDir in guestsList:
    personPath = dataPath+'/'+nameDir
    print('Leyendo las imágenes')

    #se leen las imágenes de cada carpeta
    for fileName in os.listdir(personPath):
        #print('Rostros: ', nameDir+'/'+fileName)
        labels.append(label)
        #agregamos las imágenes en escala de grises
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        Image = cv2.imread(personPath+'/'+fileName,0)
        #vemos como se van leyendo las imagenes en escala de grises. Colocar "cv2.destroyAllWindows()" fuera del primer for
        #cv2.imshow('Image', Image)
        #cv2.waitKey(10)
    label = label+1

#para control se ven las etiquetas de las 300 imágenes de caras detectadas de cada uno de los invitados
#print('labels = ', labels)
#print('Número de etiquetas 0: ', np,count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ', np,count_nonzero(np.array(labels)==1))

'''
#Trainig method 1
face_recognizer = cv2.face.EigenFaceRecognizer_create()
print('Training ...')
face_recognizer.train(facesData, np.array(labels)) #argumentos: dataset, labels
#se almacena el modelo obtenido
face_recognizer.write('modelEigenFace.xml')
print('Modelo Almacenado.')
'''

'''
#Trainig method 2
face_recognizer = cv2.face.FisherFaceRecognizer_create()
print('Training ...')
face_recognizer.train(facesData, np.array(labels)) #argumentos: dataset, labels
#se almacena el modelo obtenido
face_recognizer.write('modelFisherFace.xml')
print('Modelo Almacenado.')
'''

#Trainig method 3
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print('Training ...')
face_recognizer.train(facesData, np.array(labels)) #argumentos: dataset, labels
#se almacena el modelo obtenido
face_recognizer.write('modelLBPHFace.xml')
print('Modelo Almacenado.')









