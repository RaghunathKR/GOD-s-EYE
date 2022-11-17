import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from playsound import playsound



path = 'images'
images = []
personNames = []
myList = os.listdir(path)
print(myList)
def names(path, cu_img):
    current_Img = names(path, cu_img)
    return current_Img

def names(path, cu_img):
    current_Img = cv2.imread(f'{path}/{cu_img}')
    return current_Img

for cu_img in myList:
    current_Img = names(path, cu_img)
    images.append(current_Img)                                             
    personNames.append(os.path.splitext(cu_img)[0])  
print(personNames)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def report(name):
    with open('Gods Eye.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            #playsound('music.mp3')
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')
encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')












cap = cv2.VideoCapture(1)
cap2= cv2.VideoCapture(2)
cap3=cv2.VideoCapture(0)
while True:
    ret1, frame1 = cap.read()
    ret0, frame = cap2.read()
    ret2, frame2 = cap3.read()
    faces = cv2.resize(frame1, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            playsound('criminal.mp3')
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            report(name)
            







    faces = cv2.resize(frame2, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame2 = face_recognition.face_locations(faces)
    encodesCurrentFrame2 = face_recognition.face_encodings(faces, facesCurrentFrame2)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame2):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            playsound('criminal.mp3')
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame2, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame2, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            report(name)








    faces = cv2.resize(frame1, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame1 = face_recognition.face_locations(faces)
    encodesCurrentFrame1 = face_recognition.face_encodings(faces, facesCurrentFrame1)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame1):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            playsound('criminal.mp3')
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame1, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame1, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            report(name)   









    if (ret0):
        # Display the resulting frame
        cv2.imshow('Cam 0', frame)

    if (ret1):
        # Display the resulting frame
        cv2.imshow('Cam 1', frame1)
    if (ret2):
        # Display the resulting frame
        cv2.imshow('Cam 2', frame2)
        
    #cv2.imshow('Webcam', frame)
    #cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
