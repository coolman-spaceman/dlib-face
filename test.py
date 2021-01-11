## Start the app by typing python test.py in command line, then hit 1+q to quit the webcam window 

##################    IMPORT IMPORTANT MODULES   ####################

import face_recognition
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import os 
######################################################################

col_names =  ['Name','Date','Time']
attendance = pd.DataFrame(columns = col_names)


video_capture = cv2.VideoCapture(0) #######  VIDEO CAPTURE FROM WEBCAM  ###################################

################################## LOAD ENCODINGS HERE ####################################################

faces = np.load('enc/encodings.npy',allow_pickle = True) #### LOADING    ENCODING OBJECT       ############
encodings = faces.item()                                 #### this is    ENCODING DICTIONARY  #############

known_face_names = []                                    ####  STORING NAMES IN AN ARRAY
known_face_encodings = []                                ####  STORING ENCODINGS IN AN ARRAY

for key in encodings:
	known_face_names.append(key)
	known_face_encodings.append(encodings.get(key))
###########################################################################################################

## VARIABLES TO GET STARTED

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            ########  use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
                ts = time.time()
                vdate = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                attendance.loc[len(attendance)] = [name,vdate,timeStamp]
            face_names.append(name)
            
    process_this_frame = not process_this_frame
    

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    # Hit '1' + 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):         
        break
        
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
Hour,Minute,Second=timeStamp.split(":")
fileName="Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
attendance.to_csv(fileName,index=False)


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
