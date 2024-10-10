import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import numpy.linalg as LA

gesture = {
    0:'FIRST', 1:'ONE', 2:'TWO', 3:'THREE', 4:'FOUR', 5:'FIVE', 6:'SIX', 7:'ROCK', 8:'SPIDERMAN', 9:'YEACH', 10:'OK',
}

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
gesture_df = pd.read_csv('c:/ai_project01/gesture_train.csv', header=None)
#print("gesture_df=", gesture_df)
#print("=" * 100)
angle = gesture_df.iloc[:,:-1]
#print("angle=", angle)
#print("=" * 100)
#print("type(angle)=", type(angle))
#print("angle.values", angle.values)
#print("type(angle.values)=", type(angle.values))
angle_arr = angle.values.astype(np.float32)
#print("angle_arr=", angle_arr)
#print("=" * 100)
label = gesture_df.iloc[:, -1]
#print("label=", label)
#print("=" * 100)
#print("type(label)=", type(label))
#print("label.values", label.values)
#print("type(label.values)=", type(label.values))
label_arr = label.values.astype(np.float32)
#print("label_arr=", label_arr)
#print("=" * 100)

knn = cv2.ml.KNearest_create()
knn.train(angle_arr, cv2.ml.ROW_SAMPLE, label_arr)

cv2.namedWindow(winname='webcam_window01', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname='webcam_window01', width=1024, height=800)

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands :
    while cap.isOpened() == True :
        success, image = cap.read()
        image = cv2.flip(image, flipCode = 1)
        if success == False :
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None :
            #cv2.putText(
            #    image,
            #    text = "Detect Hand!!",
            #    org = (300, 50),
            #    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #    fontScale=1,
            #    color=(0, 0, 255),
            #    thickness=2
            #)
            for hand_landmarks in results.multi_hand_landmarks :
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark) :
                    #print("j=",j)
                    #print("lm=",lm)
                    #print("lm.x=",lm.x)
                    #print("lm.y=",lm.y)
                    #print("lm.z",lm.z)
                    joint[j] = [lm.x, lm.y, lm.z]
                    #print("=" * 100)
                    #print("joint=", joint)
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
                    v = v2 - v1
                    #print("="*100)
                    #print("v=",v)
                    #print("="*100)
                    v_normal = LA.norm(v, axis=1)
                    #print("="*100)
                    #print("v_normal=",v_normal)
                    #print("="*100)
                    v_normal2 = v_normal[:, np.newaxis]
                    #print("v_normal2=", v_normal2)
                    v2 = v/v_normal2
                    #print("="*100)
                    #print("v2=",v2)
                    #print("="*100)
                    a = v2[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:]
                    b = v2[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]
                    ein = np.einsum('ij,ij->i', a, b)
                    #print("="*100)
                    #print("ein=",ein)
                    #print("=" * 100)
                    radian = np.arccos(ein)
                    #print("radian=",radian)
                    angle = np.degrees(radian)
                    #print("angle=",angle)
                    data = np.array([angle], dtype=np.float32)
                    #print("data=",data)
                    retval, results, neighbours, dist = knn.findNearest(data, 3)

                    print("="*100)
                    print("retval=", retval)
                    print("=" * 100)
                    print("results=", results)
                    print("=" * 100)
                    print("neighbours=", neighbours)
                    print("=" * 100)
                    print("dist=", dist)
                    print("=" * 100)

                    idx = int(retval)
                    print("idx=",idx)
                    print("image.shape[0]=",image.shape[0])
                    print("image.shape[1]=", image.shape[1])
                    print("hand_landmarks.landmark[0].x=",hand_landmarks.landmark[0].x)
                    print("hand_landmarks.landmark[0].y=", hand_landmarks.landmark[0].y)
                    if 0 <= idx <= 10 :
                        cv2.putText(
                            image,
                            text=gesture[idx],
                            org=(
                                int(hand_landmarks.landmark[0].x * image.shape[1]),
                                int(hand_landmarks.landmark[0].y * image.shape[0])
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2
                        )

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
                #if hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y :
                #    cv2.putText(
                #        image,
                #        text="Open!!!",
                #        org=(300, 100),
                #        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #        fontScale=1,
                #        color=(0, 0, 255),
                #        thickness=2
                #    )
                #else :
                #    cv2.putText(
                #        image,
                #        text="Close",
                #        org=(300,100),
                #        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #        fontScale=1,
                #        color=(0, 0, 255),
                #        thickness=2
                #    )
            #print("손 찾았음!!!!!!!!!")

        cv2.imshow('webcam_window01', image)
        if cv2.waitKey(1) == ord('q') :
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #plt.imsave("cam_img.jpg", image)
            break

cap.release()