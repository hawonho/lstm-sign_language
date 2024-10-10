from flask import Flask
from flask import request
import base64
import json

from tensorflow import keras
import mediapipe as mp
import numpy.linalg as LA
import numpy as np
import cv2
import matplotlib as plt

model = keras.models.load_model('c://ai_project01/hand_lstm_train_result')
model.summary()
seq_length = 5
gesture = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
}
mp_hands = mp.solutions.hands

app = Flask(__name__)

@app.route("/lstm_detect", methods=["POST"])
def lstm_detect01():
    lstm_result = []
    seq = []
    with mp_hands.Hands() as hands:
        json_image = request.get_json()
        print("=" * 100)
        print("json_image=", json_image)
        print("=" * 100)
        encoded_data_arr = json_image.get("data")
        print("=" * 100)
        print("encoded_data_arr=", encoded_data_arr)
        print("=" * 100)
        for index, encoded_data in enumerate(encoded_data_arr):
            print("=" * 100)
            print("index=", index)
            print("=" * 100)
            print("encoded_data=", encoded_data)
            print("=" * 100)
            encoded_data = encoded_data.replace("image/jpeg;base64,", "")
            decoded_data = base64.b64decode(encoded_data)
            nparr = np.fromstring(decoded_data, np.uint8)
            print("=" * 100)
            print("nparr=", nparr)
            print("=" * 100)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print("=" * 100)
            print("image=", image)
            print("=" * 100)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks != None:
                for hand_landmarks in results.multi_hand_landmarks:
                    joint = np.zeros((21,3))
                    for j, lm in enumerate(hand_landmarks.landmark):
                        print("j=", j)
                        print("lm=", lm)
                        print("lm.x=", lm.x)
                        print("lm.y=", lm.y)
                        print("lm.z=", lm.z)
                        joint[j] = [lm.x, lm.y, lm.z]
                        print("=" * 100)

                    print("=" * 100)
                    print("joint=", joint)
                    print("=" * 100)
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
                    v = v2 - v1
                    print("=" * 100)
                    print("v=", v)
                    print("=" * 100)
                    v_normal = LA.norm(v, axis=1)
                    print("=" * 100)
                    print("v_normal=", v_normal)
                    print("=" * 100)
                    v_normal2 = v_normal[:, np.newaxis]
                    print("=" * 100)
                    print("v_normal2=", v_normal2)
                    print("=" * 100)
                    v2 = v / v_normal2
                    print("=" * 100)
                    print("v2=", v2)
                    print("=" * 100)
                    a = v2[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :]
                    b = v2[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
                    ein = np.einsum('ij,ij->i', a, b)
                    print("=" * 100)
                    print("ein=", ein)
                    print("=" * 100)
                    radian = np.arccos(ein)
                    print("=" * 100)
                    print("radian=", radian)
                    print("=" * 100)
                    angle = np.degrees(radian)
                    print("=" * 100)
                    print("angle=", angle)
                    print("=" * 100)
                    data = np.concatenate([joint.flatten(), angle])
                    print("=" * 100)
                    print("data=", data)
                    print("=" * 100)
                    seq.append(data)
                    if len(seq) < 5 :
                        continue

                    last_seq = seq[-5:]
                    input_arr = np.array(last_seq, dtype=np.float32)
                    print("input_arr=", input_arr.shape)
                    input_lstm_arr = input_arr.reshape(1,5,78)

                    print("=" * 100)
                    print("input_lstm_arr=", input_lstm_arr)
                    print("=" * 100)
                    print("=" * 100)
                    print("input_lstm_arr.shape=", input_lstm_arr.shape)
                    print("=" * 100)
                    y_pred = model.predict(input_lstm_arr)
                    print("=" * 100)
                    print("y_pred=", y_pred)
                    print("=" * 100)
                    idx = np.argmax(y_pred)
                    print("=" * 100)
                    print("idx=", idx)
                    print("=" * 100)
                    letter = gesture[idx]
                    print("=" * 100)
                    print("letter=", letter)
                    print("=" * 100)
                    conf = y_pred[0, idx]
                    print("=" * 100)
                    print("conf=", conf)
                    print("=" * 100)
                    lstm_result.append({
                        "text":f"{letter} {round(conf * 100, 2)} percent!!",
                        "x": int(hand_landmarks.landmark[0].x * image.shape[1]),
                        "y": int(hand_landmarks.landmark[0].y * image.shape[0])
                    })

                    print("=" * 100)
                    print("lstm_result=", lstm_result)
                    print("=" * 100)

    return json.dumps(lstm_result)

@app.route("/image_test02", methods=["POST"])
def image_send_test02():
    json_image = request.get_json()
    print("=" * 100)
    print("json_image=", json_image)
    print("=" * 100)
    encoded_data_arr = json_image.get("data")
    print("=" * 100)
    print("encoded_data_arr=", encoded_data_arr)
    print("=" * 100)
    for index, encoded_data in enumerate(encoded_data_arr):
        print("=" * 100)
        print("index=", index)
        print("=" * 100)
        print("encoded_data=", encoded_data)
        print("=" * 100)
        encoded_data = encoded_data.replace("image/jpeg;base64,","")
        decoded_data = base64.b64decode(encoded_data)

        with open(f"image{index}.jpg","wb") as f:
            f.write(decoded_data)

    return "스프링 컨트롤러가 보낸 이미지 잘 저장 했습니다"

@app.route("/image_test01", methods=["POST"])
def image_send_test01():
    json_image = request.get_json()
    print("json_image=", json_image)

    return "스프링 컨트롤러가 보낸 이미지 잘 받았습니다";

@app.route('/hello_rest_server', methods=['POST'])
def hello_rest1():
    return '안녕 난 rest server야'
if __name__ == '__main__':
    app.run()