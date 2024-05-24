from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import pandas as pd
import numpy as np
import time
import os
from torch.nn import CosineSimilarity
from torch import nn
from PIL import Image
from torchvision import transforms
import pypyodbc as pyodbc

class EmotionDetector(nn.Module):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8)
        self.cnn2_bn = nn.BatchNorm2d(16)
        self.cnn3_bn = nn.BatchNorm2d(32)
        self.cnn4_bn = nn.BatchNorm2d(64)
        self.cnn5_bn = nn.BatchNorm2d(128)
        self.cnn6_bn = nn.BatchNorm2d(256)
        self.cnn7_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))

        x = x.reshape(x.size(0), -1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.log_softmax(self.fc3(x))
        return x
def connect_sql_server(sql_server_info):
    try:  # Kết nối đến SQL Server
        sql_server_conn = pyodbc.connect(
        'DRIVER=' + sql_server_info['driver'] + ';'
        'SERVER=' + sql_server_info['server'] + ';'
        'DATABASE=' + sql_server_info['database'] + ';'
        'UID=' + sql_server_info['user'] + ';'
        'PWD=' + sql_server_info['password'] + ';'
        )
        cursor = sql_server_conn.cursor()
        cursor.execute("SELECT @@version AS sql_version;")
        print("Kết nối SQL Sever thành công!")

    except Exception as e:
        print(f"Kết nối SQL Sever thất bại. Lỗi: {e}")

    return sql_server_conn

def load_trained_model(model_path):
    model = EmotionDetector()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model

def extract_features(img, model, mtcnn, device):
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None
    img_cropped = img_cropped.to(device)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = model(img_cropped.unsqueeze(0))
    return img_embedding

def Stream(video_source, face_model,emote_model, mtcnn, sql_engine, device, cascade_path):
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FPS,60)
    emotion_dict = ['neutral', 'happy', 'surprised', 'sad', 'anger', 'disgusted', 'fearful']
    val_transform = transforms.Compose([
    transforms.ToTensor()])
    face_cascade = cv2.CascadeClassifier(cascade_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == 0 or frame_count % 200 == 0:
            query = "Select * from features"
            df = pd.read_sql_query(query, sql_engine)[1:]
            stored_features = torch.tensor(df.drop('name', axis=1).values).to(device)
            names = df['name'].values
        if frame_count % 5 == 0 or frame_count == 0:
            lst_info = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=6, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                features = extract_features(roi, face_model, mtcnn,device)
                if features is None:
                    continue
                #==================Face Recognition==================
                comparor = CosineSimilarity().to(device)
                similarities = comparor(features,stored_features)
                idx = torch.argmax(similarities)
                max_similarity = similarities[idx]
                #==================Emotion Detection==================
                resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                X = resize_frame / 256
                X = Image.fromarray((X))
                X = val_transform(X).unsqueeze(0).to(device)
                with torch.no_grad():
                    emote_model.eval()
                    log_ps = emote_model(X)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    pred = emotion_dict[int(top_class.cpu().numpy()[0][0])]

                if max_similarity > 0.4:
                    # Draw bounding box and text
                    face_txt = f'{names[idx]}'
                    emote_txt = f'Emotion: {pred}'
                    color = (0,255,0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame,face_txt, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, emote_txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,color, 2)
                else:
                    face_txt = f'Unknown'
                    emote_txt = f'Emotion: {pred}'
                    color = (255,0,0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame,face_txt, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, emote_txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,color, 2)
                lst_info.append((x, y, w, h, face_txt,emote_txt,color))
        else:
            for (x, y, w, h, face_txt,emote_txt,color) in lst_info:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame,face_txt, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, emote_txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,color, 2)
        
        # Vẽ FPS lên góc trái của frame
        cv2.putText(frame, f'Number of people registered: {df.shape[0]}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Streaming', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    sql_server_info = {
    'server': 'PhucDo\SQLEXPRESS01',
    'database': '9.5AI',
    'user':'phucdo',
    'password':'2203',
    'driver': '{ODBC Driver 17 for SQL Server}',  # Thay đổi driver tùy theo phiên bản SQL Server
    }
    sql_engine = connect_sql_server(sql_server_info)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_cascade = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, post_process=True, device = device) # Define MTCNN module
    face_model = InceptionResnetV1(pretrained='vggface2',device=device).eval()
    emote_model = load_trained_model(os.path.join(current_dir, 'Emotion_Detection_Ver2.pt')).to(device)
    Stream(0,face_model,emote_model,mtcnn,sql_engine,device,path_to_cascade) #0 is the default camera (webcam)


if __name__ == '__main__':
    main()