from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import pandas as pd
import numpy as np
import time
import os
from torch.nn import CosineSimilarity

def extract_features(img, model, mtcnn):
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None
    img_cropped = img_cropped.to("cuda")
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = model(img_cropped.unsqueeze(0))
    return img_embedding

def compare_with_csv_reduced(video_source, model, mtcnn, csv_path, device, cascade_path):
    cap = cv2.VideoCapture(video_source)
    df = pd.read_csv(csv_path)
    stored_features = torch.tensor(df.drop('Name', axis=1).values).to(device)
    names = df['Name'].values

    face_cascade = cv2.CascadeClassifier(cascade_path)

    prev_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if frame_count % 5 == 0:
            lst_info = []
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                features = extract_features(roi, model, mtcnn)
                if features is None:
                    continue
                comparor = CosineSimilarity().to(device)
                similarities = comparor(features,stored_features)
                idx = torch.argmax(similarities)
                max_similarity = similarities[idx]

                if max_similarity > 0.4:
                    # Draw bounding box and text
                    txt = f'{names[idx]}/ Similarity:{round(max_similarity.item(),2)}'
                    color = (0,255,0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame,txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    txt = f'Unknown/ Similarity:{round(max_similarity.item(),2)}'
                    color = (0,0,255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame,txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                lst_info.append((x, y, w, h, txt,color))
        else:
            for (x, y, w, h, txt,color) in lst_info:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame,txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                

        # Tính toán FPS thực tế
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Vẽ FPS lên góc trái của frame
        cv2.putText(frame, f'FPS: {round(fps,2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_csv = os.path.join(current_dir, 'features.csv')
    path_to_cascade = os.path.join(current_dir, 'cascade/haarcascade_frontalface_default.xml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, post_process=True, device = device) # Define MTCNN module
    model = InceptionResnetV1(pretrained='vggface2',device=device).eval()
    compare_with_csv_reduced(0,model,mtcnn,path_to_csv,device,path_to_cascade) #0 is the default camera (webcam


if __name__ == '__main__':
    main()