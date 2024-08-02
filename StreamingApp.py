from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import pandas as pd
import os
from torch.nn import CosineSimilarity


def Stream(video_source, facenet_model,mtcnn, device, cascade_path):
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FPS,60)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == 0 or frame_count % 200 == 0:
            df = pd.read_csv('features.csv')
            stored_features = torch.tensor(df.drop('Name', axis=1).values).to(device)
            names = df['Name'].values
        if frame_count % 5 == 0 or frame_count == 0:
            lst_info = []
            faces = mtcnn.detect(frame)
            faces_imbedded = mtcnn.extract(frame, faces[0], save_path = None)
            if faces[0] is not None:
                for index in range(len(faces[0])):
                    x1, y1, x2, y2 = map(int, faces[0][index])
                    features = facenet_model(faces_imbedded[index].unsqueeze(0).to(device))
                    if features is None:
                        continue
                    #==================Face Recognition==================
                    comparor = CosineSimilarity().to(device)
                    similarities = comparor(features,stored_features)
                    idx = torch.argmax(similarities)
                    max_similarity = similarities[idx]
                    if max_similarity > 0.4:
                        # Draw bounding box and text
                        face_txt = f'{names[idx]}'
                        color = (0,255,0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame,face_txt, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        face_txt = f'Unknown'
                        color = (255,0,0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame,face_txt, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    lst_info.append((x1, y1, x2, y2, face_txt,color))
        else:
            for (x1, y1, x2, y2, face_txt,color) in lst_info:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame,face_txt, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Vẽ FPS lên góc trái của frame
        cv2.putText(frame, f'Number of people registered: {df.shape[0]}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Streaming', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_cascade = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, post_process=True, device = device, keep_all= True) # Define MTCNN module
    face_model = InceptionResnetV1(pretrained='vggface2',device=device).eval()
    Stream(0,face_model, mtcnn, device, path_to_cascade) #0 is the default camera (webcam)


if __name__ == '__main__':
    main()
