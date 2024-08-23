import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from deepface import DeepFace
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# Define available backends and alignment modes
BACKENDS = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'fastmtcnn',
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]
ALIGNMENT_MODES = [True, False]

def extract_frames(video_path: str, frame_rate: int = 25) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get the original frame rate of the video
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure original_frame_rate and frame_rate are valid
    if original_frame_rate == 0 or frame_rate == 0:
        raise ValueError("Frame rate cannot be zero.")
    
    frame_interval = max(1, int(original_frame_rate / frame_rate))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames

def extract_faces(frame: np.ndarray, backend: str, align: bool) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    face_objs = DeepFace.extract_faces(img_path=frame, 
                                       detector_backend=backend, 
                                       align=align, 
                                       enforce_detection=False)
    return [(face_obj['face'], face_obj['facial_area']) if face_obj['confidence'] > 0 else (None, None) for face_obj in face_objs]

def extract_face_attributes(face_img: np.ndarray, backend: str, align: bool) -> dict:
    # Ensure the image is in uint8 format
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8)
    
    # Convert the image from BGR (OpenCV default) to RGB (DeepFace expected format)
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Get the attributes
    attributes = DeepFace.analyze(img_path=face_img_rgb, actions=['age', 'gender', 'race', 'emotion'], detector_backend=backend, align=align, silent=True, enforce_detection=False)
    return attributes[0]

def extract_face_embeddings(face_img: np.ndarray, backend: str, align: bool) -> np.ndarray:
    # Ensure the image is in uint8 format
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8)
    
    # Convert the image from BGR (OpenCV default) to RGB (DeepFace expected format)
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Get the embedding
    embedding_obj = DeepFace.represent(img_path=face_img_rgb, detector_backend=backend, align=align, enforce_detection=False)
    return np.array(embedding_obj[0]['embedding'])

def cluster_faces(face_embeddings: List[np.ndarray], num_clusters: int) -> List[int]:
    if num_clusters is None:
        # TODO: in the future we may want to have a method to determine optimal number of clusters
        raise ValueError("Number of clusters must be specified.")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(face_embeddings)
    return labels

def draw_label_on_frame(frame: np.ndarray, face: dict) -> None:
    x, y, w, h = face['attributes']['region']['x'], face['attributes']['region']['y'], face['attributes']['region']['w'], face['attributes']['region']['h']
    label = (f"ID: {face['label']}, Age: {face['attributes']['age']}, "
             f"Gender: {face['attributes']['dominant_gender']}, "
             f"Emotion: {face['attributes']['dominant_emotion']}")
    color = (0, 255, 0)  # Green for bounding box

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw label
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_json(data: dict, output_folder: str, video_name: str) -> None:
    # Recursive function to convert non-serializable objects
    def convert_item(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.int32, np.int64, np.float32, np.float64)):
            return item.item()
        elif isinstance(item, dict):
            return {k: convert_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_item(i) for i in item]
        else:
            return item

    # Convert the data to be JSON serializable
    data = convert_item(data)
    
    json_path = os.path.join(output_folder, f"{video_name}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_labeled_video(frames: List[np.ndarray], output_path: str, frame_rate: int = 25) -> None:
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()

def save_face_videos(frames: List[np.ndarray], frame_faces_data: List[List[dict]], output_subfolder: str, frame_rate: int = 25) -> None:
    face_video_writers = {}
    max_width, max_height = 0, 0

    # Determine the maximum width and height across all detected faces
    for frame_data in frame_faces_data:
        for face_data in frame_data:
            region = face_data['attributes']['region']
            max_width = max(max_width, region['w'])
            max_height = max(max_height, region['h'])

    black_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # Iterate through all frames and faces
    for frame_idx, frame_data in enumerate(frame_faces_data):
        frame = frames[frame_idx]
        
        for face_data in frame_data:
            face_id = face_data['label']
            region = face_data['attributes']['region']
            
            # Initialize video writer for the face if not already done
            if face_id not in face_video_writers:
                face_video_path = os.path.join(output_subfolder, f"face_{face_id}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                face_video_writers[face_id] = cv2.VideoWriter(face_video_path, fourcc, frame_rate, (max_width, max_height))
            
            # Crop the face from the frame
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            face_frame = frame[y:y+h, x:x+w]

            # Create a padded frame to match the largest face size
            padded_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            x_offset = (max_width - w) // 2
            y_offset = (max_height - h) // 2
            padded_frame[y_offset:y_offset+h, x_offset:x_offset+w] = face_frame

            # Write the padded face frame to the video
            face_video_writers[face_id].write(padded_frame)
        
        # Write black frames for faces not detected in this frame
        detected_ids = {face_data['label'] for face_data in frame_data}
        for face_id in face_video_writers:
            if face_id not in detected_ids:
                face_video_writers[face_id].write(black_frame)

    # Release all video writers
    for writer in face_video_writers.values():
        writer.release()

def process_video(video_path: str, num_clusters: int, output_folder: str, backend: str = 'retinaface', align: bool = True) -> None:
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create subfolder in output folder
    output_subfolder = os.path.join(output_folder, video_name)
    os.makedirs(output_subfolder, exist_ok=True)

    # Process the video
    frames = extract_frames(video_path)
    frames = frames[:50]  # TODO: This is just for testing purposes
    all_face_embeddings = []
    frame_faces_data = []

    # First pass: extract all embeddings and attributes
    for frame in tqdm(frames, desc=f"Processing frames for video {video_path}..."):
        face_objs = extract_faces(frame, backend, align)
        frame_data = []
        
        for (face_img, facial_area) in face_objs:
            if face_img is None:
                continue
            embedding = extract_face_embeddings(face_img, backend, align)
            all_face_embeddings.append(embedding)
            attributes = extract_face_attributes(face_img, backend, align)
            attributes['region'] = facial_area
            frame_data.append({'attributes': attributes, 'embedding': embedding})
        
        frame_faces_data.append(frame_data)
    
    # Second pass: cluster embeddings and assign labels
    if all_face_embeddings:
        labels = cluster_faces(all_face_embeddings, num_clusters)
        
        # Assign labels to faces in frame data
        label_idx = 0
        for frame_data in frame_faces_data:
            for face_data in frame_data:
                face_data['label'] = labels[label_idx]
                label_idx += 1

    # Save frame faces data as JSON
    save_json(frame_faces_data, output_subfolder, video_name)

    # Draw labels on video frames
    for frame, frame_data in zip(frames, frame_faces_data):
        for face_data in frame_data:
            draw_label_on_frame(frame, face_data)

    # Save the labeled video
    labeled_video_path = os.path.join(output_subfolder, f"{video_name}_labeled.mp4")
    save_labeled_video(frames, labeled_video_path)

    # Save individual face videos
    save_face_videos(frames, frame_faces_data, output_subfolder)

if __name__ == "__main__":
    video_path = "../data/0001.mp4"
    output_folder = "../output"
    process_video(video_path, 
                  num_clusters=2, 
                  output_folder=output_folder,
                  backend=BACKENDS[4], 
                  align=ALIGNMENT_MODES[0])
