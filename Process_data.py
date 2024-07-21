import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data/single'

data = []
labels = []
total_images = 0

# Calculate the total number of images for progress tracking
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            if img_full_path.endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1

processed_images = 0

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Ensure it's a directory
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            if img_full_path.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image files
                data_aux = []
                x_ = []
                y_ = []

                try:
                    img = cv2.imread(img_full_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Error reading image {img_full_path}: {e}")
                    continue

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    # Ensure that data_aux has 42/84 elements
                    if len(data_aux) == 42:
                        data.append(data_aux)
                        labels.append(dir_)
                    else:
                        print(f"Skipping image {img_full_path}: Incorrect number of features")

                processed_images += 1
                print(f"Processed {processed_images}/{total_images} images")

# Check the balance of the dataset
label_counts = Counter(labels)
print("Label distribution:", label_counts)

# Function to augment data by duplicating samples
def augment_data(data, labels):
    label_to_data = defaultdict(list)
    for d, l in zip(data, labels):
        label_to_data[l].append(d)

    augmented_data = []
    augmented_labels = []
    for label, samples in label_to_data.items():
        if len(samples) < 2:
            samples *= 2  # Duplicate samples to ensure at least 2 per class
        augmented_data.extend(samples)
        augmented_labels.extend([label] * len(samples))

    return augmented_data, augmented_labels

data, labels = augment_data(data, labels)

# Check the balance of the dataset again
label_counts = Counter(labels)
print("Augmented label distribution:", label_counts)

# Save augmented data and labels to a pickle file
with open('augmented_data_single.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete. Augmented data saved to augmented_data.pickle.")
