import numpy as np
import cv2
from keras.models import load_model
import os

# trained model ko load karte hai 
model = load_model('lasttimefile.h5')

# image data ko preprocess karne ke liye function banate hai
def preeprocess_image(img):
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

# Fsteering angle ko predict karne ke liye function banate hai
def steering_angle_predict(model, processed_image):
    steering_angle = float(model.predict(processed_image, batch_size=1))
    steering_angle = steering_angle * 100
    return steering_angle


def load_image_paths(text_file_path):
    image_paths = []
    with open(text_file_path, 'r') as file:
        for line in file:
            if line.strip():  
                image_path = line.split()[0]  # image path jo hai wo  first element hai each line ne 
                image_paths.append(image_path)
    return image_paths

# visualization karne ka setup kartE HAI 
steer_image = cv2.imread('resources/wheel2.png', 0)
rows, cols = steer_image.shape
smooth_angle = 0

image_paths = load_image_paths('driving_dataset/data.txt')
data_directory = 'driving_dataset/data/'

# loop banate hai jo har image ko process karega 
for image_path in image_paths:
    full_path = os.path.join(data_directory, image_path)
    frame = cv2.imread(full_path)
    
    if frame is not None:
        # image ko show karega python window me 
        cv2.imshow('input', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))

        # image ko 40x40 ke size me karke show karega 
        gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
        processed_image = preeprocess_image(gray)
        
        # steering angle ko predict karega 
        steering_angle = steering_angle_predict(model, processed_image)
        print("steering angle hai : ", steering_angle)  # Check the steering angle values

        # smooth angle ko update karega 
        smooth_angle += 0.2 * pow(abs((steering_angle - smooth_angle)), 2.0 / 3.0) * \
            (steering_angle - smooth_angle) / abs(steering_angle - smooth_angle) if abs(steering_angle - smooth_angle) > 1e-4 else 0
        
        print("smoothed angle hai : ", smooth_angle)  # Check the smoothed angle values
        
        # wheel ka visualization karega
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smooth_angle, 1)
        dst = cv2.warpAffine(steer_image, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("image nhi hai : ", full_path)

        
# saare windows ko close karega
cv2.destroyAllWindows()

