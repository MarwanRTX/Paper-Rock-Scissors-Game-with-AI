import torch
import cv2
import numpy as np
from torchvision import transforms
from model import CNN
import random
import time
from PIL import Image 
# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=3).to(device)
model.load_state_dict(torch.load("rps_model.pth", map_location=device))
model.eval()

class_names = ["Paper", "Rock", "Scissors"]

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_winner(player, ai):
    if player == ai:
        return "Draw"
    elif (player == "Rock" and ai == "Scissors") or \
         (player == "Paper" and ai == "Rock") or \
         (player == "Scissors" and ai == "Paper"):
        return "You Win!"
    else:
        return "AI Wins!"

cap = cv2.VideoCapture(0)

last_prediction_time = time.time()
cooldown = 3  # seconds between predictions
result_text = ""
ai_choice = ""
user_choice = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_flipped = cv2.flip(frame, 1)
    show_frame = frame_flipped.copy()

    h, w, _ = frame_flipped.shape
    roi_size = 300
    x1 = w // 2 - roi_size // 2
    y1 = h // 2 - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Draw ROI box
    cv2.rectangle(show_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Predict every cooldown seconds
    if time.time() - last_prediction_time >= cooldown:
        roi = frame_flipped[y1:y2, x1:x2]  # Crop ROI
        image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, pred = output.max(1)
            user_choice = class_names[pred.item()]

        ai_choice = random.choice(class_names)
        result_text = get_winner(user_choice, ai_choice)
        last_prediction_time = time.time()

    # Draw results
    cv2.putText(show_frame, f"Your Move: {user_choice}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(show_frame, f"AI Move: {ai_choice}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(show_frame, f"Result: {result_text}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(show_frame, "Place hand inside green box", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(show_frame, f"Next prediction in: {max(0, int(cooldown - (time.time() - last_prediction_time)))}",
            (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)


    cv2.imshow("Rock Paper Scissors Game", show_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
