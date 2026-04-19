import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        entry.delete(0, "end")
        entry.insert(0, file_path)

def process_video():
    video_path = entry.get()
    if not video_path:
        print("Te rog să selectezi un fișier video!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nu am putut încărca video-ul.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    min_area = 50 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (1000, 650))
        fgmask = fgbg.apply(frame_resized)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(fgmask)

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                contour_mask = np.zeros_like(fgmask)
                cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
                mask = cv2.bitwise_or(mask, contour_mask)

        result = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)
        cv2.imshow('Isolated Object', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

ctk.set_appearance_mode("Dark")
app = ctk.CTk()
app.title("Procesare Video")
app.geometry("500x250")

label = ctk.CTkLabel(app, text="Selectează un fișier video:")
label.pack(pady=10)

entry = ctk.CTkEntry(app, width=300)
entry.pack(pady=5)

browse_button = ctk.CTkButton(app, text="Browse", command=select_file)
browse_button.pack(pady=5)

process_button = ctk.CTkButton(app, text="Rulează procesarea", command=process_video)
process_button.pack(pady=10)

app.mainloop()
