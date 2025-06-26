from ultralytics import YOLO
import cv2
from gtts import gTTS
import os
import pygame
import time
from datetime import datetime
import ollama
import tkinter as tk
import threading

model = YOLO("asl-yolov8/sign-language-model/weights/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

def show_meaning_window(text, duration=60):
    def close_after_delay():
        time.sleep(duration)
        window.destroy()

    window = tk.Tk()
    window.title("Significado")
    window.geometry("700x500")

    label = tk.Label(window, text=text, wraplength=380, font=("Arial", 12), justify="left")
    label.pack(padx=20, pady=20)

    threading.Thread(target=close_after_delay, daemon=True).start()
    window.mainloop()

def get_meaning_from_ollama(word):
    prompt = f"Explica o significado da palavra '{word}' em português de Portugal, de uma maneira simples, em 1 ou 2 parágrafos."

    response = ollama.chat(
        model='mistral',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content'].strip()

letra_atual = None
tempo_inicial = 0
letras_confirmadas = []

def speak(word):
    meaning = get_meaning_from_ollama(word)
    if meaning:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('significados.txt', 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {word}: {meaning}\n")
        print(f"Texto guardado: {meaning}")

    full_text = f"{word}. Significado: {meaning}"
    print(full_text)

    tts = gTTS(text=full_text, lang='pt', tld='pt')
    tts.save("temp_audio.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("temp_audio.mp3")
    pygame.mixer.music.play()

    show_thread = threading.Thread(target=show_meaning_window, args=(full_text,))
    show_thread.start()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()
    os.remove("temp_audio.mp3")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame.")
        break

    frame = cv2.flip(frame, 1)

    resultado = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = resultado[0].plot()

    names = model.names
    boxes = resultado[0].boxes
    letra_detetada = None

    if boxes and len(boxes) > 0:
        cls_id = int(boxes.cls[0])
        letra_detetada = names[cls_id]

        if letra_detetada == letra_atual:
            tempo_decorrido = time.time() - tempo_inicial
            if tempo_decorrido >= 2:
                letras_confirmadas.append(letra_detetada)
                print(f"Letra confirmada: {letra_detetada}")
                letra_atual = None
        else:
            letra_atual = letra_detetada
            tempo_inicial = time.time()
    else:
        letra_atual = None

    texto_output = ''.join(letras_confirmadas)
    cv2.putText(annotated_frame, f"Texto: {texto_output}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("ASL Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        letras_confirmadas = []
        print("Texto limpo.")

    elif key == 13:
        texto_final = ''.join(letras_confirmadas)
        if texto_final:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("output.txt", "a", encoding="utf-8") as file:
                file.write(f"[{timestamp}] {texto_final}\n")
            print(f"Texto guardado: {texto_final}")
            speak(texto_final)
            letras_confirmadas = []

cap.release()
cv2.destroyAllWindows()
