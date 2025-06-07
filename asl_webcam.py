from ultralytics import YOLO
import cv2
import time
from datetime import datetime

model = YOLO("asl-yolov8/sign-language-model/weights/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

letra_atual = None
tempo_inicial = 0
letras_confirmadas = []

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
            letras_confirmadas = []

cap.release()
cv2.destroyAllWindows()
