from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2

# Origens possíveis: image, screenshot, URL, video, YouTube, Streams -> ESP32 / Intelbras / Cameras On-Line
# Mais informações em https://docs.ultralytics.com/modes/predict/#inference-sources

# Abre a captura de vídeo usando a câmera padrão
cap = cv2.VideoCapture(0)
# Se necessário, pode usar outra câmera: cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Usa o modelo YOLO
# Model       size    mAPval  Speed       Speed       params  FLOPs
#            (pixels) 50-95  CPU ONNX A100 TensorRT   (M)     (B)
#                        (ms)        (ms)
# YOLOv8n     640     37.3   80.4        0.99        3.2     8.7
# YOLOv8s     640     44.9   128.4       1.20        11.2    28.6
# YOLOv8m     640     50.2   234.7       1.83        25.9    78.9
# YOLOv8l     640     52.9   375.2       2.39        43.7    165.2
# YOLOv8x     640     53.9   479.1       3.53        68.2    257.8

# Carrega o modelo YOLOv8n (versão mais leve)
model = YOLO("yolov8n.pt")

# Dicionário para armazenar o histórico de rastreamento
track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = False  # Define se deve deixar um rastro das trajetórias

while True:
    # Captura um frame da câmera
    success, img = cap.read()

    if success:
        # Realiza rastreamento se a variável 'seguir' estiver ativa
        if seguir:
            results = model.track(img, persist=True)
        else:
            # Caso contrário, apenas realiza a detecção
            results = model(img)

        # Processa a lista de resultados
        for result in results:
            # Visualiza os resultados no frame
            img = result.plot()

            if seguir and deixar_rastro:
                try:
                    # Obtém as caixas delimitadoras e IDs dos rastreamentos
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Plota as trajetórias
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # ponto central (x, y)
                        if len(track) > 30:  # mantém 30 rastros
                            track.pop(0)

                        # Desenha as linhas de rastreamento
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except:
                    pass

        # Exibe a imagem com as anotações
        cv2.imshow("Tela", img)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
print("Desligando")
