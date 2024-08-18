# Desenvolvido por Levi Lucena - https://www.linkedin.com/in/levilucena/
from ultralytics import YOLO
from windowcapture import WindowCapture
from collections import defaultdict
import numpy as np
import cv2

# Configuração da captura de tela
offset_x = 0  # Deslocamento horizontal para capturar a tela (pode ser ajustado conforme necessário)
offset_y = 0  # Deslocamento vertical para capturar a tela (pode ser ajustado conforme necessário)
size = (1920, 1080)  # Tamanho da área de captura (largura, altura)
wincap = WindowCapture(size=size, origin=(offset_x, offset_y))  # Inicializa a captura de tela com o tamanho e origem especificados

# Inicialize o modelo YOLO
model = YOLO("yolov8n.pt")  # Carrega o modelo YOLO pré-treinado

# Dicionário para armazenar o histórico de rastreamento de objetos
track_history = defaultdict(list)
seguir = True  # Flag para determinar se o rastreamento deve continuar
deixar_rastro = False  # Flag para determinar se deve deixar um rastro dos objetos rastreados

# Configuração da janela
cv2.namedWindow("Tela", cv2.WINDOW_NORMAL)  # Cria uma janela para exibir os resultados
cv2.resizeWindow("Tela", 1920, 1080)  # Define o tamanho da janela de exibição

# Loop principal para captura e processamento de imagens
while True:
    # Captura uma imagem da tela usando o WindowCapture
    img = wincap.get_screenshot()

    # Realiza rastreamento ou detecção dependendo da flag 'seguir'
    if seguir:
        results = model.track(img, persist=True)  # Realiza rastreamento de objetos na imagem
    else:
        results = model(img)  # Apenas realiza detecção de objetos sem rastreamento

    # Copia a imagem original para exibição
    display_img = img.copy()
    for result in results:
        display_img = result.plot()  # Desenha as detecções e rastreamentos na imagem

        # Se o rastreamento estiver ativado e a opção de deixar rastro estiver habilitada
        if seguir and deixar_rastro:
            try:
                # Extrai as coordenadas das caixas delimitadoras e IDs dos objetos rastreados
                boxes = result.boxes.xywh.cpu()  # Converte para coordenadas (x, y, largura, altura)
                track_ids = result.boxes.id.int().cpu().tolist()  # Lista de IDs dos objetos rastreados

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]  # Obtém o histórico de rastreamento para o ID atual
                    track.append((float(x), float(y)))  # Adiciona a nova posição ao histórico
                    if len(track) > 30:  # Limita o histórico a 30 pontos para evitar excesso de memória
                        track.pop(0)

                    if len(track) > 1:  # Se houver mais de um ponto no histórico, desenha o rastro
                        points = np.array(track).astype(np.int32).reshape((-1, 1, 2))  # Converte o histórico para uma sequência de pontos
                        cv2.polylines(display_img, [points], isClosed=False, color=(230, 0, 0), thickness=2)  # Desenha o rastro na imagem
            except Exception as e:
                print(f"Erro ao rastrear: {e}")  # Exibe uma mensagem de erro caso ocorra uma exceção

    # Exibe a imagem processada na janela "Tela"
    cv2.imshow("Tela", display_img)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Fecha todas as janelas abertas
cv2.destroyAllWindows()
