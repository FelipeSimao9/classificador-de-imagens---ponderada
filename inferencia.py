from ultralytics import YOLO
import os

# 1. Carregue o melhor peso gerado no treino
model = YOLO("runs/classify/train5/weights/best.pt")

# 2. Defina sua fonte de dados (pode ser uma imagem, vídeo ou pasta)
#    Exemplo: inferir em todas as imagens JPEG dentro de ./test_images/
source = "test_images"  

# 3. Execute a predição
results = model.predict(
    source=source,    # arquivo, pasta ou URL
    imgsz=160,        # mesmo tamanho do treino
    batch=32,         # tamanho de lote
    device="cpu",     # ou "0" para GPU
    save=True,        # salva as imagens de saída com a label desenhada
    save_txt=False,   # não há txt em classificação
    verbose=False     # True para logs detalhados
)

# 4. Os resultados são uma lista de objetos ClassifyResult
#    Cada resultado tem:
#      • path: caminho do arquivo inferido
#      • probs: tensor de probabilidades (len == nc)
#      • names: mapeamento índice → nome da classe
for r in results:
    # converte tensor para numpy
    probs = r.probs.cpu().numpy()  
    idx  = int(probs.argmax())      # índice da classe com maior prob.
    name = r.names[idx]             # nome da classe
    conf = probs[idx]               # probabilidade dessa classe
    print(f"{os.path.basename(r.path)} → {name} ({conf*100:.1f} %)")

# 5. As imagens com o label “<classe> <conf%>” serão salvas em:
#       runs/classify/predict/
print("Imagens anotadas em:", "runs/classify/predict/")
