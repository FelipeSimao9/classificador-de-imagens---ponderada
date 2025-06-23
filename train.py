from ultralytics import YOLO
import os
print("PWD:", os.getcwd())
print("Arquivos aqui:", os.listdir())
print("Existe './data.yaml'?", os.path.isfile('./data.yaml'))
print(open('data.yaml').read())  # opcional: mostra seu YAML

model = YOLO("yolov8n-cls.pt")
results = model.train(
    data=".",          # <<-- pasta raiz do projeto
    epochs=2,           # só 2 épocas → ~1h + ~45 min = 1h45min
    imgsz=160,          # reduz um pouco a resolução
    batch=32,
    cache=True,         # acelera leitura dos arquivos
    workers=8,
    patience=2          # early-stop se não melhorar em 2 épocas
)

print("Treino concluído! Pesos em:", results.best)
