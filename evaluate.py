import os
from glob import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

# 1. Carregar o modelo treinado
model = YOLO("runs/classify/train5/weights/best.pt")

# 2. Mapeamento índice → nome de classe
names = {0: "animais", 1: "cenarios", 2: "comida", 3: "pessoas"}

# 3. Reunir todos os paths e labels verdadeiros
val_dir = "./val"
file_paths, y_true = [], []
for cls_idx, cls_name in names.items():
    cls_folder = os.path.join(val_dir, cls_name)
    for img_path in glob(os.path.join(cls_folder, "*.*")):
        file_paths.append(img_path)
        y_true.append(cls_idx)
y_true = np.array(y_true)

# 4. Inferir em chunks de, digamos, 256 imagens
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

y_pred = []
chunk_size = 256
for chunk in chunked(file_paths, chunk_size):
    results = model.predict(
        source=chunk,
        imgsz=160,
        batch=32,
        device="cpu",
        verbose=False
    )
    for r in results:
        # r.probs.top1 é o índice da classe com maior probabilidade
        pred_idx = int(r.probs.top1)        
        y_pred.append(pred_idx)

y_pred = np.array(y_pred)

# 5. Relatório de precisão/recall e matriz de confusão
print("=== Classification Report ===")
print(classification_report(
    y_true, y_pred,
    target_names=[names[i] for i in range(len(names))],
    digits=4
))

cm = confusion_matrix(y_true, y_pred)
print("=== Confusion Matrix ===")
print(cm)
