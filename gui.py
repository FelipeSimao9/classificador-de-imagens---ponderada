import os
import tempfile
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# 1) Carrega modelo treinado
model = YOLO('runs/classify/train5/weights/best.pt')

# 2) Mapeamento de classes (conforme data.yaml)
names = {0: 'animais', 1: 'cenarios', 2: 'comida', 3: 'pessoas'}

# 3) Função que retorna as top-k predições
def classify_image(path, topk=5):
    results = model.predict(source=path, imgsz=160, device='cpu', verbose=False)
    r = results[0]
    # r.probs.top5 é lista de índices, r.probs.top5conf é lista de confidências
    preds = []
    for idx, conf in zip(r.probs.top5, r.probs.top5conf):
        preds.append((names[int(idx)], float(conf)))
    return preds  # ex: [('pessoas',0.55),('comida',0.20),...]

# 4) Exibe frame OpenCV em um Label do Tkinter
def show_frame_on_label(frame, label):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((400, 400))
    tk_img = ImageTk.PhotoImage(pil)
    label.config(image=tk_img)
    label.image = tk_img

# 5) Callback “Abrir Imagem”
def on_open_image():
    path = filedialog.askopenfilename(
        title='Selecione uma imagem',
        filetypes=[('Imagens', '*.png;*.jpg;*.jpeg;*.bmp')]
    )
    if not path:
        return
    frame = cv2.imread(path)
    if frame is None:
        messagebox.showerror('Erro', 'Não foi possível ler a imagem.')
        return
    show_frame_on_label(frame, label_img)
    preds = classify_image(path, topk=5)
    # Monta texto multiline com ranking
    text = "\n".join(f"{i+1}. {cls} ({conf*100:.1f}%)"
                     for i, (cls, conf) in enumerate(preds))
    label_res.config(text=text)

# 6) Callback “Capturar Webcam”
def on_capture_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror('Erro', 'Não foi possível acessar a webcam.')
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror('Erro', 'Falha ao capturar da webcam.')
        return
    show_frame_on_label(frame, label_img)
    fd, tmp = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    cv2.imwrite(tmp, frame)
    preds = classify_image(tmp, topk=5)
    os.remove(tmp)
    text = "\n".join(f"{i+1}. {cls} ({conf*100:.1f}%)"
                     for i, (cls, conf) in enumerate(preds))
    label_res.config(text=text)

# 7) Montagem da janela Tkinter
root = tk.Tk()
root.title('Classificador de Imagens')

tk.Button(root, text='Abrir Imagem',    width=20, command=on_open_image).pack(pady=(10, 5))
tk.Button(root, text='Capturar Webcam', width=20, command=on_capture_webcam).pack(pady=5)

label_img = tk.Label(root)
label_img.pack(pady=10)

frame_res = tk.Frame(root)
frame_res.pack(pady=(5, 10))
tk.Label(frame_res, text='Resultados:', font=('Arial', 12)).pack(anchor='w')
label_res = tk.Label(frame_res, text='', font=('Arial', 12), justify='left')
label_res.pack(anchor='w')

tk.Button(root, text='Sair', width=10, command=root.destroy).pack(pady=(0,10))

root.mainloop()
