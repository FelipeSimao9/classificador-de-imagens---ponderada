import os
import shutil
import random

# 1. Configurações
BASE_DIR = '.'  # supondo que o script está em DATASETA-POND/
CLASSES = ['animais', 'cenarios', 'comida', 'pessoas']
SPLIT_RATIO = 0.8  # 80% train, 20% val
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# 2. Função para coletar todos os caminhos de imagem numa pasta, recursiva
def gather_images(src_dir):
    imgs = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMAGE_EXTS:
                imgs.append(os.path.join(root, f))
    return imgs

# 3. Limpa e cria pastas de destino
for split in ['train', 'val']:
    for cls in CLASSES:
        dst = os.path.join(BASE_DIR, split, cls)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)

# 4. Para cada classe, faz o split e copia as imagens
for cls in CLASSES:
    print(f'Processando classe "{cls}"…')
    src_dir = os.path.join(BASE_DIR, cls)
    all_imgs = gather_images(src_dir)
    random.shuffle(all_imgs)
    n_train = int(len(all_imgs) * SPLIT_RATIO)

    # copia para train/
    for img_path in all_imgs[:n_train]:
        fname = os.path.basename(img_path)
        dst_path = os.path.join(BASE_DIR, 'train', cls, fname)
        shutil.copy2(img_path, dst_path)

    # copia para val/
    for img_path in all_imgs[n_train:]:
        fname = os.path.basename(img_path)
        dst_path = os.path.join(BASE_DIR, 'val', cls, fname)
        shutil.copy2(img_path, dst_path)

    print(f'  → {len(all_imgs[:n_train])} images em train/{cls}, '
          f'{len(all_imgs[n_train:])} em val/{cls}')

print('✔ Dataset organizado e split concluído!')
