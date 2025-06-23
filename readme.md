# Classificador de imagens

## 1. Descrição do Projeto

Para realizar esse projeto, utilizei mais de 160 mil imagens e o projeto ocupa aproximadamente 44 gbs de espaço!
Antes de passar para os proximos pontos gostaria de destacar que aprendi muito com esse projeto e, no fim, eu acrescento as considerações finais dos meus principais aprendizados e o que eu faria diferente uma próxima vez.

Este projeto em Python oferece uma interface gráfica minimalista Tkinter (Utilizada pois acabou o 1 mes gratis do FreeSimpleGui) para:

- **Carregar** imagens de arquivos locais
- **Capturar** snapshots estáticos da webcam
- **Classificar** cada imagem em uma de quatro categorias pré-definidas e exibir os resultados

O objetivo é demonstrar a integração de um modelo de visão computacional treinado com uma aplicação desktop simples.

## 2. Classificador

- **Algoritmo:** Ultralytics YOLOv8 para classificação de imagens (`task='classify'`).
- **Backbone:** `yolov8n-cls.pt`, modelo leve pré-treinado e ajustado para nosso conjunto.
- **Categorias:**
  1. **animais**
  2. **cenarios**
  3. **comida**
  4. **pessoas**
- **Dataset de treinamento:**
  - **Estrutura:** pastas `train/` e `val/`, cada uma com quatro subpastas (uma por classe).
  - **Tamanho:** \~164 000 imagens no total (≈82 000 para treino e ≈82 000 para validação).
  - **Origem:** coletadas de fontes públicas e organizadas manualmente para representação balanceada (apesar de alguma variação de proporção).


[DEMONSTRAÇÃO NO YOUTUBE](https://www.youtube.com/watch?v=I_3-19G9PG0)


## 3. Instalação e Execução

1. **Criar ambiente virtual**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate   # macOS/Linux
   ```
2. **Instalar dependências**
   ```bash
   pip install --upgrade pip
   pip install ultralytics opencv-python Pillow
   ```
3. **Executar GUI**
   ```bash
   python gui.py
   ```
4. **Interagir**
   - **Abrir Imagem:** seleciona arquivo local (\*.jpg, \*.png, etc.)
   - **Capturar Webcam:** tira um snapshot estático da câmera
   - O resultado mostra as **5 classes** mais prováveis e suas confidências

## 4. Uso Avançado (Opcional)

- **Treinamento**
  ```bash
  python train.py
  ```
- **Avaliação**
  ```bash
  python evaluate.py
  ```
  Exibe tabela de *precision*, *recall*, *f1-score* e matriz de confusão.

## 5. Desempenho do Classificador

Avaliado em **82 641** imagens de validação, o modelo apresenta:

| Classe           | Precision | Recall | F1-Score | Suporte |
| ---------------- | --------- | ------ | -------- | ------- |
| animais          | 0.9719    | 0.9757 | 0.9738   | 8 186   |
| cenarios         | 0.9883    | 0.9867 | 0.9875   | 49 433  |
| comida           | 0.9808    | 0.9857 | 0.9832   | 20 200  |
| pessoas          | 0.9853    | 0.9739 | 0.9796   | 4 822   |
| **accuracy**     |           | 0.9846 |          | 82 641  |
| **macro avg**    | 0.9816    | 0.9805 | 0.9810   | 82 641  |
| **weighted avg** | 0.9847    | 0.9846 | 0.9846   | 82 641  |

### Matriz de Confusão

```text
[[ 7987   178    14     7]
 [  219 48778   375    61]
 [    7   280 19911     2]
 [    5   120     1  4696]]
```

## 6. Considerações Finais

Durante os testes, notou-se um **viés** forte para a classe **cenarios**:
Acredito que isso tenha acontecido devido a esmagadora diferença de quantidade de imagens dessa clase, sendo aproximadamente 50% de todas as imagens que treinou o modelo.

- Mesmo rostos humanos e animais eram frequentemente classificados como cenários quando o fundo era complexo.
- Para melhorar a precisão em `pessoas` e `animais`, recomenda-se:
  - **Destacar** o sujeito em primeiro plano com fundo uniforme (idealmente branco).
  - **Evitar** cenários rebuscados ao redor do objeto de interesse.

Isso reforça a importância de um dataset equilibrado e de condições de captura controladas quando se busca alta confiabilidade em classes minoritárias.