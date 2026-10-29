# Feature Visualization — InceptionV1

Reimplementación práctica de *Feature Visualization* (Olah et al., Distill 2017) en PyTorch, con los pesos originales de InceptionV1 (vía `torch-lucent`), y el atlas completo de visualizaciones generado con ella.

## Contenido

- `feature_visualization.ipynb`: el notebook, con sus salidas.
- `inceptionv1_atlas/`: resultados de la run completa (preset `full`).
  - `<capa>/channel|neuron|negative/NNNN.jpg`: imagen optimizada por canal (128 px, 1024 pasos).
  - `<capa>/dataset/NNNN.jpg`: top-4 imágenes de Imagenette (val) que más activan cada canal.
  - `logits/logit/NNNN.jpg`: una imagen por clase de ImageNet (1000).
  - `<capa>/stats.json`: valor del objetivo alcanzado por cada imagen.
  - `sprites/`: hojas de sprites por capa y objetivo (tiles de 64 px).
  - `manifest.json`: índice de todo lo anterior (capas, ramas, rutas, valores).
- El atlas también está en un solo `.zip` en la sección **Releases**.

Capas: `conv2d0–2`, `mixed3a–mixed5b` y los bottlenecks 3x3/5x5 de cada bloque Inception (31.259 imágenes en total).

## Correrlo

Necesita `torch`, `torchvision`, `torch-lucent` (`pip install --no-deps torch-lucent`), `matplotlib`, `numpy`, `pillow` y `tqdm`. Se recomienda GPU: en una RTX 4060 el tutorial tarda ~1 h 20 min con `STEPS = 2048` y el atlas `full` ~13–14 h. El dataset Imagenette se descarga solo.
