# Depth Anything 3 → GLB pour Blender

Interface Gradio simple : une **image en entrée**, deux **fichiers `.glb` en sortie**
(mesh texturé + nuage de points), prêts à importer dans Blender.

Basé sur [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance),
modèle `DA3-LARGE-1.1`.

## Pré-requis

- **Windows 10/11**
- **GPU NVIDIA** + driver récent (testé sur RTX 3070 Ti, 8 Go VRAM — l'inférence n'utilise que ~2,5 Go)
- **Python 3.10** — https://www.python.org/downloads/release/python-31011/ (cocher *Add python.exe to PATH*)
- **Git** — https://git-scm.com/download/win

Sans GPU NVIDIA, ça tourne sur CPU (beaucoup plus lent).

## Installation

Double-clic sur **`install.bat`** (une seule fois).

Ça crée un environnement Python isolé, installe PyTorch CUDA 12.1, télécharge le code
Depth Anything 3 et toutes les dépendances. ~3-4 Go de téléchargement, prévois 10-15 min.

## Lancement

Double-clic sur **`run.bat`** → l'interface s'ouvre sur http://127.0.0.1:7860

> Au tout premier lancement, le modèle (~1,4 Go) se télécharge automatiquement.

## Utilisation

1. Charge une image (PNG avec **transparence** = trou réel dans le mesh).
2. Clique **Générer GLB**.
3. Récupère les fichiers dans `outputs\<date>\` :
   - `mesh.glb` — surface 3D texturée
   - `pointcloud\scene.glb` — nuage de points
   - `depth_16bit.png` — carte de profondeur 16-bit (height map)
   - `depth_color.png` — aperçu colorisé de la profondeur
4. Dans Blender : `File > Import > glTF 2.0 (.glb)`.

### Carte de profondeur dans Blender (displacement)

`depth_16bit.png` sert de height map : sur un plan subdivisé, ajoute un modifier
**Displace**, charge cette image comme texture → relief par la profondeur (plus clair = plus loin).

### Réglages

| Réglage | Effet |
|---|---|
| **Résolution** | Plus haut = plus de détail (et plus de VRAM). 504 par défaut. |
| **Cadre complet** | Coché = relief entier sans découpe. Décoché = découpe le fond. |
| **Seuil confiance** | (décoché) plus haut = moins de bruit. |
| **Sensibilité bords** | (décoché) plus bas = coupe plus aux ruptures de profondeur. |
| **Max points** | Densité du nuage de points. |

## Modèle plus lourd (optionnel)

VRAM dispo ? Pour plus de qualité, avant `run.bat` :

```bat
set DA3_MODEL=depth-anything/DA3-GIANT-1.1
```

## Licence

Code du modèle sous Apache-2.0 (voir le dépôt Depth Anything 3 d'origine).
