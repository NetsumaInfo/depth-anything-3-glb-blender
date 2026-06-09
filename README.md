# Depth Anything 3 → GLB pour Blender

Interface Gradio simple : une **image en entrée**, des **fichiers `.glb` en sortie**
(au choix : mesh texturé, nuage de points, combiné mesh+nuage, plan+relief), prêts à
importer dans Blender.

Basé sur [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance),
modèle par défaut `DA3NESTED-GIANT-LARGE-1.1` (1,40 B, depth métrique).

## Pré-requis

- **Windows 10/11**
- **GPU NVIDIA** + driver récent (testé sur RTX 3070 Ti, 8 Go VRAM). Le modèle par défaut
  (1,40 B) dépasse 8 Go et déborde en RAM partagée → ça marche mais l'inférence prend
  ~10-15 s/image. Pour ~1 s/image, baisse la résolution ou passe à un modèle plus léger (voir plus bas).
- **Python 3.10** — https://www.python.org/downloads/release/python-31011/ (cocher *Add python.exe to PATH*)
- **Git** — https://git-scm.com/download/win

Sans GPU NVIDIA, ça tourne sur CPU (beaucoup plus lent).

## Installation

Double-clic sur **`install.bat`** (une seule fois).

Ça crée un environnement Python isolé, installe PyTorch CUDA 12.1, télécharge le code
Depth Anything 3 et toutes les dépendances. ~3-4 Go de téléchargement, prévois 10-15 min.

## Lancement

Double-clic sur **`run.bat`** → l'interface s'ouvre sur http://127.0.0.1:7860

> Au tout premier lancement, le modèle (~6,3 Go) se télécharge automatiquement.

## Utilisation

1. Charge une image (PNG avec **transparence** = trou réel dans le mesh).
2. Clique **Générer GLB**.
3. Coche les **sorties à générer** (combiné et plan+relief cochés par défaut).
4. Récupère les fichiers dans `outputs\<date>\` :
   - `mesh.glb` — surface 3D texturée seule
   - `pointcloud\scene.glb` — nuage de points seul
   - `combined.glb` — **mesh + nuage dans un seul fichier** (même repère, alignés)
   - `plane_relief.glb` — **plan image plat + relief 3D** dans un seul fichier
   - `depth_16bit.png` — carte de profondeur 16-bit (height map)
   - `depth_gray.png` — carte de profondeur noir & blanc 8-bit
   - `depth_color.png` — aperçu colorisé de la profondeur
5. Dans Blender : `File > Import > glTF 2.0 (.glb)`.

### Carte de profondeur dans Blender (displacement)

`depth_16bit.png` sert de height map : sur un plan subdivisé, ajoute un modifier
**Displace**, charge cette image comme texture → relief par la profondeur (plus clair = plus loin).

### Réglages

| Réglage | Effet |
|---|---|
| **Sorties à générer** | Coche les `.glb` / depth maps voulus. Le mesh est calculé une fois et réutilisé. |
| **Résolution** | Plus haut = plus de détail (et plus de VRAM). 504 par défaut. |
| **Cadre complet** | Coché = relief entier sans découpe. Décoché = découpe le fond. |
| **Seuil confiance** | (décoché) plus haut = moins de bruit. |
| **Sensibilité bords** | (décoché) plus bas = coupe plus aux ruptures de profondeur. |
| **Max points** | Densité du nuage de points. |

## Changer de modèle (optionnel)

Le modèle par défaut est le plus lourd (`DA3NESTED-GIANT-LARGE-1.1`, 1,40 B, meilleure
qualité). Sur 8 Go de VRAM il déborde en RAM partagée → correct mais lent (~10-15 s/image).
Pour aller plus vite : baisse la **Résolution** dans l'interface, ou repasse à un modèle
plus léger avant `run.bat` :

```bat
set DA3_MODEL=depth-anything/DA3-GIANT-1.1      REM 1,15 B
set DA3_MODEL=depth-anything/DA3-LARGE-1.1      REM 0,35 B, ~1 s/image
```

## Licence

Code du modèle sous Apache-2.0 (voir le dépôt Depth Anything 3 d'origine).
