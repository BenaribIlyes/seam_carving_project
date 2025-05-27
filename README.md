# Traitement Intelligent d’Images par Seam Carving

Ce projet rassemble plusieurs démonstrations et outils Python/Jupyter pour expérimenter le seam carving et ses usages :
- réduction/extension de taille d’image  
- suppression d’objets guidée par masque  
- intégration de seam carving comme opération de pooling dans un CNN  
- utilitaires (dessin de masque, configuration)

---

## Fichiers de configuration

- setup_env.sh  
  Script bash pour créer et activer un environnement virtuel et installer toutes les dépendances nécessaires (OpenCV, NumPy, scikit-image, PyTorch, joblib, numba, …).

---

## Modules principaux

- class_SeamCarver.py  
  - Classe SeamCarver avec :  
    - calcul de cartes d’énergie (l1, l2, Sobel, HoG, entropie, saliency, fusion Sobel+saliency)  
    - recherche de seams à moindre énergie (vertical / horizontal / optimal)  
    - suppression (remove_seam) et insertion (add_seam) de seams (accéléré par Numba)  
    - historique et visualisation des seams retirés  
    - découpage adaptatif pour suppression d’objet via masque  

- seam_carving_for_CNN.py  
  - Module PyTorch incluant :  
    - couche SeamCarvingPooling pour réduire intelligemment les feature maps  
    - deux architectures d’exemple :  
      - CNNWithSeamCarving  
      - CNNWithMaxPooling (réseau de référence)

---

## Notebooks de démonstration

Tous les notebooks s’appuient sur class_SeamCarver.py et illustrent ses usages.

1. seam_carving_demo.ipynb  
   - Chargement d’une image couleur  
   - Réduction progressive de largeur/hauteur par seam carving  
   - Comparaison visuelle entre méthodes (Sobel, saliency, HoG)  
   - Affichage interactif de l’historique des seams  

2. scaling_up_demo.ipynb  
   - Upsizing vs downsizing d’images  
   - Comparaison avant/après interpolation bilinéaire  
   - Gestion des indices pour insertion de seams  

3. object_removal.ipynb  
   - Import d’un masque binaire peint à la main  
   - Boucle de suppression de seams traversant la région masquée  
   - Effacement automatique de l’objet ciblé  

4. CNN_with_seam_carving_demo.ipynb  
   - Insertion de SeamCarvingPooling dans un pipeline PyTorch  
   - Entraînement / évaluation sur un petit dataset toy  
   - Comparaison des performances (accuracy, temps) vs max-pooling  

---

## Outils utilitaires

- selection_objet.py  
  Interface OpenCV interactive pour dessiner un masque :  
  - clic & glisser pour peindre  
  - touches + / - pour régler la taille du pinceau  
  - s pour sauvegarder le masque  

- setup_env.sh  
  Installe l’environnement et les packages Python :
  ```
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

---

## Fonctionnalités clés

1. Cartes d’énergie avancées  
   - Norme L1/L2 (Sobel)  
   - Entropie / saliency (CV2)  
   - HoG (skimage & parallélisé)  
   - Fusion Sobel + Saliency  

2. Suppression & insertion de seams  
   - Optimisations Numba  
   - Seams verticales & horizontales  
   - Stratégie “seam optimal”  

3. Redimensionnement intelligent  
   - Méthodes seam_carve, upsize, downsize  

4. Suppression d’objets guidée  
   - Masque binaire → suppression ciblée  

5. Intégration dans CNN  
   - SeamCarvingPooling pour PyTorch  
   - Plug-in facile dans n’importe quel modèle  

6. Démos interactives  
   - Notebooks Jupyter  
   - Outil de dessin de masque  

---

## Installation rapide

```
git clone https:https://github.com/BenaribIlyes/seam_carving_project
cd image_processing_project
bash setup_env.sh
jupyter lab
```

## 🔽 Cloner uniquement la branche `main`

```bash
git clone --single-branch --branch main https://github.com/BenaribIlyes/seam_carving_project.git
cd seam_carving_project

---

**Contribuer**  
PRs bienvenues pour ajouter :  
- Nouveaux modes d’énergie (TV, saliency par DL, …)  
- Accélération GPU (CUDA)  
- Benchmarks sur datasets standards
