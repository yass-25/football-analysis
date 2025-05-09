# ⚽ Football Match Tactical Analysis App

> Projet de fin d'études — Télécom SudParis — VAP MAIA 2024-2025  
> Réalisé par Marinet Henein, Yasmine Ider et Laura Peyret  
> Tuteur : Julien Romero

## 🎯 Objectif

Développer une application complète d’analyse vidéo des matchs de football permettant :
- la détection automatique des joueurs et du ballon
- le suivi des trajectoires
- l’identification de l’équipe de chaque joueur
- la visualisation sur une carte tactique
- le calcul de statistiques (distance parcourue, vitesse, temps passé à gauche/droite)
- l’affichage de heatmaps et courbes de vitesse
- l’export des statistiques sous forme de CSV

## 🧠 Technologies utilisées

- **YOLOv8** : détection en temps réel des joueurs, ballons, points du terrain
- **DeepSORT** : suivi multi-objets avec attribution d’IDs
- **OpenCV / Numpy** : traitement d’image, homographie, visualisation
- **Streamlit** : interface utilisateur interactive
- **Pandas / Seaborn / Matplotlib** : statistiques et visualisation
- **PyTorch** : backend pour YOLOv8
- **Git / GitHub** : gestion de version

## 📁 Structure du projet

```
📦 Football-Analysis-using-Computer-Vision
├── Streamlit web app/               # Interface utilisateur
│   ├── main.py                      # Script principal Streamlit
│   ├── detection.py                 # Module de détection/tracking
│   └── outputs/                     # Vidéos générées
├── models/                          # Poids YOLOv8 personnalisés
│   └── Yolo8M Field Keypoints/
│       └── weights/
├── player_heatmaps/                # Heatmaps et courbes de vitesse
├── config pitch dataset.yaml       # Mapping des points terrain
├── config players dataset.yaml     # Mapping classes joueurs
├── pitch map labels position.json  # Coordonnées tactiques
├── README.md                       # Ce fichier
└── requirements.txt                # Dépendances Python
```

## ▶️ Lancer l’application

1. Clone ce repo :
```bash
git clone https://github.com/yass-25/football-analysis2.git
cd football-analysis2
```

2. Crée un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate
```

3. Installe les dépendances :
```bash
pip install -r requirements.txt
```

4. Lance Streamlit :
```bash
cd "Streamlit web app"
streamlit run main.py
```

## 📊 Résultats générés

- Carte tactique avec position des joueurs
- Heatmaps individuelles
- Courbes de vitesse
- Tableau récapitulatif des performances
- CSV téléchargeable

## ⚠️ Attention

- Les fichiers de modèles lourds (`.pt`, vidéos `.mp4`, dossiers `venv/`) ne sont **pas trackés** dans le dépôt.
- Pense à ajouter tes propres modèles dans `models/.../weights/`.

## 📎 À venir (perspectives)

- Détection des actions clés (passes, tirs)
- Suivi de la balle plus intelligent
- Ajout de classification d’événements
- Analyse de stratégie collective
