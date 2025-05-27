#!/bin/bash

echo "🔁 Nettoyage du dépôt Git existant..."
rm -rf .git

echo "🚀 Réinitialisation du dépôt Git..."
git init
git checkout -b main

echo "📄 Création du fichier .gitignore..."
cat > .gitignore <<EOF
# Environnements virtuels
env/
env_backup/

# Données brutes à ignorer
datas/
seam_carving_project/CUB_200_2011/

# Cache et temporaires
__pycache__/
*.py[cod]
*.so
*.log
*.tmp

# VSCode
.vscode/

# Jupyter
.ipynb_checkpoints/

# Variables d'environnement
.env
EOF

echo "➕ Ajout des fichiers au dépôt..."
git add .
git commit -m "Clean initial commit"

echo "🔗 Lien avec le dépôt distant GitHub..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/BenaribIlyes/seam_carving_project.git

echo "📤 Push vers GitHub (main)..."
git push -f -u origin main

echo "✅ Déploiement terminé avec succès !"
