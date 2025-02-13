# Installation et Lancement du Projet

Ce guide explique comment installer et exécuter le projet sur Ubuntu.

## Prérequis

Avant de commencer, assurez-vous d'avoir :
- **Python 3.8+**
- **Git**
- **Docker** (pour Qdrant)
- **CUDA** (si utilisation GPU)

---

## 1. Cloner le projet

```bash
git clone https://github.com/cheetoszer/spe1.git
cd spe1
```

## 2. Installer les dépendances Python

Créez un environnement virtuel et installez les dépendances :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Installer et Lancer Qdrant via Docker

Téléchargez et exécutez l'image Docker de Qdrant :

```bash
docker pull qdrant/qdrant
docker run -d --name qdrant_container -p 6333:6333 qdrant/qdrant
```

Vérifiez que Qdrant tourne bien :

```bash
docker ps | grep qdrant
telnet localhost 6333
```

---

### 4. Installer et Lancer LlamaCpp (llama-server)

### Installation CPU :

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

### Installation GPU avec CUDA :

1. Assurez-vous d'avoir installé le toolkit CUDA depuis le site officiel de NVIDIA.
2. Clonez le dépôt LlamaCpp et compilez avec CUDA activé :

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

Si votre GPU n'est pas détecté automatiquement, ajoutez la Compute Capability manuellement (ex : RTX 3080 Ti) :

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build --config Release
```

### Lancement du serveur :

Assurez vous d'être dans le répertoire llamacpp.

```bash
./build/bin/llama-server -m <chemin_vers_mistral_instruct_v0.2.guuf_q5km.gguf>
```
Pour ajouter des layers du model sur le gpu utilisez :

```bash 
-ngl <nombre_de_layer>
```

Vérifiez qu'il fonctionne :

```bash
curl http://127.0.0.1:8080/v1/models
```

**Documentation complète de LlamaCpp :** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

---

## 5. Initialiser la base de données vectorielle

Exécutez toutes les cellules de `ingest.ipynb` pour transformer les documents en vecteurs et les stocker dans Qdrant.

---

## 6. Lancer l'application

Une fois la base Qdrant initialisée, lancez l'application :

```bash
python app.py
```