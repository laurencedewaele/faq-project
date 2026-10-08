FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système (optionnel mais utile pour torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app.py .
COPY config.py .
COPY models.py .
COPY faq_loader.py .
COPY embeddings.py .

# Copier les données
COPY data/ data/

# Port utilisé par HF Spaces (7860 par défaut)
EXPOSE 7860

# Lancer l'application
# HF Spaces redirige le port 7860 vers le port de l'interface web
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
