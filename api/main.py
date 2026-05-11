# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    """Données d'entrée : les symptômes d'un patient."""
    age: int = Field(..., ge=0, le=120, description="Age en années")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Température en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Présence de toux")
    fatigue: bool = Field(..., description="Présence de fatigue")
    maux_tete: bool = Field(..., description="Présence de maux de tête")
    region: str = Field(..., description="Région du Sénégal")

class DiagnosticOutput(BaseModel):
    """Données de sortie : le résultat du diagnostic."""
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# Créer l'application
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

# --- Charger le modèle et les encodeurs au démarrage ---
# Note : Il est préférable de gérer les erreurs de chargement de fichiers
try:
    print("Chargement du modèle...")
    model = joblib.load("models/model.pkl")
    le_sexe = joblib.load("models/encoder_sexe.pkl")
    le_region = joblib.load("models/encoder_region.pkl")
    # feature_cols = joblib.load("models/feature_cols.pkl") # Optionnel si non utilisé explicitement
    print(f"Modèle chargé : {type(model).__name__}")
    print(f"Classes détectées : {list(model.classes_)}")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")

# Route de base
@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {"status": "ok", "message": "SenSante API is running"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """Prédire un diagnostic à partir des symptômes."""
    
    # 1. Encoder les variables catégoriques
    try:
        # .strip() pour éviter les erreurs d'espaces inutiles
        sexe_enc = le_sexe.transform([patient.sexe.strip()])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Sexe invalide: {patient.sexe}. Utiliser M ou F.")

    try:
        region_enc = le_region.transform([patient.region.strip()])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Région inconnue: {patient.region}")

    # 2. Construire le vecteur de features
    # Assurez-vous que l'ordre correspond exactement à l'entraînement du modèle
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Prédire
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    # 4. Déterminer le niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    #5.Générer la recommandation (Attention aux espaces dans vos clés dict)
    messages = {
        "palu": "Suspicion de paludisme. Consultez un médecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandés.",
        "typh": "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain": "Pas de pathologie détectée. Continuez à surveiller."
    }

    # 6. Renvoyer le résultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin pour un diagnostic approfondi.")
    )