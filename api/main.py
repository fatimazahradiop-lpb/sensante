# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()
# 1. EN PREMIER : Les imports
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 2. EN DEUXIÈME : La création de l'application (C'EST ICI qu'on rajoute app = FastAPI())
app = FastAPI()

# 3. EN TROISIÈME : Le montage des dossiers statiques
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# 4. EN DERNIER : Vos routes et fonctions
@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse("frontend/index.html")

# Client Groq (charge au demarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")

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
from fastapi.middleware.cors import CORSMiddleware
class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)
class ExplainOutput(BaseModel):
    explication : str = Field (... ,description =" Explication en francais ")
    modele_llm : str = Field (default ="llama -3.1 -8b- instant ",description =" Modele LLM utilise ")

SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple, comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )
    
    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} (probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
        modele_used = "llama-3.1-8b-instant"
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
        modele_used = "aucun"
        
    return ExplainOutput(
        explication=explication,
        modele_llm=modele_used
    )

# Autoriser les requetes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

