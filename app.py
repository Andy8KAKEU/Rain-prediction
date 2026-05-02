import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ajout du dossier courant au PATH pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import params
from models.model_registry import MODEL_REGISTRY

st.set_page_config(page_title="Prédiction de Pluie", layout="wide")

st.title("🌦️ Simulateur de Pluie (What-If Scenarios)")
st.write("Modifiez les paramètres météorologiques pour voir comment la prédiction évolue en temps réel selon le modèle sélectionné.")

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.header("1. Configuration")

# Sélection du modèle
modeles_disponibles = list(MODEL_REGISTRY.keys())
selected_model_name = st.sidebar.selectbox("Choisissez un modèle :", modeles_disponibles)

# Chargement du modèle
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(params['model_output_dir'], model_name, f"{model_name}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model(selected_model_name)

if model is None:
    st.sidebar.error(f"Le modèle '{selected_model_name}' n'a pas encore été entraîné. Lancez run.py d'abord.")
    st.stop()

st.sidebar.success(f"Modèle {selected_model_name.upper()} chargé !")

# --- LECTURE DES MÉTRIQUES DU MODÈLE ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performances du modèle")

import glob
report_pattern = os.path.join(params['model_output_dir'], selected_model_name, "report_*.txt")
report_files = glob.glob(report_pattern)

if report_files:
    # Prendre le rapport le plus récent
    latest_report = max(report_files, key=os.path.getctime)
    
    acc = None
    roc = None
    with open(latest_report, "r", encoding="utf-8") as f:
        report_text = f.read()
        # Petite extraction rapide de l'accuracy et du ROC-AUC
        for line in report_text.split('\n'):
            if line.strip().startswith("accuracy:"):
                acc = line.split(":")[1].strip()
            elif line.strip().startswith("roc_auc:"):
                roc = line.split(":")[1].strip()
                
    if acc and roc:
        col_m1, col_m2 = st.sidebar.columns(2)
        col_m1.metric("Accuracy", f"{float(acc):.1%}")
        col_m2.metric("ROC-AUC", f"{float(roc):.1%}")
        
    with st.sidebar.expander("📝 Rapport complet et Validation Croisée"):
        st.text(report_text)
else:
    st.sidebar.info("Aucun rapport d'expérimentation récent trouvé pour ce modèle.")

# --- INTERFACE PRINCIPALE ---
st.header("2. Paramètres Météo")
st.write("Ajustez les 8 variables les plus importantes. Les autres variables sont gelées à des valeurs moyennes.")

col1, col2 = st.columns(2)

with col1:
    pressure3pm = st.slider("Pression à 15h (hPa)", 980.0, 1040.0, 1015.0)
    humidity3pm = st.slider("Humidité à 15h (%)", 0, 100, 50)
    windgustspeed = st.slider("Vitesse des rafales (km/h)", 0.0, 130.0, 40.0)
    sunshine = st.slider("Ensoleillement (heures)", 0.0, 15.0, 8.0)

with col2:
    pressure9am = st.slider("Pression à 9h (hPa)", 980.0, 1040.0, 1015.0)
    windspeed3pm = st.slider("Vitesse du vent à 15h (km/h)", 0.0, 90.0, 20.0)
    cloud3pm = st.slider("Couverture nuageuse à 15h (octats)", 0, 9, 4)
    rainfall = st.slider("Précipitations du jour (mm)", 0.0, 100.0, 0.0)

# --- RECONSTRUCTION DE LA DONNÉE ---
# On doit créer un DataFrame avec TOUTES les colonnes numériques attendues par le Scaler
valeurs_defaut = {
    'MinTemp': 12.0,
    'MaxTemp': 23.0,
    'Rainfall': rainfall,
    'Evaporation': 5.0,
    'Sunshine': sunshine,
    'WindGustSpeed': windgustspeed,
    'WindSpeed9am': 14.0,
    'WindSpeed3pm': windspeed3pm,
    'Humidity9am': 68.0,
    'Humidity3pm': humidity3pm,
    'Pressure9am': pressure9am,
    'Pressure3pm': pressure3pm,
    'Cloud9am': 4.0,
    'Cloud3pm': cloud3pm,
    'Temp9am': 17.0,
    'Temp3pm': 21.0
}

# Assurons-nous que l'ordre des colonnes correspond exactement à celui du param.yaml
colonnes_num = params['colonnes_numeriques']
df_input = pd.DataFrame([valeurs_defaut])[colonnes_num]

# --- PRÉDICTION ---
st.header("3. Résultat de la Prédiction")

try:
    # On utilise le scaler qui est déjà stocké dans le modèle entraîné !
    X_scaled = model._scaler.transform(df_input)
    
    # On essaie d'obtenir une probabilité
    try:
        proba = model.predict_proba(X_scaled)[0, 1]
        st.metric(label="Probabilité qu'il pleuve demain", value=f"{proba:.1%}")
        
        # Barre de progression visuelle
        st.progress(float(proba))
        
        if proba > 0.5:
            st.warning("Préparez votre parapluie ! ☔")
        else:
            st.success("Il devrait faire beau ! ☀️")
            
    except AttributeError:
        # Si le modèle ne supporte pas predict_proba (ex: SGD avec hinge loss)
        pred = model.predict(X_scaled)[0]
        st.write("Ce modèle ne fournit pas de probabilités (SVM pur).")
        if pred == 1:
            st.error("Prédiction : PLUIE 🌧️")
        else:
            st.success("Prédiction : BEAU TEMPS ☀️")

except Exception as e:
    st.error(f"Erreur lors de la prédiction : {e}")

# --- COURBES D'ÉVALUATION ---
st.markdown("---")
st.header("4. Courbes d'évaluation du modèle")

image_path = os.path.join(params['model_output_dir'], selected_model_name, f"{selected_model_name}_courbes_auc.png")
if os.path.exists(image_path):
    st.image(image_path, caption=f"Courbes ROC et Precision-Recall pour {selected_model_name.upper()}", use_container_width=True)
else:
    st.info("Les courbes d'évaluation n'ont pas encore été générées pour ce modèle.")
