import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import glob

# Ajout du dossier courant au PATH pour les imports relatifs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import params
from models.model_registry import MODEL_REGISTRY

# Configuration de la page Streamlit
st.set_page_config(page_title="🌦️ Rain Predictor Australia", layout="wide")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(params['model_output_dir'], model_name, f"{model_name}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# --- BARRE LATÉRALE ---
st.sidebar.title("⚙️ Configuration")
modeles_disponibles = list(MODEL_REGISTRY.keys())
selected_model_name = st.sidebar.selectbox("Choisir le modèle IA", modeles_disponibles)

model = load_model(selected_model_name)

if model is None:
    st.sidebar.error(f"Le modèle '{selected_model_name}' n'est pas entraîné.")
    st.stop()

st.sidebar.success(f"Modèle {selected_model_name.upper()} prêt")

# Lecture des métriques
report_pattern = os.path.join(params['model_output_dir'], selected_model_name, "report_*.txt")
report_files = glob.glob(report_pattern)
if report_files:
    latest_report = max(report_files, key=os.path.getctime)
    with open(latest_report, "r", encoding="utf-8") as f:
        report_text = f.read()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Performance du modèle")
    for line in report_text.split('\n'):
        if "accuracy:" in line:
            acc_val = line.split(":")[1].strip()
            st.sidebar.metric("Accuracy", f"{float(acc_val):.1%}")
        if "roc_auc:" in line:
            roc_val = line.split(":")[1].strip()
            st.sidebar.metric("ROC-AUC", f"{float(roc_val):.1%}")
    
    with st.sidebar.expander("Voir le rapport détaillé"):
        st.text(report_text)

# --- CORPS PRINCIPAL ---
st.title("🌦️ Simulateur de Pluie en Australie")
st.write("Modifiez les paramètres ci-dessous pour voir l'impact sur la prévision du lendemain.")

# 1. RÉCUPÉRATION DES FEATURES ATTENDUES
# On utilise feature_names_ stocké dans le modèle pour garantir l'ordre et le contenu
if hasattr(model, 'feature_names_') and model.feature_names_ is not None:
    features_attendues = model.feature_names_
else:
    # Fallback si le modèle est ancien
    st.error("Le modèle chargé ne contient pas la liste des features. Veuillez le ré-entraîner.")
    st.stop()

# 2. INTERFACE DE SAISIE (3 colonnes)
st.header("📍 Conditions Actuelles")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌡️ Température")
    temp3pm = st.slider("Température à 15h (°C)", -5.0, 48.0, 21.0, help="Forte corrélation avec la pression")
    temp9am = st.slider("Température à 9h (°C)", -5.0, 40.0, 17.0)
    rainfall = st.slider("Pluie aujourd'hui (mm)", 0.0, 100.0, 0.0)

with col2:
    st.subheader("💧 Humidité & Pression")
    humidity3pm = st.slider("Humidité à 15h (%)", 0, 100, 50, help="Indicateur majeur de pluie")
    humidity9am = st.slider("Humidité à 9h (%)", 0, 100, 68)
    pressure3pm = st.slider("Pression à 15h (hPa)", 980.0, 1040.0, 1015.0)
    pressure9am = st.slider("Pression à 9h (hPa)", 980.0, 1040.0, 1015.0)

with col3:
    st.subheader("☀️ Autres facteurs")
    windgustspeed = st.slider("Vitesse du vent max (km/h)", 0.0, 135.0, 40.0)
    sunshine = st.slider("Ensoleillement (heures)", 0.0, 15.0, 8.0)
    cloud3pm = st.slider("Couverture nuageuse 15h", 0, 9, 4)

# 3. CONSTRUCTION DE LA DONNÉE
# Dictionnaire complet avec valeurs par défaut neutres (médians du dataset)
input_data = {f: 0.0 for f in features_attendues}

# Mise à jour avec les sliders
input_data.update({
    'Temp3pm': temp3pm,
    'Temp9am': temp9am,
    'Rainfall': rainfall,
    'Humidity3pm': humidity3pm,
    'Humidity9am': humidity9am,
    'Pressure3pm': pressure3pm,
    'Pressure9am': pressure9am,
    'WindGustSpeed': windgustspeed,
    'Sunshine': sunshine,
    'Cloud3pm': cloud3pm,
    # Valeurs fixes pour les colonnes non présentes dans l'UI mais nécessaires
    'MinTemp': 12.0,
    'MaxTemp': 23.0,
    'Evaporation': 4.8,
    'WindSpeed9am': 14.0,
    'WindSpeed3pm': 19.0,
    'Cloud9am': 4.4,
    'Location': 2.0,
    'WindGustDir': 13.0,
    'WindDir9am': 13.0,
    'WindDir3pm': 14.0,
    'RainToday': 1 if rainfall > 1.0 else 0,
    'Day': 0.5,
    'Month': 0.0,
    'Year': 0.25
})

# Calcul dynamique des différences
input_data['Temp_diff'] = input_data['Temp3pm'] - input_data['Temp9am']
input_data['Pressure_diff'] = input_data['Pressure3pm'] - input_data['Pressure9am']
input_data['Humidity_diff'] = input_data['Humidity3pm'] - input_data['Humidity9am']

# Création du DataFrame avec l'ordre EXACT des features du modèle
df_input = pd.DataFrame([input_data])[features_attendues]

# 4. TRANSFORMATION IDENTIQUE AU TRAIN
def apply_preprocessing(df, model_obj):
    df_t = df.copy()
    
    # On identifie les colonnes à scaler (uniquement les numériques du config)
    colonnes_num = params.get('colonnes_numeriques', [])
    cols_asym = ["Rainfall", "Evaporation"]
    
    # 1. Variables asymétriques (Log + Robust)
    cols_to_robust = [c for c in cols_asym if c in df_t.columns]
    if cols_to_robust:
        df_t[cols_to_robust] = np.clip(df_t[cols_to_robust], 0, None)
        df_t[cols_to_robust] = np.log1p(df_t[cols_to_robust])
        # On ne passe que ces colonnes au scaler_robust
        df_t[cols_to_robust] = model_obj._scaler_robust.transform(df_t[cols_to_robust])
        
    # 2. Variables symétriques numériques (Standard)
    # On filtre pour ne pas passer les variables catégorielles/temporelles au scaler
    cols_to_standard = [c for c in colonnes_num if c in df_t.columns and c not in cols_asym]
    if cols_to_standard:
        df_t[cols_to_standard] = model_obj._scaler_standard.transform(df_t[cols_to_standard])
        
    return df_t

# --- PRÉDICTION ET RÉSULTATS ---
st.markdown("---")
st.header("🔮 Résultat de la Prédiction")

try:
    # Application du preprocessing
    df_final = apply_preprocessing(df_input, model)
    
    # Prédiction de probabilité
    try:
        proba = model.predict_proba(df_final)[0, 1]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilité de pluie", f"{proba:.1%}")
            if proba > 0.5:
                st.error("☔ Prévoyez de la pluie !")
            else:
                st.success("☀️ Journée ensoleillée en vue.")
        
        with c2:
            st.progress(float(proba))
            st.write(f"Niveau de confiance : {'Élevé' if proba > 0.8 or proba < 0.2 else 'Modéré'}")
            
    except (AttributeError, NotImplementedError):
        # Fallback pour SGD (hinge loss)
        prediction = model.predict(df_final)[0]
        if prediction == 1:
            st.error("☔ PRÉDICTION : PLUIE")
        else:
            st.success("☀️ PRÉDICTION : BEAU TEMPS")

except Exception as e:
    st.error(f"Erreur technique : {e}")

# --- ANALYSE ---
st.markdown("---")
st.header("📉 Courbes de Validation du Modèle")
image_path = os.path.join(params['model_output_dir'], selected_model_name, f"{selected_model_name}_courbes_auc.png")
if os.path.exists(image_path):
    st.image(image_path, caption=f"Visualisation ROC et Precision-Recall ({selected_model_name.upper()})", use_container_width=True)
else:
    st.info("Graphiques de performance non générés.")
