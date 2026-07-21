# =========================================
# 🛠️ SYSTEM COMPATIBILITY & CONFIG
# =========================================
import sys
import os
import pydantic
from dotenv import load_dotenv, find_dotenv

# 🛑 THESE MUST BE AT THE VERY TOP BEFORE ANY ML IMPORTS
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 🔥 THIS IS THE CRITICAL LINE: It actually reads the .env file!
load_dotenv(find_dotenv(), override=True)

# 1. THE MAGIC BRIDGE: Fixes 'ModuleNotFoundError' for langchain_core.pydantic_v1
try:
    from pydantic import v1 as pydantic_v1
except ImportError:
    import pydantic as pydantic_v1

# Inject the existing pydantic module into the missing path
sys.modules["langchain_core.pydantic_v1"] = pydantic_v1

# 2. PATCH FOR PYDANTIC V2 Conflicts
try:
    if pydantic.__version__.startswith("2"):
        pydantic_v1.main.ModelMetaclass.__pydantic_model__ = None
except Exception:
    pass


# =========================================
# 📦 CORE FRAMEWORK IMPORTS
# =========================================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import base64
import io
import asyncio
from datetime import datetime
from PIL import Image

st.set_page_config(
    layout='wide',
    page_icon='🩺',
    page_title='DiagnoVix'
)

# =========================================
# 🤖 AI & MACHINE LEARNING IMPORTS
# =========================================
try:
    from google import genai
except ImportError:
    genai = None
try:
    import tensorflow as tf
except ImportError:
    tf = None

# LangChain & Vector DB
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# =========================================
# 📄 DOCUMENT & UI EXTRAS
# =========================================
import markdown2
from xhtml2pdf import pisa
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

# =========================================
# ⚙️ APP CONSTANTS
# =========================================
DB_FAISS_PATH = "vectorstore/db_faiss"


def get_gemini_client():
    if genai is None:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    return genai.Client(api_key=api_key)


def notify_startup_warning(message):
    st.session_state.setdefault("startup_warnings", [])
    if message not in st.session_state.startup_warnings:
        st.session_state.startup_warnings.append(message)

#SQL db
import sqlite3
import warnings

# 1. Hide those scikit-learn version warnings from the terminal
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Database Helper Function
def db_action(query, params=(), fetch=False):
    try:
        conn = sqlite3.connect('diagnovix_persistence.db', check_same_thread=False)
        c = conn.cursor()
        c.execute(query, params)
        data = c.fetchall() if fetch else None
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(f"DB Error: {e}")
        return None

# 3. Initialize Tables (Run once)
db_action('''CREATE TABLE IF NOT EXISTS stats 
             (id INTEGER PRIMARY KEY, diagnoses INTEGER, images INTEGER, chats INTEGER)''')
if not db_action("SELECT * FROM stats WHERE id=1", fetch=True):
    db_action("INSERT INTO stats VALUES (1, 0, 0, 0)")
    
# --- Admin Access Credentials ---
ADMIN_USER = "Nikhil"
ADMIN_PASS = "Nikhil@2004"

# =========================================
# 📊 GLOBAL SESSION STATE INIT
# =========================================

# 1. Pull current counts from Database so they PERSIST on restart
saved_stats = db_action("SELECT diagnoses, images, chats FROM stats WHERE id=1", fetch=True)
db_diag, db_img, db_chat = saved_stats[0] if saved_stats else (0, 0, 0)

# 2. Assign those DB values to your session state
if "total_diagnoses" not in st.session_state:
    st.session_state.total_diagnoses = db_diag

if "images_analyzed" not in st.session_state:
    st.session_state.images_analyzed = db_img

if "chat_queries" not in st.session_state:
    st.session_state.chat_queries = db_chat

if "active_models" not in st.session_state:
    st.session_state.active_models = 11

# 3. Use the DB values for the chart history too
if "analytics_history" not in st.session_state:
    st.session_state.analytics_history = [{
        "Diagnoses": db_diag, 
        "Images": db_img, 
        "Chat": db_chat
    }]

if "activity_log" not in st.session_state:
    st.session_state.activity_log = ["System Initialized - Database Connected"]
    


# =========================================
# ⚙️ 1. GLOBAL CONSTANTS
# =========================================
# Defining these at the very top ensures they are available for the entire script.
NAV_HEIGHT_PX = 60
APP_NAME = " 🩺 DiagnoVix"
APP_VERSION = "1.0.0"

PAGE_HOME = "PAGE_HOME"
PAGE_DASHBOARD = "Dashboard"
PAGE_CHATBOT = "PAGE_CHATBOT"
PAGE_DIAGNOSE = "PAGE_DIAGNOSE"

DIAGNOSE_MODE_FORM = "Form Based"
DIAGNOSE_MODE_IMAGE = "Image Based"

IMAGE_AI_GEMINI = "Gemini"
IMAGE_AI_BRAIN_STROKE = "Brain Stroke"

CHATBOT_GEMINI = "ChatBot (Gemini)"
CHATBOT_MISTRAL = "Mistral"

# =========================================
# 📌 2. ROUTING & PAGE CONFIG
# =========================================
query_params = st.query_params
current_page = query_params.get("page", PAGE_HOME)

if isinstance(current_page, list):
    current_page = current_page[0]

# =========================================
# 🌙 3. THEME PERSISTENCE ENGINE (CLEANED)
# =========================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark" 

# Sync session state with URL parameters
url_theme = st.query_params.get("theme")
if url_theme in ["light", "dark"]:
    st.session_state.theme = url_theme

# Handle the Toggle Signal
if st.query_params.get("toggle_theme") == "true":
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    # Update URL to the new theme and stay on current page
    st.query_params.clear()
    st.query_params["page"] = current_page 
    st.query_params["theme"] = st.session_state.theme
    st.rerun()


# =========================================
# 🧠 MODEL LOADING
# =========================================
@st.cache_resource
def load_diabetes_model_and_scaler():
    model = joblib.load('./models/diabetes_model.pkl')
    scaler = joblib.load('./models/diabetes_scaler.pkl')
    return model, scaler

@st.cache_resource
def load_kidney_model():
    return joblib.load('./models/kidneyDisease_model.pkl')

@st.cache_resource
def load_heart_model_and_scaler():
    model = joblib.load('models/heartDisease_randomForest_model.pkl')
    scaler = joblib.load('models/heartDisease_scaler.pkl')
    return model, scaler

@st.cache_resource
def load_hypertension_model():
    return joblib.load('models/hypertension_model.pkl')

@st.cache_resource
def load_breast_cancer_model():
    return joblib.load('./models/breastCancer_randomForest_model.pkl')

@st.cache_resource
def load_lung_cancer_model():
    return joblib.load('./models/lungCancer_XGBClassifier_model.pkl')

@st.cache_resource
def load_liver_disease_model():
    return joblib.load('./models/liverDisease_rf_model.pkl')

@st.cache_resource
def load_thyroid_model():
    return joblib.load('./models/thyroid_cat_model.pkl')

@st.cache_resource
def load_brainstroke_tf_model():
    try:
        if tf is None:
            raise RuntimeError("TensorFlow is not installed")

        model = tf.keras.models.load_model(
            "models/brainstroke_model.h5",
            compile=False,
            custom_objects={"InputLayer": tf.keras.layers.InputLayer}
        )

        print("✅ Brain Stroke Model Loaded Successfully")
        return model

    except Exception as e:
        print("❌ Model Load Error:", e)
        return None

# Load models
def load_cached_resource(loader, label, default=None):
    try:
        return loader()
    except Exception as e:
        notify_startup_warning(f"{label} could not be loaded: {e}")
        return default


diabetes_model, diabetes_scaler = load_cached_resource(
    load_diabetes_model_and_scaler, "Diabetes model", (None, None)
)
kidney_model = load_cached_resource(load_kidney_model, "Kidney disease model")
heart_model, heart_scaler = load_cached_resource(
    load_heart_model_and_scaler, "Heart disease model", (None, None)
)
hypertension_model = load_cached_resource(load_hypertension_model, "Hypertension model")
breast_model = load_cached_resource(load_breast_cancer_model, "Breast cancer model")
lung_model = load_cached_resource(load_lung_cancer_model, "Lung cancer model")
liver_model = load_cached_resource(load_liver_disease_model, "Liver disease model")
thyroid_model = load_cached_resource(load_thyroid_model, "Thyroid disease model")

brainStrokemodel = load_cached_resource(load_brainstroke_tf_model, "brainstroke model")
# brainStrokemodel = None  # Force demo mode

# --- Feature Lists (as provided) ---
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['Age', 'BMI', 'Systolic_BP','Diastolic_BP',  'Total_Cholesterol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                  'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_phosphate', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin',
 'Albumin_Globulin_ratio']
THYROID_FEATURES = ['Age','Sex','On Thyroxine','Query on Thyroxine','On Antithyroid Meds','Is Sick','Is Pregnant','Had Thyroid Surgery',
                    'Had I131 Treatment','Query Hypothyroid','Query Hyperthyroid','On Lithium','Has Goitre','Has Tumor','Psych Condition',
                    'TSH Level','T3 Level','TT4 Level','T4U Level','FTI Level', 'TBG Level']


# --- Feature Input Definitions (as provided, with slight adjustments for consistency if needed) ---
FEATURE_INPUTS = {
    # DIABETES
    'Pregnancies': ('slider', 0, 20, 1), 'Glucose': ('slider', 40, 300, 1),
    'BloodPressure': ('slider', 30, 180, 1), 'SkinThickness': ('slider', 0, 100, 1),
    'Insulin': ('slider', 0, 900, 1), 'BMI': ('slider', 10.0, 70.0, 0.1),
    'DiabetesPedigreeFunction': ('slider', 0.01, 2.5, 0.01), 'Age': ('slider', 1, 120, 1),
    # KIDNEY 
    'age': ('slider', 1, 100, 1), 'bp': ('slider', 40, 180, 1), 'al': ('slider', 0, 5, 1),
    'su': ('slider', 0, 5, 1), 'rbc': ('select', ['normal', 'abnormal']),
    'pc': ('select', ['normal', 'abnormal']), 'pcc': ('select', ['present', 'notpresent']),
    'ba': ('select', ['present', 'notpresent']), 'bgr': ('slider', 70, 500, 1),
    'bu': ('slider', 5, 200, 1), 'sc': ('slider', 0.1, 20.0, 0.1),
    'pot': ('slider', 2.5, 10.0, 0.1), 'wc': ('slider', 1000, 25000, 100),
    'htn': ('select', ['yes', 'no']), 'dm': ('select', ['yes', 'no']),
    'cad': ('select', ['yes', 'no']), 'pe': ('select', ['yes', 'no']),
    'ane': ('select', ['yes', 'no']),
    # HEART
    'sex': ('select', ['male', 'female']), 'cp': ('slider', 0, 3, 1),
    'trestbps': ('slider', 80, 200, 1), 'chol': ('slider', 100, 600, 1),
    'fbs': ('select', ['yes', 'no']), 'restecg': ('slider', 0, 2, 1),
    'thalach': ('slider', 60, 220, 1), 'exang': ('select', ['yes', 'no']),
    'oldpeak': ('slider', 0.0, 6.0, 0.1), 'slope': ('slider', 0, 2, 1),
    'ca': ('slider', 0, 4, 1), 'thal': ('slider', 0, 3, 1),
    # HYPERTENSION 
    'Systolic_BP': ('slider', 80, 200, 1), 'Diastolic_BP': ('slider', 40, 120, 1),
    'Total_Cholesterol': ('slider', 100, 400, 1),
    'smoking': ('select', ['yes', 'no']), 'exercise': ('select', ['yes', 'no']),
    'alcohol': ('select', ['yes', 'no']),
    # BREAST CANCER
    'mean_radius': ('slider', 5.0, 30.0, 0.1), 'mean_texture': ('slider', 5.0, 40.0, 0.1),
    'mean_perimeter': ('slider', 30.0, 200.0, 0.1), 'mean_area': ('slider', 100.0, 2500.0, 1.0),
    'mean_smoothness': ('slider', 0.05, 0.2, 0.001), 'compactness_mean': ('slider', 0.01, 1.0, 0.01),
    'concavity_mean': ('slider', 0.01, 1.0, 0.01), 'concave points_mean': ('slider', 0.01, 0.5, 0.01),
    'symmetry_mean': ('slider', 0.1, 0.5, 0.01), 'fractal_dimension_mean': ('slider', 0.01, 0.2, 0.001),
    # LUNG CANCER
    'GENDER': ('select', ['Male', 'Female']), 'AGE': ('slider', 10, 100, 1),
    'SMOKING': ('select', ['Yes', 'No']), 'YELLOW_FINGERS': ('select', ['Yes', 'No']),
    'ANXIETY': ('select', ['Yes', 'No']), 'PEER_PRESSURE': ('select', ['Yes', 'No']),
    'CHRONIC_DISEASE': ('select', ['Yes', 'No']), 'FATIGUE': ('select', ['Yes', 'No']), 
    'ALLERGY': ('select', ['Yes', 'No']), 'WHEEZING': ('select', ['Yes', 'No']),
    'ALCOHOL_CONSUMING': ('select', ['Yes', 'No']), 'COUGHING': ('select', ['Yes', 'No']), 
    'SHORTNESS_OF_BREATH': ('select', ['Yes', 'No']), 'SWALLOWING_DIFFICULTY': ('select', ['Yes', 'No']),
    'CHEST_PAIN': ('select', ['Yes', 'No']),
    # LIVER
    'Gender': ('select', ['Male', 'Female']), 'Total_Bilirubin': ('slider', 0.1, 10.0, 0.1),
    'Alkaline_phosphate': ('slider', 50, 3000, 1), 'Alamine_Aminotransferace': ('slider', 1, 2000, 1),
    'Aspartate_Amino': ('slider', 1, 2000, 1), 'Protien': ('slider', 2.0, 10.0, 0.1), # 'Protien' vs 'Protein'
    'Albumin': ('slider', 1.0, 6.0, 0.1), 'Albumin_Globulin_ratio': ('slider', 0.1, 3.0, 0.1),
    # THYROID
    'Sex': ('select', ['Female', 'Male']), 'On Thyroxine': ('select', ['No', 'Yes']),
    'Query on Thyroxine': ('select', ['No', 'Yes']), 'On Antithyroid Meds': ('select', ['No', 'Yes']),
    'Is Sick': ('select', ['No', 'Yes']), 'Is Pregnant': ('select', ['No', 'Yes']),
    'Had Thyroid Surgery': ('select', ['No', 'Yes']), 'Had I131 Treatment': ('select', ['No', 'Yes']),
    'Query Hypothyroid': ('select', ['No', 'Yes']), 'Query Hyperthyroid': ('select', ['No', 'Yes']),
    'On Lithium': ('select', ['No', 'Yes']), 'Has Goitre': ('select', ['No', 'Yes']),
    'Has Tumor': ('select', ['No', 'Yes']), 'Psych Condition': ('select', ['No', 'Yes']),
    'TSH Level': ('slider', 0.01, 20.0, 0.1), 'T3 Level': ('slider', 0.2, 5.0, 0.1),
    'TT4 Level': ('slider', 50, 300, 1), 'T4U Level': ('slider', 0.3, 1.5, 0.01),
    'FTI Level': ('slider', 3, 50, 0.1), 'TBG Level': ('slider', 10, 50, 1),
}

# --- Symptom Data for Multi-Disease Diagnosis (as provided) ---
diseases_data = {
    'Common Cold': ['runny nose', 'sore throat', 'cough', 'sneezing', 'mild headache', 'fatigue', 'nasal congestion'],
    'Influenza (Flu)': ['fever', 'body aches', 'chills', 'fatigue', 'cough', 'sore throat', 'headache', 'muscle pain', 'nausea', 'vomiting'],
    'Allergies': ['sneezing', 'itchy eyes', 'runny nose', 'nasal congestion', 'itchy throat', 'watery eyes'],
    'Strep Throat': ['sore throat', 'fever', 'swollen lymph nodes', 'difficulty swallowing', 'red spots on roof of mouth', 'headache', 'nausea', 'vomiting'],
    'Bronchitis': ['cough', 'chest discomfort', 'shortness of breath', 'fatigue', 'sore throat', 'mild fever'],
    'Pneumonia': ['cough', 'fever', 'chills', 'shortness of breath', 'chest pain', 'fatigue', 'nausea', 'vomiting'],
    'Migraine': ['severe headache', 'pulsating pain', 'sensitivity to light', 'sensitivity to sound', 'nausea', 'vomiting', 'aura'],
    'Tension Headache': ['mild headache', 'pressure around head', 'neck pain', 'shoulder pain'],
    'Gastroenteritis (Stomach Flu)': ['nausea', 'vomiting', 'diarrhea', 'abdominal cramps', 'mild fever', 'fatigue'],
    'Urinary Tract Infection (UTI)': ['frequent urination', 'painful urination', 'burning sensation during urination', 'cloudy urine', 'pelvic pain', 'fever', 'chills']
}

# --- Feature Full Forms (as provided) ---
feature_fullforms = {
    # 'Choose Disease': {'Select the disease you want to diagnose': 'Please choose one disease from the list'},
    'Symptom Checker': {'1': "Instant health insights based on your symptoms.",'2':"Check what might be causing your symptoms in seconds.",
                        '3':"Smart diagnosis support from the symptoms you enter.",'4':"Input your symptoms. Get possible conditions instantly."},
    'Diabetes': {'Pregnancies': 'Number of times pregnant', 'Glucose': 'Plasma glucose concentration', 'BloodPressure': 'Diastolic blood pressure (mm Hg)',
                  'SkinThickness': 'Triceps skin fold thickness (mm)', 'Insulin': '2-Hour serum insulin (mu U/ml)', 'BMI': 'Body Mass Index', 
                  'DiabetesPedigreeFunction': 'Diabetes pedigree function', 'Age': 'Age in years'},
    'Kidney Disease': {'age': 'Age', 'bp': 'Blood Pressure', 'al': 'Albumin', 'su': 'Sugar', 'rbc': 'Red Blood Cells', 'pc': 'Pus Cell', 'pcc': 'Pus Cell Clumps',
                        'ba': 'Bacteria', 'bgr': 'Blood Glucose Random', 'bu': 'Blood Urea', 'sc': 'Serum Creatinine', 'pot': 'Potassium', 'wc': 'White Blood Cell Count',
                          'htn': 'Hypertension', 'dm': 'Diabetes Mellitus', 'cad': 'Coronary Artery Disease', 'pe': 'Pedal Edema', 'ane': 'Anemia'},
    'Heart Disease': {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholesterol', 
                      'fbs': 'Fasting Blood Sugar > 120 mg/dl', 'restecg': 'Resting ECG Results', 'thalach': 'Maximum Heart Rate Achieved', 
                      'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression', 'slope': 'Slope of ST Segment', 'ca': 'Number of Major Vessels Colored',
                      'thal': 'Thalassemia'},
    'Hypertension': {'Age':'Age', 'BMI': 'Body Mass Index', 'Systolic_BP': 'Systolic Blood Pressure', 'Diastolic_BP': 'Diastolic Blood Pressure', 
                     'Total_Cholesterol':'Total Cholesterol', 'smoking': 'Smoking Habit', 'exercise': 'Physical Activity', 'alcohol': 'Alcohol Consumption'},
    'Breast Cancer': {'mean_radius': 'Mean Radius', 'mean_texture': 'Mean Texture', 'mean_perimeter': 'Mean Perimeter', 'mean_area': 'Mean Area', 
                      'mean_smoothness': 'Mean Smoothness', 'compactness_mean': 'Mean Compactness', 'concavity_mean': 'Mean Concavity', 
                      'concave points_mean': 'Mean Concave Points', 'symmetry_mean': 'Mean Symmetry', 'fractal_dimension_mean': 'Mean Fractal Dimension'},
    'Lung Cancer': {'GENDER': 'Gender', 'AGE': 'Age', 'SMOKING': 'Smoking', 'YELLOW_FINGERS': 'Yellow Fingers', 'ANXIETY': 'Anxiety', 
                    'PEER_PRESSURE': 'Peer Pressure', 'CHRONIC_DISEASE': 'Chronic Disease', 'FATIGUE': 'Fatigue', 'ALLERGY': 'Allergy', 'WHEEZING': 'Wheezing', 
                    'ALCOHOL_CONSUMING': 'Alcohol Consumption', 'COUGHING': 'Coughing', 'SHORTNESS_OF_BREATH': 'Shortness of Breath', 
                    'SWALLOWING_DIFFICULTY': 'Swallowing Difficulty', 'CHEST_PAIN': 'Chest Pain'},
    'Liver Disease': {'Age': 'Age', 'Gender': 'Gender', 'Total_Bilirubin': 'Total Bilirubin', 'Alkaline_phosphate': 'Alkaline Phosphotase', 
                      'Alamine_Aminotransferace': 'Alamine Aminotransferase', 'Aspartate_Amino': 'Aspartate Aminotransferase', 'Protien': 'Total Protein', 
                      'Albumin': 'Albumin Level', 'Albumin_Globulin_ratio': 'Albumin to Globulin Ratio'},
    'Thyroid Disease': {'Age': 'Age', 'Sex': 'Sex', 'On Thyroxine': 'On Thyroxine Medication', 'Query on Thyroxine': 'Query on Thyroxine', 
                        'On Antithyroid Meds': 'On Antithyroid Medication', 'Is Sick': 'Currently Sick', 'Is Pregnant': 'Currently Pregnant', 
                        'Had Thyroid Surgery': 'History of Thyroid Surgery', 'Had I131 Treatment': 'History of I131 Treatment', 'Query Hypothyroid': 'Query Hypothyroid',
                          'Query Hyperthyroid': 'Query Hyperthyroid', 'On Lithium': 'On Lithium Medication', 'Has Goitre': 'Presence of Goitre', 
                          'Has Tumor': 'Presence of Tumor', 'Psych Condition': 'Psychological Condition Reported', 'TSH Level': 'TSH Level', 'T3 Level': 'T3 Level',
                        'TT4 Level': 'Total T4 Level', 'T4U Level': 'T4U Level', 'FTI Level': 'Free Thyroxine Index (FTI) Level', 'TBG Level': 'TBG Level'}
}


# --- MODEL_MAPPING (Ensure HYPERTENSION_FEATURES is updated if necessary) ---
MODEL_MAPPING = {
    "Symptom Checker": {"name": "Symptom Checker", "features": [], "model": None, "scaler": None},
    "Diabetes": {"name": "Diabetes", "features": DIABETES_FEATURES, "model": diabetes_model, "scaler": diabetes_scaler},
    "Kidney Disease": {"name": "Kidney Disease", "features": KIDNEY_FEATURES, "model": kidney_model, "scaler": None},
    "Heart Disease": {"name": "Heart Disease", "features": HEART_FEATURES, "model": heart_model, "scaler": heart_scaler},
    "Hypertension": {"name": "Hypertension", "features": HYPERTENSION_FEATURES, "model": hypertension_model, "scaler": None},
    "Breast Cancer": {"name": "Breast Cancer", "features": BREAST_FEATURES, "model": breast_model, "scaler": None},
    "Lung Cancer": {"name": "Lung Cancer", "features": LUNG_FEATURES, "model": lung_model, "scaler": None},
    "Liver Disease": {"name": "Liver Disease", "features": LIVER_FEATURES, "model": liver_model, "scaler": None},
    "Thyroid Disease": {"name": "Thyroid Disease", "features": THYROID_FEATURES, "model": thyroid_model, "scaler": None},
}
# ==============================================================================
# 🎨 4. UI/UX ARCHITECTURE: DYNAMIC THEME ENGINE & CUSTOM CSS
# ==============================================================================

# 1. DEFINE THE PALETTES
if st.session_state.theme == "light":
    bg_color, card_bg, text_color = "#f8fafc", "#ffffff", "#1e293b"
    border_color, nav_bg = "#e2e8f0", "rgba(255, 255, 255, 0.90)"
    secondary_text, hover_bg = "#64748b", "rgba(0, 0, 0, 0.05)"
else:
    bg_color, card_bg, text_color = "#0f1116", "#1a1d24", "#f8fafc"
    border_color, nav_bg = "#2d3748", "rgba(15, 17, 22, 0.85)"
    secondary_text, hover_bg = "#8b949e", "rgba(255, 255, 255, 0.05)"

# 2. INJECT DYNAMIC CSS (Fixed with double curly braces {{ }})
st.markdown(f"""
<style>
/* =========================================
   🌍 1. BASE / GLOBAL STYLES
========================================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

.stApp {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    font-family: 'Inter', sans-serif;
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

.block-container {{
    padding-top: {NAV_HEIGHT_PX + 40}px;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}}

/* =========================================
   🧭 2. TOP NAVIGATION BAR
========================================= */
.topnav {{
    position: fixed;
    top: 0; left: 0; width: 100%;
    height: {NAV_HEIGHT_PX}px;
    background: {nav_bg};
    backdrop-filter: blur(12px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 4rem;
    border-bottom: 1px solid {border_color};
    z-index: 9999;
}}

.topnav h1 {{
    color: {text_color} !important;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
}}

.nav-links {{
    display: flex;
    gap: 15px;
}}

.topnav a {{
    color: {secondary_text} !important;
    padding: 8px 16px;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    border-radius: 8px;
    transition: all 0.3s ease;
}}

.topnav a:hover {{
    color: {text_color} !important;
    background: {hover_bg};
}}

.nav-active {{
    background-color: #2563eb !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}}

/* =========================================
   📌 3. SIDEBAR
========================================= */
[data-testid="stSidebar"] {{
    background: {card_bg} !important;
    border-right: 1px solid {border_color};
}}

/* =========================================
   💬 4. CHAT UI
========================================= */
.chat-box {{
    height: 500px;
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}}

.message.user {{
    background: #2563eb !important;
    color: white !important;
    border-radius: 12px 12px 0 12px;
}}

.message.bot {{
    background: {border_color} !important;
    color: {text_color} !important;
    border-radius: 12px 12px 12px 0;
}}

/* =========================================
   🧩 5. CARDS & COMPONENTS
========================================= */
.card, .disease-card-home, [data-testid="stMetricContainer"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    padding: 2rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border-color 0.3s ease;
    color: {text_color} !important;
}}

.disease-card-home:hover {{
    transform: translateY(-8px);
    border-color: #2563eb !important;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
}}

/* =========================================
   🧾 6. FORMS, INPUTS & BUTTONS
========================================= */
.stButton>button {{
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
}}

div[data-baseweb="select"], div[data-baseweb="input"] input, .stTextArea textarea {{
    background-color: {card_bg} !important;
    border-radius: 8px !important;
    border: 1px solid {border_color} !important;
    color: {text_color} !important;
}}

/* Force all text elements to respect theme */
h1, h2, h3, h4, h5, h6, p, span, label, li {{
    color: {text_color} !important;
}}



/* =========================================
   📱 8. RESPONSIVE DESIGN
========================================= */
@media screen and (max-width: 768px) {{
    .topnav {{ padding: 0 1.5rem; }}
    .nav-links {{ gap: 5px; }}
    .topnav a {{ padding: 6px 10px; font-size: 0.8rem; }}
}}
</style>
""", unsafe_allow_html=True)





# =========================================
# 🛠️ HELPERS + GEMINI (FIXED)
# =========================================

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()


# =========================================
# 💬 CHAT HISTORY
# =========================================
def load_chat_history_from_file():
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [("bot", "Welcome! (Previous history corrupted)")]
    return [("bot", "Welcome! How can I help you today?")]


def save_chat_history_to_file(chat_history_data):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history_data, f)


# =========================================
# 🤖 GEMINI CHAT (FIXED)
# =========================================
def get_gemini_chat_response(message_text, history=None):
    try:
        if history is None:
            history = []

        prompt = (
            "You are a medical AI assistant. Answer only health-related questions. "
            "If unrelated, politely refuse.\n\n"
        )

        for role, msg in history:
            if role == "user":
                prompt += f"User: {msg}\n"
            else:
                prompt += f"Assistant: {msg}\n"

        prompt += f"User: {message_text}\nAssistant:"

        # ✅ FIXED MODEL NAME
        gemini_client = get_gemini_client()
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# =========================================
# 🖼️ GEMINI IMAGE ANALYSIS (UNCHANGED)
# =========================================
def get_gemini_image_diagnosis_response(prompt, image):
    try:
        gemini_client = get_gemini_client()
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",   # use correct model
            contents=[prompt, image.convert("RGB")]
        )

        return response.text

    except Exception as e:
        print("Gemini Error:", e)

        # ✅ FALLBACK RESPONSE (IMPORTANT FOR DEMO)
        return """
        ⚠️ AI analysis is temporarily unavailable.

        Basic Guidance:
        - Please consult a radiologist for accurate diagnosis.
        - Check for visible abnormalities manually.
        - Ensure proper scan quality.

        (Demo mode response)
        """

# =========================================
# 📄 PDF + IMAGE PREPROCESSING
# =========================================
def convert_html_to_pdf_bytes(source_html_content):
    pdf_io = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(source_html_content), dest=pdf_io)

    if pisa_status.err:
        st.error(f"PDF Conversion Error: {pisa_status.err}")
        return None

    pdf_bytes_content = pdf_io.getvalue()
    pdf_io.close()
    return pdf_bytes_content


def preprocess_brain_stroke_image(image_obj: Image.Image):
    image_obj = image_obj.resize((224, 224)).convert('RGB')
    image_array_data = np.array(image_obj) / 255.0
    return np.expand_dims(image_array_data, axis=0)
# =========================================
# 📊 PROFESSIONAL GLOSSY DASHBOARD
# =========================================
def render_dashboard():
    # --- Dynamic Page Title ---
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>📊 System Analytics Dashboard</h1>", unsafe_allow_html=True)

    # =========================================
    # ⚙️ ENSURE SESSION VARIABLES EXIST
    # =========================================
    # We maintain your exact logic for initializing variables
    for key, default in {
        "total_diagnoses": 0, "images_analyzed": 0, 
        "chat_queries": 0, "active_models": 11,
        "analytics_history": [], "activity_log": []
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # =========================================
    # 📌 KPI / GLOSSY METRICS SECTION
    # =========================================
    col1, col2, col3, col4 = st.columns(4)

    # Metric Helper to maintain the Glossy Look
    def glossy_metric(column, label, value, color_hex):
        column.markdown(f"""
        <div class="disease-card-home" style="border-top: 4px solid {color_hex}; padding: 1.5rem 1rem; background: #1a1d24; border-radius: 12px; text-align: center;">
            <p style="color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">{label}</p>
            <h2 style="color: #ffffff; margin: 0; font-size: 2rem;">{value}</h2>
        </div>
        """, unsafe_allow_html=True)

    glossy_metric(col1, "Total Diagnoses", st.session_state.total_diagnoses, "#3fb950") 
    glossy_metric(col2, "Images Analyzed", st.session_state.images_analyzed, "#58a6ff") 
    glossy_metric(col3, "Chat Queries", st.session_state.chat_queries, "#bc8cf2")     
    glossy_metric(col4, "Active Models", st.session_state.active_models, "#f85149")   

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================
    # 📈 ANALYTICS & RECENT LOGS (SIDE BY SIDE)
    # =========================================
    chart_col, log_col = st.columns([2, 1])

    with chart_col:
        st.markdown("### 📈 Interaction Frequency Index")

        # --- 1. DATA SNAPSHOT LOGIC ---
        total_load = (st.session_state.total_diagnoses + 
                      st.session_state.images_analyzed + 
                      st.session_state.chat_queries)
        
        new_point = {
            "Diagnoses": st.session_state.total_diagnoses,
            "Images": st.session_state.images_analyzed,
            "Chat": st.session_state.chat_queries,
            "Total Load": total_load
        }
        
        # Original Logic: Prevent redundant data points
        if not st.session_state.analytics_history or st.session_state.analytics_history[-1] != new_point:
            st.session_state.analytics_history.append(new_point)
        
        # Prepare DataFrame for Plotly
        df_plot = pd.DataFrame(st.session_state.analytics_history).tail(12)

        # --- 2. PLOTLY GRAPH LOGIC (Permanent Dots + Lines) ---
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Mapping metrics to colors as per your logic
            metrics_map = [
                ("Diagnoses", "#22c55e"), 
                ("Images", "#3b82f6"),    
                ("Chat", "#a855f7"),      
                ("Total Load", "#ffffff") 
            ]

            for column, color in metrics_map:
                if column in df_plot.columns:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(df_plot))), 
                        y=df_plot[column],
                        mode='lines+markers',        # 🔥 Visible dots + lines
                        name=column,
                        line=dict(color=color, width=3),
                        marker=dict(size=10, symbol='circle', line=dict(width=2, color='#1a1d24'))
                    ))

            # Professional Styling
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=30, b=10),
                height=350,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Interaction Steps"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Count"),
                hovermode="x unified",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.line_chart(df_plot)
            st.error(f"Visualization Error: {e}")
        
        st.markdown(f"**Current System Status:** `{total_load}` Total Transactions Recorded")

    with log_col:
        st.markdown("### 🧾 Activity Feed")
        recent_logs = st.session_state.activity_log[-6:][::-1]
        
        if recent_logs:
            log_html = "<div style='background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>"
            for log in recent_logs:
                log_html += f"<p style='color: #8b949e; font-size: 0.85rem; border-bottom: 1px solid rgba(255,255,255,0.03); padding-bottom: 8px; margin-top: 8px;'>🕒 {log}</p>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.info("System standby. No recent activity detected.")

    st.markdown("---")
# =========================================
    # ⚙️ SYSTEM HEALTH STATUS (OPTIMIZED UI)
    # =========================================
    st.markdown("### ⚙️ System Integrity & Health")
    
    # Injection of specific styles for the health dashboard
    st.markdown("""
    <style>
    .health-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 20px;
        border-radius: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .health-container:hover {
        transform: translateY(-2px);
    }
    .status-indicator {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px currentColor; /* Creates the glowing LED effect */
    }
    .health-label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #f8fafc;
    }
    .health-status-text {
        font-weight: 700;
        margin-left: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.markdown("""
        <div class="health-container" style="background: rgba(63, 185, 80, 0.05); border: 1px solid rgba(63, 185, 80, 0.3);">
            <span class="status-indicator" style="background: #3fb950; color: #3fb950;"></span>
            <span class="health-label">API Gateway: <span class="health-status-text" style="color: #3fb950;">Connected</span></span>
        </div>
        """, unsafe_allow_html=True)

    with status_col2:
        st.markdown("""
        <div class="health-container" style="background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.3);">
            <span class="status-indicator" style="background: #58a6ff; color: #58a6ff;"></span>
            <span class="health-label">Neural Engine: <span class="health-status-text" style="color: #58a6ff;">Operational</span></span>
        </div>
        """, unsafe_allow_html=True)

    with status_col3:
        st.markdown("""
        <div class="health-container" style="background: rgba(248, 81, 73, 0.05); border: 1px solid rgba(248, 81, 73, 0.3);">
            <span class="status-indicator" style="background: #f85149; color: #f85149;"></span>
            <span class="health-label">Rate Limit: <span class="health-status-text" style="color: #f85149;">Free Tier Active</span></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    footer()
    
def footer():
    st.markdown("""
    <style>
    .footer {
        text-align: center;
        padding: 15px;
        margin-top: 30px;   /* reduced from 50px */
        margin-bottom: 0px;
        font-size: 14px;
        color: #cccccc;
        background-color: #1a5a96;
        border-radius: 8px;
    }

    .footer a {
        color: #cccccc;
        text-decoration: none;
        margin: 0 8px;
    }

    .footer a:hover {
        color: #ffffff;
    }
    </style>

    <div class="footer">
        🤖 <strong style="color:#ffffff;">DiagnoVix</strong><br>
        Revolutionizing Healthcare with AI 🧠<br>
        © 2026 DiagnoVix<br><br>
        🔗 <a href="https://github.com/NIKHIL-MISHRA-COAT">GitHub</a> 
        
    </div>
    """, unsafe_allow_html=True)
    # | 💼 <a href="https://www.linkedin.com/in/roshan-yadavv10">LinkedIn</a>
# =========================================
# 🏠 HOME PAGE RENDER FUNCTION
# =========================================
def render_home_page():

    # =========================================
    # 🧾 APP TITLE + INTRO
    # =========================================
    st.title(APP_NAME)
    st.subheader("Advanced Multimodal AI Medical Diagnosis System")

    st.markdown(f"""
    <div class="disease-card-home" style="text-align: left; margin-bottom: 25px; border-left: 5px solid #2563eb;">
        <p style="font-size: 1.1rem; line-height: 1.6; margin: 0;">
            <b>DiagnoVix (v{APP_VERSION})</b> is an Intelligent Decision Support System (IDSS) designed for early-stage 
            medical screening. By synthesizing clinical parameters, medical imagery, and natural language symptoms, 
            the platform provides data-driven insights to facilitate preventative healthcare.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Guidance
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        st.info("🧪 **Diagnose:** Access ML models for disease prediction.")
    with nav_col2:
        st.success("🤖 **Chatbot:** Consult our AI for medical terminology guidance.")

    st.markdown("---")

# =========================================
    # 🧬 DISEASE CARDS WITH INTERACTIVE INFO
    # =========================================
    st.markdown("<h3 class='section-header'>🧬 Clinical Assessments Covered</h3>", unsafe_allow_html=True)

    # 📖 Professional Disease Information Database
    DISEASE_INFO_DB = {
        "Symptom Checker": "A general diagnostic aid that cross-references user-reported symptoms against common viral and bacterial conditions using a weighted matching algorithm.",
        "Diabetes": "Analyzes clinical parameters such as Glucose levels, Insulin, and BMI to predict the risk of Type-2 Diabetes based on metabolic patterns.",
        "Kidney Disease": "Evaluates chronic kidney disease (CKD) risks by processing physiological metrics like serum creatinine, albumin levels, and blood pressure.",
        "Heart Disease": "Calculates cardiovascular risk factors by evaluating metrics including cholesterol, maximum heart rate achieved, and resting ECG results.",
        "Hypertension": "Predicts hypertensive risk by analyzing systolic and diastolic blood pressure trends in correlation with patient age and Body Mass Index (BMI).",
        "Breast Cancer": "Utilizes a classification model to distinguish between malignant and benign tumor characteristics based on mean radius, texture, and concavity.",
        "Lung Cancer": "Assesses respiratory health risks by correlating lifestyle factors (e.g., smoking history) with clinical symptoms like wheezing and chest pain.",
        "Liver Disease": "Predicts hepatic health conditions by evaluating vital enzyme levels (SGOT, SGPT) and bilirubin concentrations from clinical data.",
        "Thyroid Disease": "Identifies endocrine imbalances by processing TSH, T3, and T4 hormone levels to determine the risk of hypo- or hyperthyroidism."
    }

    disease_names_list = list(MODEL_MAPPING.keys())
    emojis = ["🩺", "🩸", "⚕️", "❤️", "💢", "🎀", "🫁", "🏥", "🦋"]

    # Use 3 columns for a balanced glossy grid
    num_cols = 3
    cols = st.columns(num_cols)

    for i, disease_key in enumerate(disease_names_list):
        disease_name = MODEL_MAPPING[disease_key]["name"]
        # Fetch professional description
        info_text = DISEASE_INFO_DB.get(disease_name, "Diagnostic module for assessing health indicators.")
        
        with cols[i % num_cols]:
            # The static Glossy Card UI
            st.markdown(f"""
            <div class="disease-card-home" style="padding-bottom: 5px; margin-bottom: 0px;">
                <div class="emoji">{emojis[i % len(emojis)]}</div>
                <h5 style="margin-bottom: 10px;">{disease_name}</h5>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Interactive "Quick Info" Popover ---
            # use_container_width=True aligns the button to the card width
            with st.popover(f"ℹ️ About {disease_name}", use_container_width=True):
                st.markdown(f"#### {emojis[i % len(emojis)]} {disease_name} Analysis")
                st.write(info_text)
                
                # Professional CTA (Call to Action)
                if st.button(f"Launch {disease_name} Diagnosis", key=f"nav_{i}"):
                    # Update page and rerun (standard Streamlit routing logic)
                    st.query_params["page"] = PAGE_DIAGNOSE
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    # =========================================
    # 📖 DETAILED PROJECT OVERVIEW
    # =========================================
    st.markdown("<h3 class='section-header'>📖 About the Project</h3>", unsafe_allow_html=True)

    # --- Mission & Architecture ---
    st.markdown("""
    <div class="disease-card-home" style="text-align: left; margin-bottom: 20px;">
        <h4 style="color: #58a6ff; margin-top:0;">🌟 Project Vision</h4>
        <p style="color: #8b949e; font-size: 0.95rem;">
            DiagnoVix was conceptualized to address the need for <b>Proactive Health Monitoring</b>. 
            The system employs a <b>Multimodal AI Framework</b>—capable of processing structured clinical data, 
            unstructured symptom text, and visual medical scans within a unified environment. 
            This project demonstrates the practical application of <b>Deep Learning (CNNs)</b> and 
            <b>Supervised Machine Learning</b> in modern medical informatics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Tech Stack & Layers ---
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; border-left: 3px solid #238636; height:100%;">
            <h6 style="color: #3fb950; margin:0;">Intelligence Layer</h6>
            <p style="font-size: 0.8rem; color: #8b949e; margin-top:10px;">
                <b>Back-end:</b> Python, Scikit-learn, TensorFlow.<br>
                <b>Core:</b> Random Forest, XGBoost, and CNN for image classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with t2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; border-left: 3px solid #58a6ff; height:100%;">
            <h6 style="color: #58a6ff; margin:0;">Experience Layer</h6>
            <p style="font-size: 0.8rem; color: #8b949e; margin-top:10px;">
                <b>Front-end:</b> Streamlit (Reactive Framework).<br>
                <b>Design:</b> Custom CSS3 .
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================
    # 👨‍💻 DEVELOPER & FORMAL DISCLAIMER
    # =========================================
    st.markdown("---")
    
    dev_col1, dev_col2 = st.columns([2, 1])
    with dev_col1:
        st.error("⚠️ **Formal Medical Disclaimer:**")
        st.caption("""
            DiagnoVix is an Medical Diagnosis academic project intended for preliminary screening and educational purposes. 
            The predictions generated by the ML models are based on statistical probabilities and must NOT be used 
            as a substitute for professional medical diagnosis. Always consult a certified healthcare professional 
            for medical advice and clinical decisions.
        """)
    
    with dev_col2:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 20px; color: #8b949e; font-size: 0.85rem;">
            <b>Lead Developer:</b> Roshan<br>
            <b>Academic Session:</b> 2026<br>
            <b>Project Hub:</b> GitHub | LinkedIn
        </div>
        """, unsafe_allow_html=True)

    footer()
   
# =========================================
# 🩺 GENERIC PREDICTION FORM
# =========================================
def prediction_form(disease_name_form, features_list, model_obj, scaler_obj=None):
    if model_obj is None:
        st.error(f"{disease_name_form} model is unavailable. Check the model file and dependency versions.")
        return


    st.subheader(f"🩺 {disease_name_form} Clinical Assessment")

    # =========================================
    # 📥 INPUT FORM
    # =========================================
    with st.form(f"{disease_name_form.lower().replace(' ', '_')}_form", clear_on_submit=False):

        st.markdown("#### Enter Clinical Parameters:")

        form_cols = st.columns(3)
        inputs_dict = {}

        for i, feature_item in enumerate(features_list):

            with form_cols[i % 3]:

                f_key = feature_item.strip()
                label_text = feature_fullforms.get(disease_name_form, {}).get(f_key, f_key)

                if f_key in FEATURE_INPUTS:
                    ftype, *params = FEATURE_INPUTS[f_key]

                    if ftype == 'slider':
                        min_val, max_val, step = params

                        inputs_dict[f_key] = st.number_input(
                            label_text,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(min_val),
                            step=float(step),
                            format="%.2f"
                        )
                    elif ftype == 'select':
                        inputs_dict[f_key] = st.selectbox(
                            label_text,
                            params[0],
                            key=f"{f_key}_{disease_name_form}_{i}"
                        )
                else:
                    inputs_dict[f_key] = st.text_input(
                        f_key,
                        key=f"{f_key}_{disease_name_form}_fallback"
                    )

        submitted_form = st.form_submit_button("🔍 Predict")

    # =========================================
    # 🔍 PREDICTION LOGIC (FIXED POSITION)
    # =========================================
    if submitted_form:

        def convert_categorical_to_numeric(val):
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in ['yes', 'present', 'male', 'normal']:
                    return 1
                if val_lower in ['no', 'notpresent', 'female', 'abnormal']:
                    return 0
            return val

        try:
            input_values_ordered = [
                convert_categorical_to_numeric(inputs_dict[feat.strip()])
                for feat in features_list
            ]

            data_df = pd.DataFrame([input_values_ordered], columns=features_list)
            prediction_data = scaler_obj.transform(data_df) if scaler_obj else data_df

            prediction_result_val = model_obj.predict(prediction_data)[0]

            confidence_val = 0.0
            if hasattr(model_obj, 'predict_proba'):
                proba = model_obj.predict_proba(prediction_data)[0]
                confidence_val = proba[1] if prediction_result_val == 1 else proba[0]

            diagnosis_str = "Positive" if prediction_result_val == 1 else "Negative"

            st.session_state['prediction_result_data'] = {
                'title': disease_name_form,
                'inputs': inputs_dict,
                'processed_inputs': dict(zip(features_list, input_values_ordered)),
                'diagnosis': diagnosis_str,
                'confidence': f"{confidence_val:.2%}"
            }

            # 🎯 RESULT UI
            st.markdown("### 🧾 AI Diagnosis Result")

            risk = float(confidence_val) if prediction_result_val == 1 else 1 - float(confidence_val)

            if diagnosis_str == "Positive":
                if risk < 0.3:
                    st.success("🟢 Low Risk Detected")
                elif risk < 0.7:
                    st.warning("🟡 Moderate Risk Detected")
                else:
                    st.error("🔴 High Risk Detected")
            else:
                st.success("🟢 No Significant Risk Detected")

            st.progress(int(risk * 100))
            st.caption(f"Model Confidence: {round(confidence_val * 100, 2)}%")

            st.info(f"Diagnosis: {diagnosis_str}")

            # 🧾 Patient Summary
            patient_data = st.session_state.get("current_p_data", {"name": "N/A", "age": "N/A"})

            st.markdown(f"""
            <div class="glass-card">
                <h4>🧾 Patient Summary</h4>
                <p><b>Name:</b> {patient_data.get('name')}</p>
                <p><b>Age:</b> {patient_data.get('age')}</p>
                <p><b>Assessment:</b> {disease_name_form}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # 📊 Analytics
            st.session_state.total_diagnoses += 1
            db_action("UPDATE stats SET diagnoses = diagnoses + 1 WHERE id = 1")
            st.session_state.activity_log.append(f"Performed {disease_name_form} diagnosis")

            # 📄 Report
            show_medical_report_pdf()

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
# =========================================
# 📄 MEDICAL REPORT DISPLAY + PDF DOWNLOAD
# =========================================
def show_medical_report_pdf():
    result_data = st.session_state.get('prediction_result_data', {})
    # Retrieves the name captured in the UI session state
    patient_data = st.session_state.get('current_p_data', {"name": "N/A", "age": "N/A"})

    if not result_data:
        st.error("No clinical data found to generate report.")
        return

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=70)

    styles = getSampleStyleSheet()
    
    # --- FIXED STYLE DEFINITIONS ---
    # Increased leading to prevent vertical overlapping
    styles.add(ParagraphStyle(name='CenteredSmall', parent=styles['Normal'], fontSize=9, alignment=1, leading=12))
    styles.add(ParagraphStyle(name='SectionLabel', fontSize=12, leading=14, fontName='Helvetica-Bold', spaceBefore=20, spaceAfter=10))
    styles.add(ParagraphStyle(name='FooterText', fontSize=8, leading=10, textColor=colors.grey, alignment=1))

    story = []

    # --- 🏥 STABILIZED BRANDING HEADER (FIXES OVERLAP & LOGO) ---
    # We combine logo and title into a nested table to lock their positions
    header_content = [
        [Paragraph("🩺 DiagnoVix", ParagraphStyle(name='HeaderTitle', fontSize=26, leading=32, alignment=1, fontName='Helvetica-Bold'))],
        [Paragraph("<b>MEDICAL ANALYSIS REPORT</b>", styles['CenteredSmall'])],
        [Paragraph(f"System Version: {APP_VERSION} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['CenteredSmall'])]
    ]
    
    # colWidths=[480] ensures it takes up the center of the A4 page
    header_table = Table(header_content, colWidths=[480])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(header_table)
    
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.black, thickness=1.5))
    story.append(Spacer(1, 25))

    # --- 👤 I. PATIENT RECORD ---
    story.append(Paragraph("I. PATIENT RECORD", styles['SectionLabel']))
    
    # Properly printing Sakshi Kale or other patient names
    p_info = [
        ["Patient Name:", str(patient_data['name'])], 
        ["Age:", str(patient_data['age'])], 
        ["Clinical Reference:", f"DVX-{datetime.now().strftime('%M%S')}"]
    ]
    t1 = Table(p_info, colWidths=[140, 300])
    t1.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ]))
    story.append(t1)

    # --- 📑 II. CLINICAL INPUT PARAMETERS ---
    story.append(Paragraph("II. CLINICAL INPUT PARAMETERS", styles['SectionLabel']))
    input_rows = [["Clinical Indicator", "Observed Value"]]
    for k, v in result_data.get('inputs', {}).items():
        label_k = feature_fullforms.get(result_data['title'], {}).get(k, k)
        input_rows.append([label_k, str(v)])
    
    t2 = Table(input_rows, colWidths=[300, 140])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)

    # --- 📊 III. DIAGNOSTIC SUMMARY ---
    story.append(Paragraph("III. DIAGNOSTIC SUMMARY", styles['SectionLabel']))
    summary_data = [
        ["Assessment Target:", str(result_data['title'])],
        ["Result Findings:", str(result_data['diagnosis'])],
        ["AI Confidence Level:", str(result_data['confidence'])]
    ]
    t3 = Table(summary_data, colWidths=[160, 280])
    t3.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1.2, colors.black), 
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke), 
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t3)

    # --- ⚖️ BOTTOM FOOTER ---
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.black)
        canvas.line(40, 65, 555, 65)
        footer_p = Paragraph("<b>LIMITATION OF LIABILITY:</b> This automated report is for screening purposes only. Final clinical diagnosis must be validated by a board-certified physician.", styles['FooterText'])
        w, h = footer_p.wrap(doc.width, doc.bottomMargin)
        footer_p.drawOn(canvas, doc.leftMargin, 35)
        canvas.restoreState()

    try:
        doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Formal Clinical Report", 
            data=buffer, 
            file_name=f"Report_{patient_data['name'].replace(' ', '_')}.pdf", 
            mime="application/pdf", 
            use_container_width=True
        )
    except Exception as e:
        st.error(f"PDF Generation Error: {e}")
                        
# =========================================
# 🔍 SYMPTOM SELECTION HELPER
# =========================================
def get_selected_symptoms_from_checkboxes(all_symptoms_list, column_layout):

    selected = []

    for i, symptom in enumerate(all_symptoms_list):

        current_col = column_layout[i % len(column_layout)]

        with current_col:
            if st.checkbox(
                symptom.capitalize(),
                key=f"symptom_{i}"
            ):
                selected.append(symptom)

    return selected

# =========================================
# 🚀 MAIN APPLICATION CONTROLLER
# =========================================

# --- 📌 Get current page from URL query params ---
query_params = st.query_params
current_page = query_params.get("page", PAGE_HOME)

# Streamlit sometimes returns list → handle safely
if isinstance(current_page, list):
    current_page = current_page[0]


# =========================================
# 🧭 TOP NAVIGATION BAR (THE DEFINITIVE FIX)
# =========================================
# We define the current state variables
current_theme = st.session_state.theme
theme_icon = "☀️" if current_theme == "dark" else "🌙"

# We use ONE single markdown call to ensure NO code leaks
st.markdown(f"""
<div class="topnav">
    <h1>🩺 DiagnoVix</h1>
    <div class="nav-links">
        <a href="?page={PAGE_HOME}&theme={current_theme}" target="_self" class="{'nav-active' if current_page==PAGE_HOME else ''}">Home</a>
        <a href="?page={PAGE_DASHBOARD}&theme={current_theme}" target="_self" class="{'nav-active' if current_page==PAGE_DASHBOARD else ''}">Dashboard</a>
        <a href="?page={PAGE_DIAGNOSE}&theme={current_theme}" target="_self" class="{'nav-active' if current_page==PAGE_DIAGNOSE else ''}">Diagnose</a>
        <a href="?page={PAGE_CHATBOT}&theme={current_theme}" target="_self" class="{'nav-active' if current_page==PAGE_CHATBOT else ''}">Chatbot</a>
        <a href="?page={current_page}&toggle_theme=true&theme={current_theme}" target="_self" style="font-size: 1.2rem; margin-left: 15px; text-decoration: none;">{theme_icon}</a>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================

for warning in st.session_state.get("startup_warnings", []):
    st.warning(warning)

# 📄 PAGE RENDERING LOGIC
# =========================================

# ---------- 🏠 HOME PAGE ----------
if current_page == PAGE_HOME:
    # Render landing page UI
    render_home_page()

  
# ---------- 📊 DASHBOARD ----------
# ---------- 📊 DASHBOARD (SECURED ADMIN PORTAL) ----------
elif current_page == PAGE_DASHBOARD:
    # 1. Initialize the specific login state for Admin
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        # 📢 PUBLIC NOTIFICATION MESSAGE
        st.markdown("""
        <div style="background: rgba(248, 81, 73, 0.1); padding: 20px; border-radius: 12px; border-left: 5px solid #f85149; border: 1px solid rgba(248, 81, 73, 0.2); margin-bottom: 25px;">
            <h4 style="color: #f85149; margin-top:0;">🚫 Restricted Access</h4>
            <p style="color: #f8fafc; font-size: 0.95rem; margin-bottom: 0;">
                The <b>System Analytics Dashboard</b> is a private research module reserved for authorized administrators. 
                It contains internal clinical statistics, activity logs, and persistence data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- GLOSSY LOGIN INTERFACE ---
        _, login_col, _ = st.columns([1, 2, 1])
        
        with login_col:
            st.markdown('<div class="card" style="padding: 30px; border-top: 4px solid #2563eb;">', unsafe_allow_html=True)
            with st.form("admin_access_gate"):
                st.markdown("### 🔐 Admin Login")
                u_input = st.text_input("Username")
                p_input = st.text_input("Password", type="password")
                
                if st.form_submit_button("🔓 Access Dashboard", use_container_width=True):
                    if u_input == ADMIN_USER and p_input == ADMIN_PASS:
                        st.session_state.admin_logged_in = True
                        st.success("Access Granted.")
                        st.rerun()
                    else:
                        st.error("Invalid Credentials.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        # --- RENDER THE PRIVATE DASHBOARD DATA ---
        logout_col1, logout_col2 = st.columns([6, 1])
        with logout_col2:
            if st.button("🔒 Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()
        
        # 🔥 ONLY CALL THIS HERE (Inside the 'else')
        render_dashboard()
        
    
# ---------- 💬 CHATBOT ----------
elif current_page == PAGE_CHATBOT:

    import os
    from google import genai
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # 1. Initialize persistent chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "chat_queries" not in st.session_state:
        st.session_state.chat_queries = 0

    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []

    st.title("💬 AI Medical Assistant")

    # --- ℹ️ INFO SECTION ---
    st.markdown("""
    <div class="card" style="margin-bottom: 25px; border-left: 5px solid #2563eb;">
        <h4>📖 About this Intelligent System</h4>
        <p>
            This assistant uses <b>RAG (Retrieval-Augmented Generation)</b> with FAISS 
            and Gemini AI to provide clinically grounded answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- ENGINE CONFIG ---
    st.markdown("### ⚙️ Engine Configuration")
    col1, col2 = st.columns(2)

    with col1:
        # 🔥 FORCE FREE MODEL
        SELECTED_ENGINE = "gemini-2.5-flash"
        st.success(f"Using Model: {SELECTED_ENGINE}")

    with col2:
        st.info("🟢 API Status: Connected")

    st.markdown("---")

    # --- VECTOR DB LOADER ---
    @st.cache_resource(show_spinner="🔄 Loading Medical Knowledge Base...")
    def get_vectorstore():
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"❌ FAISS DB not found at {DB_FAISS_PATH}")
            return None

        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )

        try:
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"DB Load Error: {e}")
            return None

    # --- CHAT FUNCTION ---
    def run_gemini_rag_chat():

        # Show history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt_input = st.chat_input("Ask medical questions...")

        if not prompt_input:
            return

        # Show user message
        st.chat_message("user").markdown(prompt_input)
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt_input
        })

        with st.spinner("🤖 Thinking..."):

            try:
                GEMINI_KEY = os.getenv("GEMINI_API_KEY")
                if not GEMINI_KEY:
                    st.error("❌ GEMINI_API_KEY missing!")
                    return

                # Gemini Client
                client = get_gemini_client()

                # Load FAISS
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    return

                # Retrieve context
                docs = vectorstore.similarity_search(prompt_input, k=3)
                context = "\n".join([d.page_content for d in docs])

                if not context.strip():
                    context = "No relevant context found."

                # RAG Prompt
                rag_prompt = f"""
You are a professional medical AI.

ONLY use the context below.
If answer not found, say "I don't know"    .

Context:
{context}

Question:
{prompt_input}
"""

                # 🔥 CORRECT API CALL (FREE MODEL)
                response = client.models.generate_content(
                    model=SELECTED_ENGINE,
                    contents=rag_prompt
                )

                answer = response.text or "No response generated."

            except Exception as e:
                # 🔥 SAFE FALLBACK
                answer = "⚠️ Free quota reached or API issue. Try again later."
                st.error(f"Error: {e}")

        # Show response
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.chat_queries += 1
        st.session_state.activity_log.append("AI Chat Used")

        st.rerun()

    # Run chatbot
    run_gemini_rag_chat()
    
    
    
# =========================================
# 🔬 CLINICAL ASSESSMENT CENTER (RESTRUCTURED)
# =========================================
elif current_page == PAGE_DIAGNOSE:

    # --- ℹ️ STATE MANAGEMENT ---
    # Separate states for each assessment type to allow independent shuffling
    if "diag_view_state" not in st.session_state:
        st.session_state.diag_view_state = "INPUT" 
    if "img_view_state" not in st.session_state:
        st.session_state.img_view_state = "INPUT"

    st.title("🔬 Clinical Assessment Center")
    
    # --- ℹ️ PROFESSIONAL HEADER ---
    st.markdown(f"""
    <div style="background: {card_bg}; padding: 20px; border-radius: 12px; border-left: 5px solid #3fb950; border: 1px solid {border_color}; margin-bottom: 25px;">
        <h4 style="color: {text_color}; margin-top:0;">📋Diagnostic Suite</h4>
        <p style="color: {secondary_text}; font-size: 0.9rem; margin-bottom: 0;">
         Multimodal AI-powered decision support system for clinical analysis, disease prediction, and radiographic intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Unique Tab Styling via the Three-Tab System
    tab_symptom, tab_ml, tab_image = st.tabs([
        "🧠 Symptom Assessment", 
        "🧪 Disease Prediction", 
        "🖼️ Radiographic Analysis"
    ])

    # =========================================
    # 🧠 TAB 1: SMART SYMPTOM MAPPING
    # =========================================
    with tab_symptom:
        if st.session_state.diag_view_state == "INPUT":
            st.markdown("### 🧬 Patient Registration & Symptom Mapping")
            
            # Isolated Registration for this Tab
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                s_name = st.text_input("Full Patient Name", key="s_name_in")
            with col_s2:
                s_age = st.number_input("Patient Age", 1, 120, 25, key="s_age_in")

            all_symptoms = sorted(set(s for sub in diseases_data.values() for s in sub))
            with st.expander("🩺 Select Symptoms", expanded=True):
                cols = st.columns(4)
                selected_symptoms = []
                for i, symptom in enumerate(all_symptoms):
                    if cols[i % 4].checkbox(symptom.capitalize(), key=f"s_symp_{i}"):
                        selected_symptoms.append(symptom)

            # Compact Button Logic
            # --- Inside Tab 1: Smart Symptom Mapping ---
            btn_col, _ = st.columns([1, 3])
            with btn_col:
                if st.button("🚀 Run Analysis", type="primary", use_container_width=True, key="run_symptom_btn"):
                    if not selected_symptoms:
                        st.warning("Please select at least one indicator.")
                    elif not s_name:
                        st.warning("⚠️ Patient Name is required to generate the report record.")
                    else:
                        # 🔥 THE FIX: Explicitly save the name and age here
                        st.session_state.current_p_data = {
                            "name": s_name, 
                            "age": s_age
                        }
                        st.session_state.current_selected_symptoms = selected_symptoms
                        st.session_state.diag_view_state = "RESULT"
                        st.rerun()

        elif st.session_state.diag_view_state == "RESULT":
            st.markdown("### 📑 Clinical Correlation Report")
            p = st.session_state.current_p_data
            
            # Glossy Patient Record Banner
            st.markdown(f"""
                <div style="background: rgba(37, 99, 235, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #2563eb; margin-bottom: 20px;">
                    <span style="color: {text_color};"><b>PATIENT RECORD:</b> {p['name']}</span> | 
                    <span style="color: {text_color};"><b>AGE:</b> {p['age']}</span>
                </div>
            """, unsafe_allow_html=True)

            # Pattern Recognition Logic
            disease_scores = {d: len([s for s in st.session_state.current_selected_symptoms if s in symps]) 
                             for d, symps in diseases_data.items()}
            results = sorted([(d, s) for d, s in disease_scores.items() if s > 0], key=lambda x: x[1], reverse=True)[:3]

            if not results:
                st.info("No matching clinical patterns identified.")
            else:
                for disease, score in results:
                    total = len(diseases_data[disease]); conf = int((score/total)*100)
                    color = "#ef4444" if conf > 50 else "#fb923c"
                    st.markdown(f"""
                    <div style="background:{card_bg}; padding:24px; border-radius:12px; border-left:8px solid {color}; border:1px solid {border_color}; margin-bottom:20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="margin:0; color:{text_color};">{disease}</h3>
                            <span style="background:{color}; color:white; padding:6px 14px; border-radius:50px; font-size:0.8rem; font-weight:700;">{conf}% MATCH</span>
                        </div>
                        <p style="color:{secondary_text}; font-size:0.85rem; margin-top:15px;">The patient exhibits <b>{score} out of {total}</b> primary clinical indicators.</p>
                    </div>""", unsafe_allow_html=True)

            # --- REPORT DOWNLOAD & NAVIGATION ---
            # Preparing result_data for existing PDF function
            st.session_state.prediction_result_data = {
                'title': 'Smart Symptom Mapping',
                'diagnosis': results[0][0] if results else "Inconclusive",
                'confidence': f"{int((results[0][1]/len(diseases_data[results[0][0]]))*100)}%" if results else "0%",
                'inputs': {f"Indicator {i+1}": s for i, s in enumerate(st.session_state.current_selected_symptoms)}
            }
            
            nav_col1, nav_col2 = st.columns([1, 1])
            with nav_col1:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    show_medical_report_pdf()
            with nav_col2:
                if st.button("⬅️ New Assessment", use_container_width=True):
                    st.session_state.diag_view_state = "INPUT"; st.rerun()

    # =========================================
    # 🧪 TAB 2: DISEASE-SPECIFIC ML
    # =========================================
    with tab_ml:
        st.markdown("### 🧪 Targeted Clinical ML Models")
        
        # Isolated Registration for ML Tab
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            m_name = st.text_input("Full Patient Name", key="m_name_in")
        with col_m2:
            m_age = st.number_input("Patient Age", 1, 120, 25, key="m_age_in")
            
        available_ml = [k for k in MODEL_MAPPING.keys() if k != "Symptom Checker"]
        selected_ml = st.selectbox("🎯 Select Targeted Clinical Assessment", available_ml)

        # Force state sync for the PDF generator
        st.session_state.current_p_data = {"name": m_name, "age": m_age}

        m_info = MODEL_MAPPING[selected_ml]
        prediction_form(m_info["name"], m_info["features"], m_info["model"], m_info.get("scaler"))

    # =========================================
    # 🖼️ TAB 3: RADIOGRAPHIC ANALYSIS
    # =========================================
    with tab_image:
        if st.session_state.img_view_state == "INPUT":
            st.markdown("### 🧬 Radiographic AI Analysis")
            
            # Isolated Registration for Image Tab
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                i_name = st.text_input("Full Patient Name", key="i_name_in")
            with col_i2:
                i_age = st.number_input("Patient Age", 1, 120, key="i_age_in")
                
            img_engine = st.selectbox("Select AI Engine:", [IMAGE_AI_BRAIN_STROKE, IMAGE_AI_GEMINI])
            img_file = st.file_uploader("Upload Radiograph (CT/MRI/X-Ray)", type=["jpg", "png"], key="i_upload")

            if img_file:
                st.image(Image.open(img_file), width=350, caption="Uploaded Clinical Asset")
                btn_col, _ = st.columns([1, 3])
                with btn_col:
                    if st.button("🔍 Analyze Radiograph", type="primary", use_container_width=True):
                        if not i_name:
                            st.warning("Patient Name is required.")
                        else:
                            # Lock data for image shuffle
                            st.session_state.img_p_data = {"name": i_name, "age": i_age, "file": img_file, "model": img_engine}
                            st.session_state.img_view_state = "RESULT"
                            st.rerun()

        elif st.session_state.img_view_state == "RESULT":
            st.markdown("### 📄 Radiographic Insight Report")
            data = st.session_state.img_p_data
            image = Image.open(data["file"])
            
            # Glossy Banner
            st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #3b82f6; margin-bottom: 20px;">
                    <span style="color: {text_color};"><b>PATIENT:</b> {data['name']}</span> | 
                    <span style="color: {text_color};"><b>AGE:</b> {data['age']}</span> |
                    <span style="color: #3b82f6;"><b>ENGINE:</b> {data['model']}</span>
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("🤖 Analyzing radiographic patterns..."):
                if data["model"] == IMAGE_AI_BRAIN_STROKE:
                    if brainStrokemodel is not None:
                        processed = preprocess_brain_stroke_image(image)
                        pred = brainStrokemodel.predict(processed)[0][0]
                        res_text = "Acute Stroke Detected" if pred > 0.5 else "No Stroke Detected"
                        res_color = "#ef4444" if pred > 0.5 else "#22c55e"
                        content = f"<b>Diagnostic Findings:</b> {res_text} (Confidence: {pred:.2%})<br><b>Methodology:</b> CNN-based radiographic density classification."
                    else:
                        res_color = "#f59e0b"
                        content = (
                            "<b>Model Unavailable:</b> The stroke detection model could not be loaded.<br>"
                            "<b>Status:</b> This feature is currently under maintenance — please try again later "
                            "or contact support if the issue persists."
                        )
                else:
                    prompt = f"Analyze this medical image for patient {data['name']}, age {data['age']}. Provide a professional clinical summary."
                    content = get_gemini_image_diagnosis_response(prompt, image)
                    res_color = "#3b82f6"

            st.markdown(f"""
            <div style="background:{card_bg}; padding:24px; border-radius:12px; border-left:8px solid {res_color}; border: 1px solid {border_color};">
                <h3 style="margin:0; color:{text_color};">Radiographic Analysis Report</h3>
                <div style="margin-top:15px; padding-top:15px; border-top: 1px solid {border_color}; color:{text_color}; line-height:1.6;">
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Analytics updates
            st.session_state.images_analyzed += 1
            db_action("UPDATE stats SET images = images + 1 WHERE id = 1")

            if st.button("⬅️ New Analysis", key="back_img_btn"):
                st.session_state.img_view_state = "INPUT"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    footer()
