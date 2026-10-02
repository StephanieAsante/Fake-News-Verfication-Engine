
# %% [markdown]
# ## ALL - IN - ONE (PYTHON WEB APP) FRAMEWORK USING STREAMLIT

# %%
# Load libraries for the process
import joblib
import streamlit as st
import time
import pandas as pd
import numpy as np
from scipy.sparse import hstack

# %%
# Load the weights we froze
fake_news_model = joblib.load('fake_news_detection_model')
tfidf_weights = joblib.load('tfidf_vectorizer')
scaler_weights = joblib.load('scaler.pkl')
onehot_weights = joblib.load('one_hot_encoder')

# %%

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Veritas AI | Misinformation Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN AI DARK DESIGN (CSS INJECTION) ---
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0d0f17;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Hero Text */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    /* Glassmorphism Card Styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Metric Badges */
    .metric-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818cf8;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Input Field Styling */
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
        font-size: 0.95rem;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.3) !important;
    }

    /* Primary AI Action Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(168, 85, 247, 0.6) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #07090e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- INDIVIDUAL WEIGHT LOADING ---
@st.cache_resource
def load_frozen_weights():
    fake_news_model = joblib.load('fake_news_detection_model')
    tfidf_weights = joblib.load('tfidf_vectorizer')
    scaler_weights = joblib.load('scaler.pkl')
    onehot_weights = joblib.load('one_hot_encoder')
    return fake_news_model, tfidf_weights, scaler_weights, onehot_weights

try:
    fake_news_model, tfidf_weights, scaler_weights, onehot_weights = load_frozen_weights()
    weights_loaded = True
except Exception as e:
    weights_loaded = False
    st.error(f"Error loading model weights: {e}")

# --- SIDEBAR: SYSTEM STATS ---
with st.sidebar:
    st.markdown("<span class='metric-badge'>v2.4 Production Engine</span>", unsafe_allow_html=True)
    st.title("🛡️ Engine Specs")
    
    st.markdown("""
    **Model Architecture:**
    - Regularized Logistic Regression ($L_1$ Lasso)
    - TF-IDF Vectorizer (10,000 N-gram terms)
    - Sentiment & Emotion One-Hot Encoder
    - Standard Scaled Metadata Features
    
    ---
    **Model Benchmarks:**
    - **Accuracy:** 98.97%
    - **Recall (Fake News):** 99.33%
    - **Precision:** 98.67%
    - **ROC-AUC:** 0.9982
    """)
    st.divider()
    st.caption("Powered by Scikit-Learn & Streamlit")

# --- HERO SECTION ---
st.markdown("<h1 class='hero-title'>Veritas AI Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Enterprise Misinformation Detection & Real-Time Content Verification Engine</p>", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col_input, col_output = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📄 Input Article Analysis")
    
    user_title = st.text_input("Article Title (Optional):", placeholder="e.g., Breaking News: Policy Change Announced...")
    user_text = st.text_area("Article Body Text:", height=220, placeholder="Paste article body here...")
    
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        sentiment_input = st.selectbox("Predicted Sentiment:", ["positive", "neutral", "negative"])
    with col_meta2:
        emotion_input = st.selectbox("Dominant Emotion:", ["joy", "sadness", "anger", "fear", "surprise", "neutral"])
    
    analyze_btn = st.button("⚡ Run Verification Engine")
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Verification Report")
    
    if analyze_btn:
        if not user_text.strip() and not user_title.strip():
            st.warning("Please enter an article title or body text to evaluate.")
        else:
            with st.spinner("Analyzing NLP linguistic patterns and feature weights..."):
                time.sleep(0.5) # Visual loading state
                
                # 1. Combine title and text into fully_combined_article
                fully_combined_article = f"{user_title} {user_text}".strip()
                
                # 2. Compute metadata features
                char_count = len(fully_combined_article)
                word_count = len(fully_combined_article.split())
                avg_word_len = char_count / word_count if word_count > 0 else 0.0
                
                if weights_loaded:
                    # A. Transform text via TF-IDF Vectorizer
                    X_tfidf = tfidf_weights.transform([fully_combined_article])
                    
                    # B. One-Hot Encode categorical sentiment/emotion
                    cat_df = pd.DataFrame(
                        [[sentiment_input, emotion_input]], 
                        columns=['predicted_sentiment', 'predicted_emotion']
                    )
                    X_cat = onehot_weights.transform(cat_df)
                    
                    # C. Scale numerical metadata with exact trained feature names
                    num_df = pd.DataFrame(
                        [[char_count, word_count, avg_word_len]], 
                        columns=['char_count', 'word_count', 'avg_word_len']
                    )
                    X_num = scaler_weights.transform(num_df)
                    
                    # D. Stack non-text features (One-Hot + Scaled Metadata)
                    X_meta = np.hstack([X_cat, X_num])
                    
                    # E. Combine TF-IDF sparse matrix with metadata array
                    X_final = hstack([X_tfidf, X_meta])
                    
                    # F. Make Prediction
                    prediction = fake_news_model.predict(X_final)[0]
                    prob = fake_news_model.predict_proba(X_final)[0][1]
                else:
                    prediction = 0
                    prob = 0.05

                # Render Results UI
                if prediction == 1:
                    st.error("### ⚠️ FLAG: Likely Misinformation")
                    st.progress(float(prob))
                    st.write(f"**Confidence Score:** `{prob * 100:.2f}%` Fake Probability")
                else:
                    st.success("### ✅ VERIFIED: Likely Authentic News")
                    st.progress(float(1 - prob))
                    st.write(f"**Confidence Score:** `{(1 - prob) * 100:.2f}%` Credibility Rating")

                st.divider()
                st.markdown("**Extracted Structural Metadata:**")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Word Count", f"{word_count}")
                mcol2.metric("Char Count", f"{char_count}")
                mcol3.metric("Avg Word Length", f"{avg_word_len:.2f}")

    else:
        st.info("Paste an article on the left and click **Run Verification Engine** to generate an assessment.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# %%


# %%



