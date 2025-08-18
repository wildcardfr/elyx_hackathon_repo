import streamlit as st
import pandas as pd
import numpy as np

# =========================
# App Configuration
# =========================
st.set_page_config(
    page_title="Elyx Hackathon App",
    page_icon="🤖",
    layout="wide"
)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["🏠 Home", "📝 Text Analysis", "📂 File Upload", "📊 Data Insights", "ℹ️ About"])

# =========================
# Helper Functions
# =========================
def analyze_text(text):
    """Basic text analysis - word & char count"""
    words = text.split()
    return {
        "Word Count": len(words),
        "Character Count": len(text),
        "Unique Words": len(set(words))
    }

def load_csv(uploaded_file):
    """Load and return CSV file as DataFrame"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        st.error("❌ Could not read the file. Please upload a valid CSV.")
        return None

# =========================
# Main Pages
# =========================
if menu == "🏠 Home":
    st.title("🤖 Elyx Hackathon App")
    st.markdown("Welcome to the demo app for **Hackathon Submission** 🎉")
    st.info("👉 Use the sidebar to explore different features.")

elif menu == "📝 Text Analysis":
    st.title("📝 Text Analysis")
    user_input = st.text_area("Enter text to analyze:", "")
    
    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            result = analyze_text(user_input)
            st.success("✅ Analysis Complete")
            st.json(result)

elif menu == "📂 File Upload":
    st.title("📂 File Upload & Preview")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head())

elif menu == "📊 Data Insights":
    st.title("📊 Data Insights")
    st.markdown("Demo visualization with random data.")
    
    data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["Feature A", "Feature B", "Feature C"]
    )
    st.line_chart(data)

elif menu == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This is a **Streamlit-based app** built for Hackathon submission.  
    It demonstrates:
    - Text analysis 📝  
    - File upload & preview 📂  
    - Data visualization 📊  
    - User-friendly navigation 🚀
    """)

