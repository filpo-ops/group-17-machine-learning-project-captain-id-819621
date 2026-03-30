"""Streamlit app for NoiPA Data Quality Agents."""
import os, sys, io, json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

try:
    from agents import run_pipeline
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False

st.set_page_config(page_title="NoiPA Data Quality", page_icon="🔍", layout="wide")

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "final_state" not in st.session_state:
    st.session_state.final_state = None

with st.sidebar:
    st.title("🔍 NoiPA Data Quality")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Carica un CSV", type=["csv"])
    if uploaded_file:
        st.session_state.uploaded_df = pd.read_csv(uploaded_file)
        st.success(f"✓ {uploaded_file.name} caricato")
    
    st.markdown("---")
    max_iter = st.slider("Max iterazioni", 1, 5, 2)
    
    if st.session_state.uploaded_df is not None and MODULES_AVAILABLE:
        if st.button("▶️ Analizza", type="primary", use_container_width=True):
            with st.spinner("Analisi..."):
                try:
                    final_state = run_pipeline(st.session_state.uploaded_df, 
                                               dataset_name=uploaded_file.name, 
                                               max_iterations=max_iter)
                    st.session_state.final_state = final_state
                    st.session_state.analysis_complete = True
                    st.success("✓ Analisi completata!")
                except Exception as e:
                    st.error(f"Errore: {e}")

st.title("🎯 NoiPA — Data Quality Analysis")

if st.session_state.uploaded_df is None:
    st.info("👈 Carica un CSV dal pannello laterale")
elif not st.session_state.analysis_complete:
    st.subheader("📋 Anteprima")
    st.metric("Righe", len(st.session_state.uploaded_df))
    st.metric("Colonne", len(st.session_state.uploaded_df.columns))
    st.dataframe(st.session_state.uploaded_df.head(10), use_container_width=True)
elif st.session_state.final_state:
    st.subheader("✓ Analisi Completata")
    fs = st.session_state.final_state
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Dettagli", "🔧 Remediation"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{fs.get('reliability_score', 0):.1%}")
        with col2:
            st.metric("Iterazioni", fs.get("iteration", 1))
        with col3:
            all_i = (fs.get("schema_report",{}).get("issues",[]) + 
                    fs.get("completeness_report",{}).get("issues",[]))
            st.metric("Issues", len(all_i))
    
    with tab2:
        st.write(fs.get("schema_report",{}).get("summary","N/A"))
    
    with tab3:
        if fs.get("dataset") is not None:
            csv = fs["dataset"].to_csv(index=False).encode()
            st.download_button("📥 CSV pulito", csv, "dataset.csv", "text/csv")

st.markdown("---")
st.markdown("<div style='text-align:center; font-size:0.8rem;'>NoiPA — LUISS</div>", unsafe_allow_html=True)
