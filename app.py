import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
from groq import Groq
import os
from dotenv import load_dotenv
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. SETUP & CONFIG
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="ANOVA Pro: Replication Analysis", layout="wide")

st.title("📊 Two-Way ANOVA Suite (with repetitions)")

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    st.sidebar.warning("⚠️ Groq API Key missing in .env")

# 2. SIDEBAR - SETTINGS
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Table format with replications)", type="csv")

st.sidebar.header("📊 Hypothesis Settings")
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05)
user_context = st.sidebar.text_area("Analysis Context", "Enter context (e.g., Effect of Feeding and Plant Type on Growth)")

if uploaded_file is not None:
    # 3. DATA LOADING & AUTOMATIC REFORMATTING
    raw_df = pd.read_csv(uploaded_file)
    
    # Identify Factor A (first column) and Factor B (remaining headers)
    factor_a_col = raw_df.columns[0]
    factor_b_cols = raw_df.columns[1:].tolist()
    
    # Convert "Wide" table to "Long" format for ANOVA
    df_long = raw_df.melt(id_vars=factor_a_col, 
                         value_vars=factor_b_cols, 
                         var_name='Factor_B', 
                         value_name='Value')
    df_long.rename(columns={factor_a_col: 'Factor_A'}, inplace=True)
    
    # Clean names for statsmodels formula compatibility
    df_long['Factor_A'] = df_long['Factor_A'].astype(str).str.replace(' ', '_')
    df_long['Factor_B'] = df_long['Factor_B'].astype(str).str.replace(' ', '_')

    st.markdown("### 1. Data Preview (Formatted for Analysis)")
    st.caption(f"Detected Factor A: '{factor_a_col}' | Factor B: '{', '.join(factor_b_cols)}'")
    st.dataframe(df_long.head(), use_container_width=True)

    # 4. STATISTICAL CALCULATIONS
    # Formula includes Interaction: Value ~ A + B + A:B
    model_formula = "Value ~ C(Factor_A) + C(Factor_B) + C(Factor_A):C(Factor_B)"
    model = ols(model_formula, data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # 5. ANOVA DASHBOARD
    st.markdown("---")
    st.markdown("### 2. ANOVA Summary Table")
    
    p_a = anova_table.loc["C(Factor_A)", "PR(>F)"]
    p_b = anova_table.loc["C(Factor_B)", "PR(>F)"]
    p_inter = anova_table.loc["C(Factor_A):C(Factor_B)", "PR(>F)"]

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Sig: {factor_a_col}", "Yes" if p_a < alpha else "No", f"p={p_a:.4f}")
    m2.metric(f"Sig: Factor B Groups", "Yes" if p_b < alpha else "No", f"p={p_b:.4f}")
    m3.metric("Sig: Interaction", "Yes" if p_inter < alpha else "No", f"p={p_inter:.4f}")

    st.table(anova_table)

    # 6. INTERACTION VISUALIZATION
    st.markdown("### 3. Interaction & Distribution Plots")
    v1, v2 = st.columns(2)
    with v1:
        fig1 = px.box(df_long, x="Factor_B", y="Value", color="Factor_A", 
                      title=f"Response by {factor_a_col} across Groups", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    with v2:
        # Interaction Line Plot (Marginal Means)
        means = df_long.groupby(['Factor_A', 'Factor_B'])['Value'].mean().reset_index()
        fig2 = px.line(means, x="Factor_B", y="Value", color="Factor_A", markers=True,
                       title="Interaction Plot (Means)", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    # 7. POST HOC ANALYSIS (Tukey HSD)
    tukey_text = "Not required (No significant effects found)."
    if p_a < alpha or p_b < alpha or p_inter < alpha:
        st.markdown("---")
        st.markdown("### 🔍 4. Post Hoc Analysis (Tukey HSD)")
        
        # Combine factors to see which specific combinations differ
        df_long['Combined'] = df_long['Factor_A'] + " x " + df_long['Factor_B']
        tukey = pairwise_tukeyhsd(endog=df_long['Value'], groups=df_long['Combined'], alpha=alpha)
        
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        sig_pairs = tukey_df[tukey_df['reject'] == True]
        
        # UI Metrics
        k1, k2 = st.columns(2)
        k1.metric("Significant Pairs Found", len(sig_pairs))
        top_group = df_long.groupby('Combined')['Value'].mean().idxmax()
        top_val = df_long.groupby('Combined')['Value'].mean().max()
        k2.metric("Highest Impacted Combination", top_group, f"Mean: {top_val:.2f}")

        st.dataframe(
            tukey_df.style.applymap(lambda x: 'background-color: lightgreen' if x == True else '', subset=['reject']), 
            use_container_width=True
        )
        tukey_text = sig_pairs[['group1', 'group2', 'meandiff']].to_string()

    # 8. CONSOLIDATED POINT-WISE AI REPORT
    st.markdown("---")
    if st.button("Generate Point-Wise Strategic Report"):
        if GROQ_API_KEY:
            with st.spinner("Generating Structured Report..."):
                prompt = f"""
                You are a Senior Data Scientist. Provide a high-readability report.
                
                CONTEXT: {user_context}
                FACTORS: {factor_a_col} and {factor_b_cols}
                INTERACTION P-VALUE: {p_inter:.4f} (Alpha: {alpha})
                SIGNIFICANT PAIRS: {tukey_text}

                FORMATTING INSTRUCTIONS:
                - Use ONLY bullet points for every section.
                - Do not write long paragraphs.
                - Use bold text for key findings.

                CRITICAL INSTRUCTION: Do not use the word 'Winner' or 'Best'. 
                Use neutral terms like 'Highest Recorded Impact', 'Lowest Recorded Impact', 
                'Most Variable Combination', or 'Stable Combination'. 
                Evaluate if a 'High' value is desirable or undesirable based on the context (e.g., high wear is bad).

                REQUIRED SECTIONS:
                1. Interaction Summary: 
                    - State if variables are interdependent.
                2. Impact Analysis:
                    - Identify which combinations show the most extreme differences.
                    - Reference specific mean differences from the Tukey data.
                3. Operational Implications:
                    - Provide 3 strategic bullet points for stakeholders based on the significant findings.
                """                
                try:
                    res = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    st.success("Analysis Complete")
                    st.info(res.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI Error: {e}")    

    # # 8. CONSOLIDATED AI INFERENCE
    # st.markdown("---")
    # if st.button("Generate Strategic AI Report"):
    #     if GROQ_API_KEY:
    #         with st.spinner("Analyzing Variance & Interactions..."):
    #             prompt = f"""
    #             You are a Senior Statistician. Analyze this Two-Way ANOVA (with Replications) results.
                
    #             CONTEXT: {user_context}
    #             FACTOR A (Rows): {factor_a_col} (p={p_a:.4f})
    #             FACTOR B (Columns): {factor_b_cols} (p={p_b:.4f})
    #             INTERACTION EFFECT: p={p_inter:.4f}
    #             ALPHA: {alpha}
                
    #             SIGNIFICANT PAIRWISE DIFFERENCES:
    #             {tukey_text}

    #             REQUIRED TASKS:
    #             1. Executive Summary: Is there a significant interaction? Explain what that means for {user_context}.
    #             2. Factor Comparison: Which factor (A or B) had the strongest influence?
    #             3. Best Combination: Based on means and significance, which specific combination is the 'Winner'?
    #             4. Strategic Recommendation: 3 bullet points for a stakeholder using the names from the context.
    #             """
    #             try:
    #                 res = client.chat.completions.create(
    #                     model="llama-3.3-70b-versatile",
    #                     messages=[{"role": "user", "content": prompt}],
    #                     temperature=0.2
    #                 )
    #                 st.info(res.choices[0].message.content)
    #             except Exception as e:
    #                 st.error(f"AI Error: {e}")
        else:
            st.error("API Key missing.")
else:
    st.info("Please upload a CSV file in the 'Table with Replications' format to begin.")