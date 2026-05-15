# ANOVA Pro: Two-Way with Replication

A high-performance statistical dashboard built with **Streamlit** and **Groq LLM** to perform Two-Way ANOVA (with Interaction) and automated Post-Hoc Analysis.

## 🚀 Features
- **Replication Support:** Handles multiple observations per cell (n > 1).
- **Interaction Effects:** Calculates Factor A, Factor B, and A×B interaction significance.
- **Tukey HSD Post Hoc:** Automatically identifies significant pairings.
- **AI Insights:** Uses Llama 3 via Groq to translate p-values into business strategy.

## 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/anova-pro-replication.git](https://github.com/your-username/anova-pro-replication.git)
   cd anova-pro-replication

2. pip install -r requirements.txt

3. GROQ_API_KEY=your_actual_api_key_here

4. streamlit run app.py

5. ### 💡 Pro-Tip for Execution
When you run the app for the first time, ensure your `.env` file is in the **same folder** as your `app.py`. If you're deploying this to a platform like **Streamlit Community Cloud**,           don't use a `.env` file; instead, go to **Settings > Secrets** and paste your `GROQ_API_KEY` there.

anova-pro-replication/
├── .env                # Private API keys (Not for Git)
├── .gitignore          # Tells Git which files to ignore
├── app.py              # The main Streamlit code provided previously
├── requirements.txt    # List of Python dependencies
└── README.md           # Project documentation