# POSB Twin-Stage Credit Studio

A Streamlit application for Bayesian and XGBoost-based credit scoring, model comparison, borrower-level risk forecasting, scenario testing, and PDF report generation.

## Local run

```bash
pip install -r requirements.txt
streamlit run main.py
```

Default demo login:

```text
Username: admin
Password: admin123
```

## Streamlit Community Cloud deployment

1. Create a GitHub repository.
2. Upload all files in this folder to the repository root.
3. Confirm these files are present in the repository root:
   - `main.py`
   - `requirements.txt`
   - `runtime.txt`
   - `data.xlsx`
   - `utils/`
   - `.streamlit/config.toml`
4. Go to Streamlit Community Cloud and create a new app from the GitHub repository.
5. Set the main file path to:

```text
main.py
```

6. Deploy the app.

## Notes

- The SQLite database file is created automatically at runtime and is intentionally ignored by Git.
- Do not commit private client data, credentials, or `.streamlit/secrets.toml`.
- If the deployment fails, first check the build log for missing packages or file path case-sensitivity issues.
