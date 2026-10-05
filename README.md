# CABG PMI Streamlit calculator — upload package

All FIVE files are in one directory. Upload all five files directly into the root of your GitHub repository. Do not upload the ZIP itself or an extra enclosing folder.

## Included files
- `app.py`: Streamlit application.
- `final_model_bundle.joblib`: original locked fitted model, unchanged.
- `requirements.txt`: dependencies, including scikit-learn==1.7.2.
- `model-specification.json`: hyperparameters, all calibration coefficients, feature order and limitations.
- `README.md`: these instructions.

## GitHub upload / GitHub 上传
1. Create a new repository named `cabg-pmi-calculator` under your account. Select **Private**. Do not initialize it with README, .gitignore or a license; this package already contains README.md.
2. On the new repository's Quick setup page, choose **uploading an existing file**. If a file list is already shown, choose **Add file > Upload files**.
3. Extract this ZIP. Select all five files inside and drag them into the upload area.
4. Wait until all five files appear. Enter `Add CABG PMI calculator` as the commit message, then click **Commit changes**.
5. Confirm that `app.py`, `requirements.txt` and `final_model_bundle.joblib` are directly visible in the repository root. Check the actual branch name (usually `main`).

## Streamlit deployment / Streamlit 部署
- Repository: `Pluto1105/cabg-pmi-calculator`, if that is the repository you created.
- Branch: the actual branch shown in GitHub (e.g. `main`).
- Main file path: `app.py`.
- App URL: `cabg-pmi-study-1971681` (availability must be checked by Streamlit).
- Advanced settings > Python version: **3.12**.
- Click **Deploy**. If your private repository is not listed, authorize Streamlit to access that repository.
- Once deployed, open app settings > Sharing and allow public access if you want reviewers to use it without invitations. Keep the GitHub repository private.

## Scope
This is a research implementation. The model file matches the locked model used in the original 193-person external validation. Its SHA-256 is `0c8206ff983d6cecd27639e1f2067614ec2765f77c569558d69db7a1ff9c9861`. No model training or recalibration is performed.

All ten input fields must be completed. Initial numeric entries are illustrative, not observed patient values. This interface does not impute missing values. ASA III=0 and ASA IV=1; other grades are outside this encoding. MCHC uses g/L, and neutrophils use percent. The cryoprecipitate unit mapping remains a study limitation.

The package contains no patient data spreadsheets or analysis datasets. The fitted SVM contains support vectors, so a public model-file release should be considered separately from public access to the app. Deploying this calculator alone does not provide a public archive of the full analysis code.

The application is not intended to diagnose myocardial infarction, guide treatment, or estimate causal oxycodone effects.

## Local run
Use Python 3.12, then run:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
