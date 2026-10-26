# CABG PMI research calculator

Model version: 2026-10-26. Entry point: `app.py`. Python 3.12.

## Model and outcome

Fixed ten-predictor calibrated RBF SVM, refitted using laboratory-verified development outcomes. Valid repeated measurements from the same blood sample are averaged first; the highest specimen-level cTnI during postoperative days 1–3 defines the peak. PMI threshold: ≥70×URL = 2.52 μg/L (URL 0.036 μg/L).

Development: 1,140 patients, 170 events. Fixed-ten averaged out-of-fold AUROC 0.743 (95% CI 0.701–0.783). Separate selection-inclusive internal validation: mean repeat AUROC 0.711 (95% CI 0.674–0.746), averaged-probability AUROC 0.735 (95% CI 0.694–0.773).

External: 252 patients, 37 events; AUROC 0.718 (95% CI 0.628–0.809), calibration slope 1.018, O/E 0.892. The retained research cutoff is 0.14232085428111443. External outcomes were not used for fitting or recalibration.

The fixed predictor set arose from earlier outcome-informed development and was retained for refitting. The separate fold-wise analysis repeats predictor selection within training data; its estimates are distinct from fixed-ten validation.

## Files

- `app.py`: Streamlit interface.
- `final_model_bundle.joblib`: final model and fitted continuous iterative/binary modal imputers.
- `requirements.txt`: numerical-library versions matching model serialization.
- `model-specification.json`: feature order, hyperparameters, calibration coefficients, model checksum.
- `validation_metrics.json`: final development and external performance.
- `verification_example.json`: illustrative input and expected prediction.

## Run

Use Python 3.12, install requirements, then run `streamlit run app.py`.

All ten web inputs are required. Operative time is surgical duration, not anesthesia duration; cryoprecipitate is in mL; ASA III=0 and IV=1. CK, LDH and neutrophils were recorded in the preoperative laboratory section, but exact sampling times could not be verified. Their prospective end-of-surgery availability is uncertain. The interface is for research, and the cutoff is not a validated clinical action threshold.

Example default inputs should display **8.9%**. No patient-level data are included in this deployment package. Model fitting and validation methods are described in the manuscript and supplement.
