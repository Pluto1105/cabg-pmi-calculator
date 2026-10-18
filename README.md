# CABG PMI research calculator

Release: 2026-10-18. Fixed ten-predictor calibrated SVM for research after isolated on-pump CABG.

## Validation

Development: 1,140 patients, 175 PMI events. Selection-inclusive mean repeat-level AUROC: 0.687 (95% CI 0.653–0.720). Fixed-predictor averaged out-of-fold AUROC: 0.738 (95% CI 0.697–0.779), potentially optimistic because of historical predictor selection.

Combined external validation: 252 patients, 40 events; AUROC 0.701 (95% CI 0.613–0.791), AUPRC 0.366, Brier score 0.123, calibration slope 0.929 (95% CI 0.526–1.394), observed-to-expected ratio 0.929. External validation remains preliminary.

The model incorporates corrected cryoprecipitate volumes; external performance uses corrected operative time. The original research threshold 0.14232085428111443 is retained unchanged, not a validated clinical action threshold. No runtime training or recalibration is performed.

## Inputs and limitations

All ten inputs are required; defaults are illustrative. ASA III=0 and ASA IV=1; other grades are not represented. Neutrophils use percent; albumin and MCHC use g/L; CK and LDH use U/L.

CK, LDH and neutrophil percentage were recorded in the preoperative laboratory section of source records, but exact sampling times could not be confirmed. Availability before the end of surgery was not verified for every measurement, limiting interpretation as a prospective end-of-surgery tool.

Operative time is recorded surgery start to end, not anesthesia duration. Cryoprecipitate is total volume in mL (0 if none). Historical study entries in units were converted using the study-specific 25 mL/unit mapping; do not assume this conversion for other institutions or products.

The outcome is biomarker-defined PMI: recorded peak cTnI at least 70 times the study upper reference limit on postoperative days 1–3, not a stand-alone myocardial infarction diagnosis. Predictions do not establish clinical treatment benefit or causal oxycodone effects.

## Deployment

Replace the five same-named files in the existing `Pluto1105/cabg-pmi-calculator` repository. Upload extracted files, not the ZIP. Preserve repository visibility and Streamlit settings. Use Python 3.12, scikit-learn 1.8.0 and entry point `app.py`. Reboot after deployment to clear the cached previous model, confirm the displayed metrics and test a prediction. Updating this package alone does not update the hosted app.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Hyperparameters, calibration coefficients, feature order and model SHA-256 are in `model-specification.json`. Complete inputs are passed in fitted feature order to the calibrated estimator, including fitted standardization. Development imputers remain in the bundle but are not used by the complete-input interface.

No patient spreadsheets are included. The fitted SVM contains training-derived support vectors; public model-file release requires appropriate data-sharing authorization. This package is not the complete study analysis archive.
