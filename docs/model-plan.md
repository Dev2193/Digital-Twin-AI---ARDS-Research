# Model Adoption Plan

## Phase 1: phenotype and baselines

Implement an auditable ARDS phenotype outside the model code. Train L2 logistic regression and XGBoost on the same pre-index feature matrix. This phase is complete only when patient-level splitting, leakage checks, calibration, and aggregate evaluation all run reproducibly.

## Phase 2: temporal physiology

Represent the preceding 24 hours as hourly values, observation masks, and elapsed-time tensors. Compare GRU-D with XGBoost rather than assuming a neural sequence model is superior.

## Phase 3: report-level criteria

Fine-tune BioClinicalBERT on pre-index radiology reports for bilateral opacity, equivocal imaging, exclusion findings, and cardiogenic-edema evidence. Evaluate this component at the criterion level before connecting it to the ARDS predictor.

## Phase 4: late fusion

Generate out-of-fold component probabilities and combine them with calibrated logistic regression. Include a report-availability indicator so the fusion model can operate when no qualifying report exists.

## Required ablations

- structured data only;
- report text only;
- structured plus report text;
- definitional variables present versus removed;
- explicit ARDS terminology present versus masked;
- concurrent detection versus 12-hour prediction;
- strict versus permissive phenotype;
- internal temporal versus external validation.
