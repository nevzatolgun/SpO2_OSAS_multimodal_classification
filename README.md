# SpO2_OSAS_multimodal_classification

**Zenodo DOI:** 

## Multimodal Fusion of SpO₂ Time Series and Clinical Data for Explainable Classification of Obstructive Sleep Apnea Severity

This repository contains the Python source code and reproducibility documentation associated with the manuscript:

**Multimodal Fusion of SpO₂ Time Series and Clinical Data for Explainable Classification of Obstructive Sleep Apnea Severity**

**Authors:** Aycan Baş, Nevzat Olgun, Ahmet Haşim Yurttakal, Şule Çilekar

## Study Overview

The workflow performs subject-level obstructive sleep apnea severity classification using SpO₂-derived features together with clinical and hematological variables.

The analysis includes:

- SpO₂ feature extraction from non-overlapping 60-s segments sampled at 16 Hz;
- time-domain and Welch PSD features;
- A1–A11 multimodal feature configurations;
- k-nearest neighbors, Random Forest, and Extra Trees classifiers;
- stratified 5-fold subject-level cross-validation;
- optional bootstrap and permutation analyses;
- Extra Trees feature importance and SHAP-based interpretation.

## Software Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

## How to Reproduce the Analysis

The main analysis script is:

```text
code/spo2_osas_multimodal_classification.py
```

Run:

```bash
python code/spo2_osas_multimodal_classification.py --data all_subjects_spo2_combined.csv --output OSAS_results
```

For the A4 Extra Trees bootstrap and permutation analyses:

```bash
python code/spo2_osas_multimodal_classification.py --data all_subjects_spo2_combined.csv --output OSAS_results --bootstrap-config A4 --bootstrap-model ExtraTrees --permutation-config A4 --permutation-model ExtraTrees
```

## Citation

Please cite the associated article and the archived software record when using this repository. Citation metadata are provided in `CITATION.cff`.

## License

This source code is released under the MIT License.

## About

Source code and reproducibility materials for explainable multimodal classification of obstructive sleep apnea severity using SpO₂ time series and clinical data.
