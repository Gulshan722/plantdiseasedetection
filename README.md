# Multi-Crop Plant Disease Classifier: A Domain-Gap Case Study

## TL;DR

An initial model trained on clean, lab-condition photos (PlantVillage) scored
92% validation accuracy — and only 42% macro-F1 on independent, real-world
field photos. This repo documents diagnosing that gap, closing it through
three measured iterations (**0.42 → 0.48 → 0.54 macro-F1**), and a follow-up
experiment that produced an equally important *negative* result: naively
scaling from 17 to 34 classes across more crops actually **hurt** accuracy on
the original classes, due to cross-crop class interference.

The core lesson, and the point of sharing this: **a good validation score
does not mean a deployable model.** Most of the work here is in building the
methodology to catch that before shipping, not in chasing a bigger number.

## The problem

Public plant disease datasets (PlantVillage, and similar) are collected under
lab conditions: single leaf, uniform background, controlled lighting. Real
users photograph diseased plants in the field: cluttered backgrounds, natural
light, imperfect framing. A model that only ever sees the former learns
shortcuts (background, framing, color balance) that don't transfer to the
latter — producing misleadingly high validation scores that collapse in
actual use.

## Methodology

1. **Held out an independent, distribution-different test set (PlantDoc)**
   from all training — not just a random split of the same data, since a
   random split cannot detect a domain gap that exists between *how* datasets
   were collected, only within one collection method.
2. **Measured baseline real-world performance** (0.42 macro-F1) against this
   held-out set, confirming a genuine ~50-point gap versus the 0.92 validation
   score.
3. **Iteratively added real-world/field-collected training data** from
   additional sources (PlantDoc's own train split, then a filtered subset of
   FieldPlant), re-measuring against the *same* fixed held-out set after each
   change — isolating the effect of each addition rather than conflating
   multiple changes.
4. **Verified suspicious data before trusting it**, rather than assuming
   labels were correct: caught a mistranslated French disease label in one
   source (visually inspected sample images against the claimed label) and a
   folder-naming error in another (cross-referenced against the dataset's own
   publication), correcting both before training.
5. **Scaled scope** from 3 crops (maize, potato, tomato — 17 classes) to 5
   crops (+ rice, wheat — 34 classes), and discovered the scaled model
   *degraded* on original classes despite a higher aggregate validation score
   — diagnosed as cross-crop interference (multiple visually-similar "healthy"
   classes competing in one softmax), documented rather than hidden.

## Results

| Model version | Val Macro-F1 | Real-world (PlantDoc) Macro-F1 |
|---|---|---|
| v1 (PlantVillage + PlantWild only) | 0.92 | 0.42 |
| v2 (+ PlantDoc train split) | 0.90 | 0.48 |
| v4 (+ filtered FieldPlant subset) | 0.91 | 0.54 |
| v5 (scaled to 34 classes, 5 crops) | 0.93 | 0.49 *(regression on original 17)* |

## Stack

PyTorch, EfficientNet-B0 (transfer learning), Kaggle GPU (T4), multi-source
data engineering across Kaggle/Hugging Face/Roboflow Universe, ONNX export,
FastAPI serving.

## Data sources

PlantVillage, PlantWild, PlantDoc, FieldPlant, PaddyDoctor, DhanShomadhan,
Wheat Disease Dataset (Small). See notebooks for full citations. **Raw
datasets are not redistributed in this repo** — several sources carry
non-commercial/no-derivative licenses (notably PlantWild, CC BY-NC-ND); the
notebooks fetch data programmatically from original sources instead. Trained
model weights inherit these license restrictions and are provided for
non-commercial/research use only.

## Notebooks

- `01_tier1_maize_potato_tomato.ipynb` — initial pipeline, domain-gap
  diagnosis, and the 0.42→0.48→0.54 iteration
- `02_tier2_rice_wheat_expansion.ipynb` — scope expansion to 5 crops and the
  cross-crop interference finding

## What I'd do differently / next steps

- The rice and wheat classes added in v5 were never checked against an
  independent real-world test set the way the original 17 were — a genuine
  gap in this project's methodology, not yet resolved.
- A two-stage architecture (crop classifier → per-crop disease classifier)
  is the likely fix for the cross-crop interference issue, untested here.
- Real production accuracy at this stage of data availability tops out
  around 50-55% macro-F1 on genuine field photos for several disease classes
  — not yet deployment-grade; the realistic path forward is an in-app
  feedback loop collecting real user photos over time, not further public
  dataset merging.
