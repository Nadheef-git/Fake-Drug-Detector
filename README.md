# Counterfeit Medicine Detection — CNN vs Classical ML

## Overview
This project compares Deep Learning (CNN, Transfer Learning) and classical
Machine Learning (Random Forest with hand-engineered features) approaches
to classifying medicine packaging images as authentic or counterfeit,
using a public Kaggle dataset of 661 images (240 Fake, 421 Real).

## Key Finding: Dataset Leakage Discovery
Initial results showed a Transfer Learning CNN achieving 99.00% test accuracy —
a suspiciously high score for a visually challenging counterfeit-detection task.
Investigation revealed the dataset's Fake and Real images were collected through
different processes, producing systematically different image dimensions. A
model trained on image metadata alone (dimensions, file size — no visual content)
achieved 99.70% accuracy, nearly matching the CNN and confirming the high score
reflected a dataset artifact rather than genuine counterfeit-detection ability.

## Methodology
1. **Data validation**: the dataset's provided train/test/val split was found to
   have severe overlap (up to 100% of images duplicated across splits) and was
   discarded in favor of a custom, verified-clean stratified split.
2. **Models compared**: from-scratch CNN, Transfer Learning (MobileNetV2),
   Random Forest on hand-engineered features (color statistics, edge density,
   sharpness), and a deliberate metadata-only baseline to test for leakage.
3. **Evaluation**: held-out test set, untouched until final evaluation for
   every model.

## Results

| Model | Test Accuracy | Trustworthy? |
|---|---|---|
| From-scratch CNN | 80.81% | Partial — some run-to-run instability observed |
| Transfer Learning CNN | 99.00% | No — matches metadata-leakage baseline |
| Random Forest (image features) | 85.00% | Yes — robust to leakage-suspect features |
| Metadata-only baseline | 99.70% | N/A — deliberate leakage-detection control |

## Recommendation
Random Forest is recommended over the higher-scoring CNN, as it does not rely on
the dataset's collection-process artifact. Neither model is recommended for
production deployment without a re-collected, format-balanced dataset.

## Limitations
- Small dataset (661 images total)
- Source dataset has a known collection-process bias (documented above)
- From-scratch CNN showed instability across runs without a fixed random seed

  ## Additional Limitation: Real-World Distribution Shift

Manual testing after deployment revealed the model produces unreliable
predictions on photos taken directly with a phone camera, despite reasonable
performance on the held-out test set. Investigation showed phone camera images,
after resizing to the model's expected input size, carry a texture/sharpness
signature far outside the range present in training data (which was sourced
from stock images and screenshots, none exceeding ~1200px in original width).
This is a distribution shift issue, distinct from the dataset leakage finding
above: even a model with no leakage at all would need training data collected
under conditions matching its real deployment use case (i.e., actual phone
photos of packaging) to be reliable in practice.

## Tech Stack
Python, TensorFlow/Keras, OpenCV, scikit-learn, pandas
