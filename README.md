# MSCS-634 Lab 2: Classification Using KNN and RNN Algorithms

## Purpose
This lab explores how parameter choices affect classification performance for two
distance-based algorithms — K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) —
using the Wine dataset from scikit-learn (178 samples, 13 chemical features, 3
classes).

## Key Insights
- KNN achieved 0.9722 accuracy at k=1 and k=5, improving to a perfect 1.0000 at
  k=11, 15, and 21 — indicating the three wine classes are well-separated once
  features are standardized.
- RNN accuracy declined steadily from 0.7222 (radius=350) to 0.6667 (radius=600),
  showing that larger radii dilute predictions by including more distant,
  less-relevant neighbors.
- RNN was run on unscaled features, since the lab's suggested radius values
  (350-600) only correspond meaningfully to the original feature magnitudes
  (e.g., proline ranges up to ~1680) rather than standardized values.
- Overall, KNN substantially outperformed RNN on this dataset, suggesting the
  Wine dataset's feature space is dense and evenly distributed enough that a
  fixed neighbor count works better than a fixed distance threshold.

## Challenges and How They Were Addressed
- Initially, RNN was run on standardized (scaled) features using the lab's
  suggested radius values, which produced a flat 0.3889 accuracy across every
  radius tested. Investigation showed this matched exactly the accuracy of
  always predicting the majority class — the radius values were far too large
  relative to the scaled feature space (roughly -3 to +3 per dimension), so
  every test point's neighborhood included the entire training set. Switching
  RNN to unscaled features (while keeping KNN on scaled features, where
  standardization matters most for correct distance calculations) resolved this
  and produced a meaningful, interpretable accuracy trend.
- Handling test points with zero neighbors within a given radius required setting
  `outlier_label="most_frequent"` in RadiusNeighborsClassifier to avoid runtime
  errors on sparse regions of the feature space.

