# MSCS-634 Lab 2: Classification Using KNN and RNN Algorithms

## Overview

This lab investigates how parameter selection affects classification performance
for two distance-based algorithms: K-Nearest Neighbors (KNN) and Radius Neighbors
(RNN, also called RNN classification). Both algorithms classify a new data point
based on the labels of nearby points in the training set, but they define
"nearby" in fundamentally different ways — KNN by a fixed count of neighbors,
RNN by a fixed distance. This lab uses the Wine dataset from scikit-learn to
compare how these two approaches behave as their key parameters (k and radius)
are varied.

## Dataset Description

**Source:** Wine dataset, built into `sklearn.datasets` (`load_wine`)

**Size:** 178 samples, 13 numeric chemical features, 3 target classes

**Features:** alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
color_intensity, hue, od280/od315_of_diluted_wines, proline

**Class distribution:**
| Class | Count |
|---|---|
| class_0 | 59 |
| class_1 | 71 |
| class_2 | 48 |

The classes show a moderate imbalance (class_1 is the largest at 71 samples,
class_2 the smallest at 48), which is worth keeping in mind when interpreting
accuracy, since a classifier with a bias toward the majority class has a slight
built-in advantage.

**Feature scale note:** `df.describe()` revealed substantial scale differences
across features — for example, alcohol ranges roughly 11-15, while magnesium
ranges roughly 70-162 and proline ranges up to approximately 1680. This disparity
is the central reason feature scaling plays such an important role in this lab
(see Methodology and Challenges below).

## Methodology

### 1. Data Preparation
The dataset was loaded, converted to a Pandas DataFrame for exploration, and
split 80/20 into training (142 samples) and test (36 samples) sets using
stratified sampling (`stratify=y`) to preserve the original class proportions
in both splits.

### 2. KNN Implementation
Features were standardized using `StandardScaler` (fit on the training set only,
to avoid data leakage) before training KNN, since distance-based algorithms are
sensitive to features on different scales — without scaling, a feature like
proline (range ~1680) would dominate the distance calculation over a feature
like hue (range ~0.5-1.7), regardless of each feature's actual relevance.

KNN was trained and evaluated at five values of k: 1, 5, 11, 15, and 21.

### 3. RNN Implementation
RNN was trained and evaluated at six radius values: 350, 400, 450, 500, 550, and
600. Critically, RNN was run on the **original, unscaled** features rather than
the standardized features used for KNN (see Challenges section for why).
`outlier_label="most_frequent"` was set to handle any test point that has zero
neighbors within the given radius.

### 4. Evaluation and Visualization
Both classifiers were evaluated using accuracy on the held-out test set. Results
were visualized as line plots (accuracy vs. k for KNN, accuracy vs. radius for
RNN) and compared side by side.

## Results

### KNN Accuracy by k
| k | Accuracy |
|---|---|
| 1 | 0.9722 |
| 5 | 0.9722 |
| 11 | 1.0000 |
| 15 | 1.0000 |
| 21 | 1.0000 |

### RNN Accuracy by Radius (on unscaled features)
| Radius | Accuracy |
|---|---|
| 350 | 0.7222 |
| 400 | 0.6944 |
| 450 | 0.6944 |
| 500 | 0.6944 |
| 550 | 0.6667 |
| 600 | 0.6667 |

## Key Insights

- **KNN performed excellently on this dataset**, reaching 0.9722 accuracy at
  k=1 and k=5, then improving to a perfect 1.0000 at k=11, 15, and 21. This
  indicates the three wine classes are well-separated in the standardized
  feature space, so even a fairly large neighborhood (k=21) still votes
  correctly for the right class.
- **RNN performed noticeably worse and followed the opposite trend**: accuracy
  started at 0.7222 for radius=350 and steadily declined to 0.6667 by
  radius=600. This makes intuitive sense — as the radius grows, each
  prediction incorporates more distant, less-relevant neighbors, diluting the
  vote and reducing accuracy.
- **KNN substantially outperformed RNN overall**, suggesting the Wine dataset's
  feature space (once standardized) is dense and evenly distributed enough that
  a fixed neighbor count is a more reliable strategy than a fixed distance
  threshold for this particular dataset.
- **Scaling had a critical, non-obvious effect on RNN specifically.** Radius
  values that work well on unscaled data (hundreds of units, matching the scale
  of features like proline) become meaningless on standardized data (where most
  values fall between roughly -3 and +3) — see Challenges below for the full
  story.

## Comparison and Discussion

Both KNN and RNN classify a new point based on its neighbors, but they select
neighbors differently. KNN always uses a fixed number of nearest neighbors (k),
regardless of how close or far those neighbors actually are. RNN instead uses
all neighbors within a fixed physical distance (radius), which means the number
of neighbors "voting" on a prediction can vary from point to point — and, in
edge cases, a point can end up with zero neighbors at all.

**When KNN might be preferable:** When the data is fairly evenly spaced/dense
after standardization, so picking a fixed number of neighbors reliably captures
the local neighborhood regardless of feature scale — as demonstrated here, where
KNN achieved near-perfect to perfect accuracy across nearly every tested k value.

**When RNN might be preferable:** When data density varies significantly across
the feature space and a fixed neighbor count would be misleading (e.g., in
sparse regions, the k nearest neighbors might actually be very far away and not
truly representative). RNN can also explicitly flag a point as an outlier when
no neighbors fall within the radius, which KNN cannot do. That said, choosing an
appropriate radius requires real care — as seen here, radius choice needs to
match the scale of the underlying features, and even a correctly-scaled radius
still requires tuning, since accuracy dropped by over 5 percentage points across
the tested range.

## Challenges and How They Were Addressed

- **RNN initially produced a flat, uninformative accuracy of 0.3889 across every
  radius value tested (350 through 600).** Investigation revealed this number
  matched exactly the accuracy of simply predicting the majority class for every
  test point. The root cause: RNN was initially run on standardized features
  (the same ones used for KNN), where most feature values fall within roughly
  -3 to +3 of the mean. A radius of 350 in that space is astronomically large —
  every training point falls within that radius of every test point, so each
  prediction was effectively an average over the entire training set, which
  collapses to always predicting the majority class regardless of the actual
  radius value.
- **The fix:** the lab's suggested radius values (350-600) only make sense
  relative to the original, unscaled feature magnitudes — for example, the
  proline feature alone ranges up to approximately 1680, so a radius in the
  hundreds is a reasonable, meaningful distance in that space. Re-running RNN on
  unscaled features (while keeping KNN on scaled features, where standardization
  genuinely matters most for correct distance-based comparisons across features)
  resolved the issue and produced the meaningful, interpretable declining trend
  reported above.
- **Handling points with no neighbors within the radius:** `outlier_label`
  was set to `"most_frequent"` in `RadiusNeighborsClassifier` to prevent runtime
  errors when a test point had no training points within the specified radius,
  which can occur especially at smaller radius values or in sparser regions of
  the feature space.
- **Balancing which features to scale and which not to:** this required
  recognizing that scaling is not a blanket rule to apply everywhere — it should
  match the assumptions built into how parameters (like a specific radius range)
  were chosen, rather than being applied uniformly across every distance-based
  algorithm in the notebook by default.

## Repository Contents
- `MSCS_634_Lab_2.ipynb` — full notebook with code, comments, and visualizations
- `README.md` — this file

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn`
3. Open `MSCS_634_Lab_2.ipynb` in Jupyter Notebook and run all cells
