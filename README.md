# Aerial GCP Pose Estimation

A multi-task deep learning pipeline that takes an aerial drone image and 
simultaneously predicts (1) the exact pixel location of a Ground Control 
Point (GCP) marker and (2) classifies its shape (Cross / Square / L-Shaped).

## Model Weights

Download: [best_model.pt](https://drive.google.com/file/d/1vCykw1V_4o2jA-Hd7v6vWV1LE6NZGtG7/view?usp=sharing)

Usage:
```python
model = GCPHeatmapModel()
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()
```

## Repository Structure

## Architecture

A shared **ResNet34 backbone (ImageNet pretrained)** feeds two task-specific heads:

1. **Keypoint head (localization)** — a convolutional decoder upsamples the 
   backbone's feature map into a single-channel heatmap. The predicted 
   peak (via soft-argmax) gives the (x, y) location of the marker. This 
   heatmap-based approach was chosen over direct coordinate regression 
   because it preserves spatial structure through the full feature map 
   rather than collapsing it into two numbers, which empirically gives 
   sharper, more accurate localization — the standard approach in modern 
   pose estimation literature.

2. **Classification head** — the same backbone features are global-average-pooled 
   and passed through a small linear layer to predict one of three shape classes.

Sharing one backbone between both tasks keeps the model lightweight, fast 
to train, and easy to deploy — in line with the brief's preference for 
practical, robust solutions over complex architectures.

## Training Strategy

- **Optimizer:** AdamW, cosine annealing learning rate schedule
- **Input size:** images resized to 512×512; heatmap target resolution 128×128
- **Augmentation:** horizontal flip, brightness/contrast jitter (via Albumentations, 
  with keypoint-aware transforms so labels stay correctly aligned through augmentation)
- **Loss functions:**
  - Localization: MSE between predicted and target Gaussian heatmaps 
    (target heatmap is a 2D Gaussian centered on the true marker location)
  - Classification: class-weighted CrossEntropyLoss, weights inverse to 
    class frequency, to counteract severe shape class imbalance
  - Combined loss: heatmap loss scaled up before summing with classification 
    loss, since raw MSE values on a mostly-zero heatmap are much smaller in 
    magnitude than cross-entropy values
- **Train/val split:** grouped by GCP marker ID (`GroupShuffleSplit`) rather 
  than by individual image, since multiple images exist per physical marker 
  (different drone passes) — a random split would leak near-duplicate views 
  between train and val and produce misleadingly optimistic metrics

## Dataset Challenges and How They Were Handled

- **Resolution inconsistency:** the brief stated 2048×1365, but actual image 
  dimensions varied across sites. Handled by reading true dimensions per 
  image and training on normalized (0–1) coordinates, rescaled back to each 
  image's actual size at inference.
- **Severe class imbalance:** Cross markers vastly outnumber Square and 
  L-Shaped in the training set. Handled via class-weighted loss; evaluated 
  using Macro F1 (not plain accuracy) so rare classes are weighted equally.
- **Label/file mismatches:** some JSON entries had no corresponding image 
  file (and vice versa); these were identified via a full directory scan 
  and excluded from training rather than guessed at.
- **Label spelling inconsistency:** raw training labels used "L-Shape" in 
  places; the brief specifies "L-Shaped". Trained on the raw label as-is, 
  but normalized spelling to "L-Shaped" in the final `predictions.json` 
  to match the expected submission format.
- **Messy real-world file paths:** filenames included spaces, parentheses, 
  and inconsistent prefixes (production data, not a clean academic set) — 
  handled with path-safe string operations rather than assuming clean naming.

Full reasoning behind each decision is documented in `decision_log.md`.

## Evaluation

The model is validated each epoch using:
- **PCK (Percentage of Correct Keypoints)** at 10px / 25px / 50px thresholds
- **Macro F1-score** across the three shape classes

The checkpoint with the best combined PCK@25 + Macro F1 score on the 
validation set is saved as `best_model.pt`.

## How to Run Inference

1. Open `notebooks/gcp_pipeline.ipynb`
2. Update the `TEST_ROOT` path variable to point to your local `test_dataset/` folder
3. Run all cells top to bottom — this loads `best_model.pt`, runs inference 
   on every image in `test_dataset/`, and writes `predictions.json` in the 
   same nested format as the original training labels:

```json
{
  "site/survey/gcp_id/image.JPG": {
    "mark": { "x": 1234.5, "y": 678.9 },
    "verified_shape": "Cross"
  }
}
```

## Known Limitations

See `decision_log.md` for a full account of design tradeoffs, including 
one identified-but-unresolved issue around loss-scale balancing between 
the localization and classification heads, and the proposed fix.
