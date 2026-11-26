# Model A input: whole image vs Model C crop

Produced by `eval/model_a_crop_vs_full.py` (same detector, accept rule, crop math and Model A file as the app). Accuracy is top-1 over 4 classes; 'confident' = softmax >= 0.65, the app's threshold.

sanity (stored test acc): 0.7489361702127659

### Held-out test photos (Model A test split): 235 images, Model C found the dog in 223 (95%); no-dog images use the full-frame answer in every column
| Model A input | Accuracy | Confident (>=0.65) | Accuracy when confident | lying_down | sitting | standing | walking |
|---|---|---|---|---|---|---|---|
| whole image | 75.3% | 85% | 80.5% | 53/62 | 52/62 | 29/49 | 43/62 |
| Model C crop, pad 0.15 | 68.5% | 80% | 75.4% | 55/62 | 54/62 | 35/49 | 17/62 |
| Model C crop, pad 0.35 | 70.2% | 83% | 76.3% | 54/62 | 54/62 | 31/49 | 26/62 |
| Model C crop, pad 0.6 | 73.2% | 86% | 78.7% | 53/62 | 53/62 | 30/49 | 36/62 |
| mean of whole image + crop 0.15 | 74.9% | 75% | 81.2% | 53/62 | 55/62 | 32/49 | 36/62 |

### Video clip frames (clip label applied to every frame): 111 images, Model C found the dog in 107 (96%); no-dog images use the full-frame answer in every column
| Model A input | Accuracy | Confident (>=0.65) | Accuracy when confident | lying_down | standing | walking |
|---|---|---|---|---|---|---|
| whole image | 73.9% | 99% | 73.6% | 60/60 | 0/27 | 22/24 |
| Model C crop, pad 0.15 | 62.2% | 92% | 64.7% | 60/60 | 4/27 | 5/24 |
| Model C crop, pad 0.35 | 65.8% | 92% | 66.7% | 59/60 | 1/27 | 13/24 |
| Model C crop, pad 0.6 | 71.2% | 96% | 72.0% | 60/60 | 0/27 | 19/24 |
| mean of whole image + crop 0.15 | 71.2% | 83% | 71.7% | 60/60 | 0/27 | 19/24 |
