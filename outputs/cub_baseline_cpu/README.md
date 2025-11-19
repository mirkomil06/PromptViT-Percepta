## 🔍 How to Generate This Checkpoint

The file `best_model.pth` is **not included in the GitHub repository** because model checkpoints are large and should be generated locally.

To obtain this file, you must **train the baseline model** yourself.

### ▶️ Step 1 — Run baseline training

From the project root:

```bash
python -m src.scripts.train_cub_baseline --config src/configs/cub_baseline.yaml
```

### ▶️ Step 2 — After training finishes

A folder will be created automatically:
```bash
outputs/cub_baseline_cpu/
```
Inside it, you will see:
```bash
best_model.pth
```
This is the best-performing checkpoint saved during Week 3 baseline ViT training.