# 📁 Baseline ViT-B/16 — Checkpoint Folder

This folder contains the outputs generated after training the **ViT-B/16 baseline model** on the CUB-200 dataset.

Model checkpoint files are **not tracked in the Git repository** because they are large and must be generated locally.

---

## 🔍 How This Checkpoint Was Generated

This checkpoint corresponds to the **baseline model (full fine-tuning)** used in the project before applying prompt tuning methods.

To reproduce these results, you must run the baseline training script.

---

## ▶️ Step 1 — Train the Baseline Model

From the project root directory, run:
```bash
python -m src.scripts.train_cub_baseline --config src/configs/cub_baseline.yaml
```

---

## ▶️ Step 2 — Output Directory Structure

After training completes, the following folder is created automatically:
```bash
outputs/cub_baseline/
```

Inside it, you will find:
```bash
best_model.pth
logs/
events.out.tfevents.*   ← TensorBoard logs
```

- best_model.pth — the best-performing checkpoint (selected by highest validation accuracy).
- logs/ — contains TensorBoard data used to visualize accuracy and loss curves.

---

## 📊 Summary of Expected Results

With the improved training setup (augmentations, dropout, label smoothing),
the baseline ViT-B/16 model typically achieves:

- Train Accuracy: ~90–97%
- Validation Accuracy: ~35–37%
- Training Time: ~25–35 minutes on a single GPU

These metrics serve as the reference baseline for comparison with VPT-based models.

---

## 📘 Notes

- Checkpoints are large and should not be committed to GitHub.
- To visualize results, run TensorBoard:
```bash
tensorboard --logdir outputs/cub_baseline/logs --port 6006
```