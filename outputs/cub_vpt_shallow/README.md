# 📁 VPT-Shallow (ViT-B/16) — Checkpoint Folder

This directory contains model outputs generated after training the **Visual Prompt Tuning (VPT-Shallow)** model on the CUB-200 dataset.

As with baseline outputs, the files inside this folder are **not tracked in the Git repository** because model checkpoints are large and must be produced locally.

---

## 🔍 How This Checkpoint Was Generated

This checkpoint corresponds to the **VPT-Shallow** version of the model, where:

- The ViT-B/16 backbone is **fully frozen**
- Only **prompt tokens + classification head** are trainable
- Training is significantly faster and more stable than full fine-tuning
- Final accuracy greatly surpasses the baseline ViT

---

## ▶️ Step 1 — Train the VPT-Shallow Model

From the project root directory, run:

```bash
python -m src.scripts.train_cub_vpt_shallow --config src/configs/cub_vpt_shallow.yaml
```

This will automatically start training using:

- CUB-200 dataset
- Shallow prompt tuning
- TensorBoard logging
- Frozen ViT backbone
- Custom classification head

---

## ▶️ Step 2 — Output Directory Structure

After training finishes, this folder will be created:
```bash
outputs/cub_vpt_shallow/
```

Inside it, you will find:
```bash
best_model.pth
logs/
events.out.tfevents.*   ← TensorBoard logs
```

- best_model.pth — best-performing checkpoint, selected by validation accuracy.
- logs/ — training curves (loss/accuracy) for TensorBoard visualization.

---

## 📊 Summary of Expected Results

VPT-Shallow delivers a massive performance boost compared to the baseline ViT-B/16:

| Model	| Validation Accuracy |
|-------|------------------|
| Baseline | ViT-B/16	~35–37% |
| VPT-Shallow |	~86–87% |

Key benefits:

- ✔ Dramatically improved generalization

- ✔ Avoids overfitting on small datasets

- ✔ Uses far fewer trainable parameters

- ✔ Matches published VPT research results

---

## 📘 Notes

- These checkpoints are large — do not commit them to GitHub.
- To visualize training progress, launch TensorBoard:
```bash
tensorboard --logdir outputs/cub_vpt_shallow/logs --port 6006
```