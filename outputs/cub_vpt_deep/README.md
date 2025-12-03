# 📁 VPT-Deep (ViT-B/16) — Checkpoint Folder

This directory contains model outputs generated after training the **Visual Prompt Tuning – Deep (VPT-Deep)** variant on the CUB-200 dataset.

As with other output folders, the files in this directory are not tracked in the Git repository because model checkpoints are large and are produced locally during training.

---

## 🔍 How This Checkpoint Was Generated

This checkpoint corresponds to the **VPT-Deep** version of the model, where:

- The **ViT-B/16** backbone is **fully frozen**
- Deep prompt tuning injects **trainable prompts into multiple transformer layers**
- More parameters are trainable compared to VPT-Shallow
- Training remains stable, but:
- - Validation accuracy is **similar to VPT-Shallow**
- - Model tends to slightly overfit due to deeper prompt layers

VPT-Deep offers higher model capacity, but on CUB-200 it does not outperform VPT-Shallow.

---

## ▶️ Step 1 — Train the VPT-Deep Model

From the project root directory, run:

```bash 
python -m src.scripts.train_cub_vpt_deep --config src/configs/cub_vpt_deep.yaml
```

This will train the VPT-Deep configuration using:
- CUB-200 dataset
- Prompt tokens inserted at multiple transformer layers
- Frozen ViT-B/16 backbone
- Custom classification head
- TensorBoard logging

---

## ▶️ Step 2 — Output Directory Structure

After training finishes, this folder will be created:
```bash 
outputs/cub_vpt_deep/
``` 
Inside it, you will find:
```bash 
best_model.pth
logs/
events.out.tfevents.*   ← TensorBoard logs
``` 
- **best_model.pt** — best-performing checkpoint (selected using validation accuracy)
- **logs/** — TensorBoard logs containing loss/accuracy curves

---

## 📊 Summary of Expected Results

From experiment results:

| Model	| Validation Accuracy |
|-------|------------------|
| Baseline | ViT-B/16	~35–37% |
| VPT-Shallow | ~86–87% |
| VPT-Deep | ~86.3–86.7% |

Key insights:

- ✔ VPT-Deep performs **very close** to VPT-Shallow

- ✔ Higher training accuracy (often reaches ~100%)

- ⚠ Slightly more overfitting than shallow prompting

- ⚠ No significant accuracy gain despite deeper prompts

- ✔ Results align with research showing that shallow prompts often generalize better

---

## 📘 Notes

Checkpoints are large — **do not commit them to GitHub.**

To visualize training progress, launch TensorBoard:
```bash 
tensorboard --logdir outputs/cub_vpt_deep/logs --port 6006
```