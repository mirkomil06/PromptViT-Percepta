## 🔍 How to Generate This Checkpoint (VPT-Shallow)

The folder `outputs/cub_vpt_shallow/` is **not included in the GitHub repository** because model checkpoints are large and must be generated locally.

To obtain this checkpoint, you need to **train the VPT-Shallow model**.

### ▶️ Step 1 — Run VPT-Shallow training

From the project root:

```bash
python -m src.scripts.train_cub_vpt --config src/configs/cub_vpt_shallow.yaml
```

### ▶️ Step 2 — After training completes

A folder will be created automatically:
```bash
outputs/cub_vpt_shallow/
```
Inside it, you will find:
```bash
best_model.pth
```
This is the best-performing checkpoint saved during Week 4 (Prompt-Tuning with VPT-Shallow).