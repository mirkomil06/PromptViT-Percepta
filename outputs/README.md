# 📁 Outputs Directory

This folder contains all model outputs generated during training and evaluation for the **PromptViT-Percepta** project.
It **is not included in the Git repository** because model checkpoints and TensorBoard event logs are large and must be generated locally.

Each experiment subfolder contains:
- **best_model.pth** — best checkpoint saved using validation accuracy
- **logs/** — TensorBoard event files
- Optional README files describing how the checkpoint was generated

---

## 📂 Folder Structure
```bash
outputs/
│
├── cub_baseline/
│   ├── best_model.pth
│   └── logs/
│       └── events.out.tfevents.*
│
└── cub_vpt_shallow/
│   ├── best_model.pth
│   └── logs/
│       └── events.out.tfevents.*
│
└── cub_vpt_deep/
    ├── best_model.pth
    └── logs/
        └── events.out.tfevents.*
```

---

## 🧠 Experiments Overview

### 🔵 1. Baseline ViT-B/16 (Full Fine-Tuning)

**Folder:** outputs/cub_baseline/

This experiment trains ViT-B/16 with **all 86M parameters unfrozen.**
Although the model fits the training set extremely well, it fails to generalize on CUB-200.

**Performance:**
- **Train Accuracy:** ~98–99%
- **Validation Accuracy:** ~36–37%

**Notes:**
- Suffers from strong overfitting
- Serves as a comparison baseline for VPT methods

Read more:
```bash
👉 outputs/cub_baseline/README.md
```

### 🟣 2. VPT-Shallow (Frozen ViT + Shallow Prompts)

**Folder:** outputs/cub_vpt_shallow/

This experiment uses **Visual Prompt Tuning (Shallow)**, where the ViT backbone is frozen and only prompt tokens + classification head are trained.

**Performance:**
- **Train Accuracy:** ~96–97%
- **Validation Accuracy:** ~86–87%

**Highlights:**
- +50% absolute improvement over baseline
- Extremely stable training
- ~1000× fewer trainable parameters
- Matches results from the VPT research paper

Read more:
```bash
👉 outputs/cub_vpt_shallow/README.md
```

### 🟠 3. VPT-Deep (Frozen ViT + Deep Prompts)

**Folder:** outputs/cub_vpt_deep/

This experiment inserts prompt tokens into multiple Transformer layers, increasing model capacity but keeping the backbone frozen.

**Performance:**
- **Train Accuracy:** ~99–100%
- **Validation Accuracy:** ~86.3–86.7%

**Highlights:**
- Slightly more capacity than VPT-Shallow
- Mild overfitting at deeper prompt layers
- Validation accuracy remains similar to VPT-Shallow
- Demonstrates that deeper prompting does not improve CUB-200 performance

Read more:
```bash
👉 outputs/cub_vpt_deep/README.md
```

## 📊 TensorBoard Visualization

To compare models visually, run:
```bash
tensorboard --logdir outputs --port 6006
```
This loads logs for all experiments and allows comparison of:
- Training vs. validation accuracy
- Loss curves
- Baseline vs. VPT behavior
- Training stability across epochs

## ⚠️ Important Notes
- Do not commit outputs/ to GitHub — checkpoints and logs are large.
- All checkpoints must be re-generated locally using the training scripts.
- Make sure to run training with the correct YAML config files for reproducibility.