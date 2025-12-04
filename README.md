# PromptViT-Percepta

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

**Prompt-Tuned Vision Transformers (ViTs) for Explainable Fine-Grained Recognition**

This project investigates **parameter-efficient adaptation** of Vision Transformers using **Visual Prompt Tuning (VPT)** for fine-grained classification tasks (e.g., bird species, car models, flower categories). Alongside high accuracy, the project emphasizes **interpretability**, applying techniques such as Prompt-CAM and attention rollout for meaningful visual explanations.

> Course: Computer Vision (Fall 2025), Central Asian University  
> Team: Percepta

---

# 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Objectives](#-objectives)
3. [Quickstart](#-quickstart)
4. [Repository Structure](#-repository-structure)
5. [Baseline Results (Week 3)](#-baseline-results-week-3)
6. [Prompt-Tuning Results (Week 4 — VPT-Shallow)](#-prompt-tuning-results-week-4--vpt-shallow)
7. [Prompt-Tuning Results (Week 4 — VPT-Deep)](#-prompt-tuning-results-week-4--vpt-deep)
8. [Datasets Used](#-datasets-used)
9. [Methodology](#-methodology)
10. [Project Roadmap](#-project-roadmap)
11. [Team](#-team)
12. [Tech Stack](#-tech-stack)
13. [Ethics & Compliance](#-ethics--compliance)
14. [Expected Outcomes](#-expected-outcomes)
15. [Experiments & Evaluation](#-experiments--evaluation)
16. [References](#-references)
17. [License](#-license)
19. [Repository Link](#-repository-link)

---


## 🧠 Project Overview

Fine-grained image classification deals with categories that look visually similar (e.g., bird species, car models, flower types). Traditional models achieve good accuracy but often lack **explainability** — it’s difficult to understand *why* the model makes a specific prediction.

This project combines:

- **Vision Transformers (ViT-B/16)** — as the baseline model  
- **Visual Prompt Tuning (VPT)** — lightweight parameter-efficient adaptation  
- **Prompt-CAM & attention rollout** — to visualize what the model focuses on  
- **Evaluation metrics** — accuracy, F1-score, and pointing-game interpretability score  

The goal is to build a system that is **both accurate and explainable**.

---

## 🎯 Objectives

- Build a baseline ViT-B/16 model for fine-grained recognition  
- Implement prompt-tuning (VPT-Deep / VPT-Shallow) to reduce trainable parameters  
- Add explainability methods (Prompt-CAM, attention rollout)  
- Evaluate trade-offs between full fine-tuning and prompt-tuning  
- Produce visual explanations that show why the model predicts a specific class  

---

## 🚀 Quickstart

Follow these steps to set up and run the project.

### 1. Clone the repository
```bash
git clone https://github.com/mirkomil06/PromptViT-Percepta.git
cd PromptViT-Percepta
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
Download datasets from Kaggle and place them under data/ in this structure:
```bash
data/
├── cub200/
│   └── CUB_200_2011/...
├── cars/
│   ├── car_devkit/
│   ├── cars_train/
│   └── cars_test/
└── flowers/
    └── dataset/
        ├── train/
        ├── valid/
        └── test/
```
*(The Cars/Flowers datasets are not yet in use, but the structure is prepared for future expansion.)*

### 5. Train the Baseline ViT-B/16 Model (Week 3)
```bash
python -m src.scripts.train_cub_baseline --config src/configs/cub_baseline.yaml
```
This trains a fully fine-tuned ViT-B/16 model. CUDA will be used automatically if available.

### 6. Train the VPT-Shallow Model (Week 4)
```bash
python -m src.scripts.train_cub_vpt_shallow --config src/configs/cub_vpt_shallow.yaml
```
This trains the Visual Prompt Tuning (VPT-Shallow) model with:
- Frozen ViT backbone
- Trainable prompts
- Trainable classification head

### 7. Train the VPT-Deep Model (Week 4)
```bash
python -m src.scripts.train_cub_vpt_deep --config src/configs/cub_vpt_deep.yaml
```
This trains the Visual Prompt Tuning (VPT-Deep) model with:
- Frozen ViT backbone
- Deep trainable prompts injected into multiple Transformer layers
- Trainable classification head
- Higher prompt capacity compared to VPT-Shallow

### 8. Run Inference (Baseline)
```bash
python -m src.scripts.infer_cub_baseline --image "data/cub200/CUB_200_2011/images/<your-image>.jpg"
```
Outputs:
- Predicted class
- Confidence score

### 9. (Optional) VPT Inference
A VPT inference script (*infer_cub_vpt.py*) will be added in Week 5.

### 10. Visualize Training with TensorBoard
```bash
tensorboard --logdir outputs --port 6006
```

---

## 📂 Repository Structure

```bash
PromptViT-Percepta/
│
├── README.md
├── ROADMAP.md
├── requirements.txt
├── .gitignore
├── CV25_PromptViT_Percepta_Percepta.pdf
│
├── References/
│    ├── AN IMAGE IS WORTH 16X16 WORDS.pdf
│    ├── Visual Prompt Tuning.pdf
│    ├── Learning Deep Features for Discriminative Localization.pdf
│    ├── Transformer Interpretability.pdf
│    └── Literature_Review_Summary.md
│
├── data/ # datasets (not included due to size)
│
├── results/
│    ├── cub_baseline.txt
│    ├── cub_vpt_shallow.txt
│    ├── cub_vpt_deep.txt
│    └── comparison_baseline_vs_vpt_shallow_vs_vpt_deep.txt
│
├── outputs/
│    ├── README.md
│    ├── cub_baseline/
│    │   ├── logs/
│    │   ├── README.md
│    │   └── best_model.pth
│    ├── cub_vpt_shallow/
│    │   ├── logs/
│    │   ├── README.md
│    │   └── best_model.pth
│    └── cub_vpt_deep/
│        ├── logs/
│        ├── README.md
│        └── best_model.pth
│
└── src/
    ├── configs/
    │    ├── cub_baseline.yaml
    │    ├── cub_vpt_shallow.yaml
    │    └── cub_vpt_deep.yaml
    ├── datasets/
    │    └── cub200.py
    ├── models/
    │    ├── vit_baseline.py
    │    ├── vit_vpt_shallow.py 
    │    └── vit_vpt_deep.py
    ├── training/
    │    └── trainer_baseline.py
    └── scripts/
         ├── train_cub_baseline.py
         ├── infer_cub_baseline.py
         ├── train_cub_vpt_shallow.py 
         └── train_cub_vpt_deep.py
```

---

## 📊 Baseline Results (Week 3)
**Model:** ViT-B/16 (timm, pretrained on ImageNet-21k → 1k)  
**Training Device:** GPU (CUDA)  
**Epochs:** 60  
**Dataset:** CUB-200-2011  

### 📌 Performance Summary

| Metric | Value |
|--------|--------|
| Train Accuracy | ~98–99% |
| Validation Accuracy | **~36–37%** |
| Best Validation Accuracy | **~36.7%** |
| Trainable Parameters | ~86M |

**Checkpoint Saved:**  
`outputs/cub_baseline/best_model.pth`

---

### 📝 Notes

- The baseline shows **severe overfitting**:  
  - Training accuracy reaches ~99%  
  - Validation accuracy remains very low  
- Full fine-tuning of ViT-B/16 is **not suitable** for small fine-grained datasets like CUB-200.  
- Even with long GPU training, validation accuracy does **not exceed ~37%**.  
- Serves as an important comparison point for VPT methods.

---

### ❗ Why the Baseline Performs Poorly

- ViT-B/16 has **~86 million trainable parameters**  
- CUB-200 dataset is **too small** for stable full fine-tuning  
- Fine-grained recognition requires subtle distinctions (colors, beaks, wings, patterns)  
- The model memorizes training images but **fails to generalize**

---

## 🌱 Prompt-Tuning Results (Week 4 — VPT-Shallow)

**Method:** Visual Prompt Tuning (VPT-Shallow)  
**Backbone:** ViT-B/16 (frozen)  
**Trainable Parts:** 10 prompt tokens + classification head  
**Training Device:** GPU (CUDA)  
**Epochs:** 30  
**Dataset:** CUB-200-2011  

### 📌 Performance Summary

| Model                     | Trainable Parameters | Best Val Accuracy |
|---------------------------|----------------------|-------------------|
| Baseline ViT-B/16        | ~86M                 | ~36–37%           |
| **VPT-Shallow (10 prompts)** | **~10K**              | **~86–87%**        |

### 🧾 Notes

- The ViT-B/16 backbone is **fully frozen** during training, ensuring strong generalization.
- Only the **prompt embeddings** and the **classification head** are optimized.
- Training is **highly stable** with no signs of overfitting.
- Validation accuracy surpasses the baseline by **+50 percentage points**.
- Results closely match the accuracy reported in the official VPT research paper.

### ✔ Why VPT-Shallow Works So Well

- Prompt tokens function as **task-specific adapters**, guiding the frozen transformer.
- Far fewer parameters (~10K) reduce the risk of overfitting.
- ViT pretrained features remain intact, which is crucial for small datasets like CUB-200.
- Perfect fit for **fine-grained recognition** tasks.

---

### 🟣 Conclusion

VPT-Shallow achieves **state-of-the-art performance** on CUB-200 while training only a tiny fraction of parameters.

Compared to full fine-tuning:
- **+50% improvement** in validation accuracy  
- **~1000× fewer trainable parameters**  
- **More stable** learning curves  
- **Better generalization**  

**VPT-Shallow is the recommended method for this project.**

---

## 🔥 Prompt-Tuning Results (Week 4 — VPT-Deep)

**Method:** Visual Prompt Tuning (VPT-Deep)  
**Backbone:** ViT-B/16 (frozen)  
**Trainable Parts:** Deep prompt tokens (multiple layers) + classification head  
**Training Device:** GPU (CUDA)  
**Epochs:** 30  
**Dataset:** CUB-200-2011  

### 📌 Performance Summary

| Model                     | Trainable Parameters | Best Val Accuracy |
|---------------------------|----------------------|-------------------|
| Baseline ViT-B/16        | ~86M                 | ~36–37%           |
| VPT-Shallow (10 prompts) | ~10K                 | ~86–87%           |
| **VPT-Deep (multi-layer)** | **~200K**             | **~86.3–86.7%**     |

### 🧾 Notes

- Deep prompts are injected into **multiple transformer layers**, giving the model higher capacity.  
- The ViT backbone remains **fully frozen**, preserving pretrained representations.  
- Training accuracy reaches **~100%**, showing mild overfitting.  
- Validation accuracy remains **on par with VPT-Shallow**, not higher.  
- Behavior matches the original VPT paper:  
  **shallow prompting often generalizes better than deep prompting on small datasets.**

### ✔ Why VPT-Deep Behaves This Way

- More prompt layers → more parameters → higher capacity → **more overfitting risk**.  
- CUB-200 is small, making deeper prompting **unnecessary**.  
- VPT-Shallow provides a natural regularization effect due to fewer parameters.  
- Deep prompting modifies intermediate transformer layers more aggressively, which can hurt generalization.

---

### 🟠 Conclusion

VPT-Deep is a strong method, but on CUB-200:

- It **does not outperform VPT-Shallow**  
- Validation accuracy is nearly identical (~86–87%)  
- Training curves are **slightly noisier**  
- Parameter count is significantly higher (~200K vs ~10K)

**VPT-Shallow remains the recommended approach for this project.**

---

## 📚 Datasets Used

We use three widely adopted fine-grained visual classification datasets:

| Dataset | Classes | Images | Description |
|---------|---------|---------|-------------|
| **[CUB-200-2011](https://www.kaggle.com/datasets/wenewone/cub2002011)** | 200 | 11,788 | Bird species with highly subtle inter-class variations; main dataset for baseline training |
| **[Stanford Cars](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)** | 196 | 16,185 | Fine-grained car model classification (make, year, style) |
| **[Oxford Flowers-102](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)** | 102 | 8,189 | Flower species with strong visual similarity between categories |

These datasets are ideal for testing both **model accuracy** and **explainability**, since many classes are visually difficult to distinguish.

---

## 🛠️ Methodology

Our pipeline consists of three major stages:

### **1. Baseline Vision Transformer (ViT-B/16)**  
- We fine-tune a pretrained **ViT-B/16** model using the `timm` library.  
- Only the classification head is replaced (1000 → 200 classes).  
- This baseline is used as the reference point for later prompt-tuned models.  
- Evaluation: **Accuracy** and **F1-score**.

### **2. Prompt-Tuning (VPT-Deep / VPT-Shallow)**  
VPT-Shallow was successfully completed in **Week 4**, demonstrating a major accuracy improvement (+66.8% over baseline).  
Implementation of VPT-Deep is planned for **Weeks 5–6**.

We integrate **Visual Prompt Tuning (VPT)** — a parameter-efficient alternative to full fine-tuning:

- The ViT backbone is **fully frozen**  
- Learnable **prompt tokens** are prepended to the transformer input  
- Trainable parameters reduce from ~86M → **<1%**  
- Benefits of VPT include:  
  - Much smaller memory footprint  
  - Faster and more stable training  
  - Accuracy competitive with (or exceeding) full fine-tuning  
  - Better generalization on limited data

### **3. Explainability (Prompt-CAM & Attention Rollout)**  
*Planned for Weeks 5–6.*

To visualize what the model attends to:

- **Prompt-CAM** — class activation mapping adapted for prompt-tuned ViTs  
- **Attention rollout** — averages attention across layers  
- **Pointing Game metric** — evaluates how well the heatmap highlights the correct object  

Goal: Provide **faithful, human-understandable** explanations of model predictions.

---

## 📅 Project Roadmap

| Week | Dates       | Milestone                                               | Status        |
|------|-------------|----------------------------------------------------------|---------------|
| **Week 1** | Oct 14–21 | Repository setup, project definition                     | ✅ Completed   |
| **Week 2** | Oct 21–27 | Literature review + dataset preparation                 | ✅ Completed   |
| **Week 3** | Oct 28–Nov 3 | Baseline ViT-B/16 (full fine-tuning) on CUB-200         | ✅ Completed   |
| **Week 4** | Nov 4–10 | Visual Prompt Tuning (VPT-Shallow & VPT-Deep) implemented | ✅ Completed   |
| **Week 5** | Nov 11–17 | Explainability module (Prompt-CAM, Attention Rollout)    | ⏳ In progress |
| **Week 6** | Nov 18–24 | Evaluation metrics (Accuracy, F1, Pointing Game)         | ⏳ Upcoming    |
| **Week 7** | Nov 25–Dec 1 | Report writing & slide preparation                      | ⏳ Upcoming    |
| **Week 8** | Dec 2–8  | Final polishing & project presentation                     | ⏳ Upcoming    |

🗂️ See **[ROADMAP.md](./ROADMAP.md)** for weekly updates, tasks, and issue tracking.

---

### 🔄 Progress Summary

- **Weeks 1–3:**  
  ✔ Core pipeline complete (dataset loader, baseline ViT, training scripts)

- **Week 4:**  
  ✔ Visual Prompt Tuning completed  
  ✔ **VPT-Shallow:** ~86–87% validation accuracy  
  ✔ **VPT-Deep:** ~86.3–86.7% validation accuracy

- **Week 5:**  
  🔄 Currently implementing **Prompt-CAM + Attention Rollout**  
  🔄 Preparing the explainability module for CUB-200 examples

- **Next Steps:**  
  → Model evaluation (Week 6)  
  → Final report and presentation (Weeks 7–8)

---

## 👥 Team

| Name | Role | Email |
|------|------|-------|
| **Mirkomil Mirzohidov** | Model architecture & repository management | 221408@centralasian.uz |
| **Muhammad Saidahmetov** | Experiments, evaluation metrics, prompt-tuning | 220838@centralasian.uz |
| **Asilbek Tashpulatov** | Dataset preparation, documentation & report writing | 221443@centralasian.uz |

We work together to build an explainable and efficient Vision Transformer–based system for fine-grained classification.

---

## 🛠️ Tech Stack

### 🔧 Core Languages & Frameworks
- **Python 3.11** — main development language  
- **PyTorch** — deep learning framework (training, inference, CUDA acceleration)  
- **timm** — Vision Transformer (ViT-B/16) backbone + pretrained weights  

### 🖼️ Data & Image Processing
- **torchvision** — datasets, transforms, augmentation  
- **Pillow (PIL)** — image loading/format support  

### 📊 Visualization & Logging
- **TensorBoard** — training curves (loss/accuracy)  
- **Matplotlib** — plots, visualizations, explainability figures  

### ⚙️ Utilities & Training Tools
- **tqdm** — progress bars  
- **PyYAML** — configuration files (`*.yaml`)  
- **NumPy** — numeric operations  
- **scikit-learn** — metrics (accuracy, F1, confusion matrix)  

### 🧑‍💻 Development Environment
- **Visual Studio Code** — IDE  
- **Git / GitHub** — version control & repository hosting  

---

## ⚖️ Ethics & Compliance

- All datasets used in this project (**CUB-200-2011**, **Stanford Cars**, **Oxford Flowers-102**) are **public, academic datasets** intended strictly for research and non-commercial use.
- The project does **not collect**, **store**, or **process** any personal, sensitive, or user-identifiable information.
- All model outputs, visualizations, and explainability analyses are produced solely for **educational and research** purposes.
- The training pipeline, evaluation workflow, and experimental methodology follow widely accepted standards in the **machine learning** and **computer vision** community.
- All external resources — datasets, papers, and pretrained models — are **properly credited and attributed** to their original authors in accordance with academic best practices.
- The project does not aim to deploy models in real-world decision-making systems; all work remains in the scope of **responsible academic experimentation**.

---

## 📈 Expected Outcomes

By the end of this project, the following deliverables will be completed:

### 🧠 Models
- A fully trained **baseline ViT-B/16** model on fine-grained classification tasks.
- Two prompt-tuned Vision Transformer variants:
  - **VPT-Shallow** (best generalization, ~10K trainable parameters)
  - **VPT-Deep** (multi-layer prompting, ~200K trainable parameters)
- All models trained with reproducible **YAML configurations**.

### 🔍 Explainability
- Visual explanation methods integrated into the pipeline:
  - **Prompt-CAM** (prompt-aware class activation maps)
  - **Attention Rollout** (transformer interpretability)
- Side-by-side visual comparisons across Baseline / VPT-Shallow / VPT-Deep.

### 📊 Evaluation
- Quantitative evaluation on CUB-200:
  - **Accuracy**
  - **F1-score**
  - **Pointing Game** (spatial interpretability metric)
- Qualitative evaluation using visual attention maps.

### 🛠️ Codebase & Reproducibility
- A clean, modular, research-grade codebase:
  - Dataset loaders
  - ViT baseline implementation
  - VPT modules
  - Explainability scripts
  - Training and inference pipelines
- Fully reproducible experiments via configuration files (`*.yaml`) and TensorBoard logs.

### 📄 Final Deliverables
- A complete **PDF research report** detailing:
  - Motivation
  - Methodology
  - Experiments
  - Results
  - Limitations & future work
- A polished **presentation deck** summarizing the entire project workflow and findings.

---

## 🔬 Experiments & Evaluation

The experimental pipeline is designed to compare three major components:

1. **Baseline ViT-B/16 fine-tuning**
2. **Prompt-Tuned ViT models (VPT-Shallow & VPT-Deep)**
3. **Explainability quality using Prompt-CAM and Attention Rollout**

---

## 1️⃣ Experiment Setups

| Experiment | Description | Status |
|-----------|-------------|--------|
| **E1 — Baseline ViT Training** | Full fine-tuning of ViT-B/16 on CUB-200 | ✅ Completed |
| **E2 — Prompt-Tuning (VPT-Shallow)** | Insert shallow prompt tokens (first layer only) | ✅ Completed |
| **E3 — Prompt-Tuning (VPT-Deep)** | Insert deep prompts across multiple transformer layers | ✅ Completed |
| **E4 — Explainability Module** | Generate Prompt-CAM & Attention Rollout | ⏳ In progress |
| **E5 — Pointing Game Metric** | Evaluate interpretability quantitatively | ⏳ Planned |
| **E6 — Cross-Dataset Evaluation** | Test generalization on Cars / Flowers | ⏳ Planned |

---

## 2️⃣ Evaluation Metrics

We evaluate models across **classification quality** and **interpretability quality**.

### 🔵 Classification Metrics
- **Top-1 Accuracy**
- **F1-score**
- **Confusion Matrix**
- **Train vs. Validation Curves**

### 🔴 Interpretability Metrics
- **Pointing Game** (localization accuracy using heatmaps)
- **Prompt-CAM quality** (qualitative)
- **Attention Rollout maps**
- **Comparison of attention regions across models**

---

## 3️⃣ Comparison Strategy

Our goal is to understand how prompt tuning improves both accuracy and interpretability.

| Model | Trainable Parameters | Expected Behavior |
|-------|----------------------|------------------|
| **ViT-B/16 (full fine-tuning)** | ~86M | Overfits on small datasets; moderate validation accuracy |
| **VPT-Shallow** | ~10K | Best generalization on CUB-200; ~86–87% accuracy |
| **VPT-Deep** | ~200K | Similar accuracy to VPT-Shallow; slightly higher overfitting risk |
| **Frozen ViT (no prompts)** | ~0 | Very weak baseline (for sanity checks) |

The comparison highlights how **prompt tuning outperforms full fine-tuning** with **1000× fewer parameters**.

---

## 4️⃣ Datasets for Evaluation

- **CUB-200-2011** — primary fine-grained dataset (birds)  
- **Stanford Cars** — cross-dataset generalization test  
- **Oxford Flowers-102** — ideal for explainability visualization (CAMs look very clean)

---

## 5️⃣ Deliverables per Experiment

Each experiment will generate:

- ✔ Training & validation logs (TensorBoard)  
- ✔ Best model checkpoint (`best_model.pth`)  
- ✔ Accuracy/F1 metrics  
- ✔ Confusion matrix  
- ✔ Prompt-CAM heatmaps  
- ✔ Attention rollout maps  
- ✔ Summary tables comparing Baseline vs. VPT models  

This ensures full reproducibility and clear documentation for the final report.

---

## 🧩 References  

- Chefer H., Gur S., Wolf L. *Transformer interpretability beyond attention visualization.*  
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 782–791.  

- Dosovitskiy A. *An image is worth 16x16 words: Transformers for image recognition at scale.*  
  arXiv preprint arXiv:2010.11929, 2020.  

- Jia M. et al. *Visual prompt tuning.*  
  In *European Conference on Computer Vision (ECCV)*. Cham: Springer Nature Switzerland, 2022, pp. 709–727.  

- Zhou B. et al. *Learning deep features for discriminative localization.*  
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 2921–2929.  

---

## 📜 License  
This project is conducted as part of the **Central Asian University — Computer Vision (Fall 2025)** course under academic fair use for research and educational purposes.

---

## 🌐 Repository Link  
[https://github.com/mirkomil06/PromptViT-Percepta](https://github.com/mirkomil06/PromptViT-Percepta)