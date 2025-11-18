# PromptViT-Percepta

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

Prompt-Tuned Vision Transformers (ViTs) for Explainable Fine-Grained Recognition.

This project explores prompt-tuned Vision Transformers for fine-grained image classification (birds, cars, flowers) with a focus on model **interpretability** using attention-based visual explanations.

> Course: Computer Vision (Fall 2025), Central Asian University  
> Team: Percepta

---

# 📑 Table of Contents
1. [Project Overview](#-project-overview)  
2. [Objectives](#-objectives)  
3. [Quickstart](#-quickstart)  
4. [Repository Structure](#-repository-structure)  
5. [Baseline Results (Week 3)](#-baseline-results-week-3)  
6. [Datasets Used](#-datasets-used)  
7. [Methodology](#-methodology)  
8. [Project Roadmap](#-project-roadmap)  
9. [Team](#-team-percepta)  
10. [Tech Stack](#-tech-stack)  
11. [Ethics & Compliance](#-ethics--compliance)  
12. [Expected Outcomes](#-expected-outcomes)  
13. [Experiments & Evaluation](#-experiments--evaluation)  
14. [References](#-references)  
15. [License](#-license)  
16. [Repository Link](#-repository-link)

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
source .venv/bin/activate      # Windows: .venv\Scripts\activate
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

### 5. Train the baseline model (CUB-200)
```bash
python -m src.scripts.train_cub_baseline --config src/configs/cub_baseline.yaml
```

### 6. Run inference on a single image
```bash
python -m src.scripts.infer_cub_baseline --image "data/cub200/CUB_200_2011/images/...jpg"
```
This will output the predicted class and confidence score.

---

## 📂 Repository Structure

```bash
PromptViT-Percepta/
│
├── README.md
├── ROADMAP.md
├── requirements.txt
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
│    └── cub_baseline_cpu.txt
│
├── outputs/
│    └── cub_baseline_cpu/
│        └── best_model.pth
│
└── src/
    ├── configs/
    │    └── cub_baseline.yaml
    ├── datasets/
    │    └── cub200.py
    ├── models/
    │    └── vit_baseline.py
    ├── training/
    │    └── trainer_baseline.py
    └── scripts/
         ├── train_cub_baseline.py
         └── infer_cub_baseline.py
```

---

## 📊 Baseline Results (Week 3)

**Model:** ViT-B/16 (timm, pretrained on ImageNet-21k → 1k)  
**Training Device:** CPU  
**Epochs:** 5  
**Dataset:** CUB-200-2011  

| Epoch | Validation Accuracy |
|-------|----------------------|
| 1 | 1.40% |
| 2 | 3.94% |
| 3 | 6.83% |
| 4 | 10.10% |
| 5 | **13.62%** |

**Best Validation Accuracy:** 13.62%  
**Checkpoint Saved:** outputs/cub_baseline_cpu/best_model.pth

---

### 📝 Notes
- Low accuracy is expected because:
  - training was done **only for 5 epochs**
  - training was on **CPU**
  - ViT-B/16 is a **large model (~86M params)**  
- With more epochs + GPU, accuracy will improve significantly.

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
*Planned for Weeks 4–5.*

We integrate **Visual Prompt Tuning (VPT)** — a parameter-efficient alternative to full fine-tuning:

- The ViT backbone is **frozen**  
- Learnable **prompt tokens** are prepended to transformer inputs  
- Trainable parameters reduce from ~86M → **<1%**  
- Expected benefits:  
  - Smaller memory footprint  
  - Faster training  
  - Comparable accuracy to full fine-tuning  
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

| Week | Dates | Milestone | Status |
|------|--------|-----------|---------|
| **Week 1** | Oct 14–21 | Repo setup, topic confirmation | ✅ Completed |
| **Week 2** | Oct 21–27 | Literature review + dataset preparation | ✅ Completed |
| **Week 3** | Oct 28–Nov 3 | Baseline ViT-B/16 training on CUB-200 | ✅ Completed |
| **Week 4** | Nov 4–10 | Prompt-tuning implementation (VPT-Deep / VPT-Shallow) | ⏳ In progress |
| **Week 5** | Nov 11–17 | Explainability module (Prompt-CAM, attention rollout) | ⏳ Upcoming |
| **Week 6** | Nov 18–24 | Evaluation (Accuracy, F1, Pointing Game) | ⏳ Upcoming |
| **Week 7** | Nov 25–Dec 1 | Report writing & presentation slides | ⏳ Upcoming |
| **Week 8** | Dec 2–8 | Final cleanup & project presentation | ⏳ Upcoming |

🗂️ [**ROADMAP.md**](./ROADMAP.md) file will include weekly progress updates and issue tracking.

### 🔄 Progress Summary

- Week 1–3: **Core pipeline completed**  
- Week 4: **Prompt-tuning implementation ongoing**  
- Future weeks: explainability → evaluation → report → presentation

---

## 👥 Team Percepta

| Name | Role | Email |
|------|------|-------|
| **Mirkomil Mirzohidov** | Model architecture & repository management | 221408@centralasian.uz |
| **Muhammad Saidahmetov** | Experiments, evaluation metrics, prompt-tuning | 220838@centralasian.uz |
| **Asilbek Tashpulatov** | Dataset preparation, documentation & report writing | 221443@centralasian.uz |

We work together to build an explainable and efficient Vision Transformer–based system for fine-grained classification.

---

## 🛠️ Tech Stack

- **Python 3.13**
- **PyTorch** — deep learning framework  
- **timm** — Vision Transformer (ViT-B/16) implementation  
- **Pillow / torchvision** — image loading & preprocessing  
- **Matplotlib** — visualizations and plots  
- **tqdm** — progress bars for training  
- **Visual Studio Code** — main development environment  

---

## ⚖️ Ethics & Compliance

- All datasets used in this project (**CUB-200-2011**, **Stanford Cars**, **Oxford Flowers-102**) are publicly available and intended for academic research.
- The project does **not collect**, **store**, or **process** any personal or sensitive information.
- All model outputs and visualizations are used strictly for educational and research purposes.
- The code and methods follow standard practices in the machine learning and computer vision community.
- All referenced papers and datasets are cited and credited to their original authors.

---

## 📈 Expected Outcomes

By the end of the project, we aim to deliver:

- A fully trained **baseline ViT-B/16** model on fine-grained datasets  
- A **prompt-tuned Vision Transformer** (VPT-Deep / VPT-Shallow) with <1% trainable parameters  
- Explainability visualizations using **Prompt-CAM** and **attention rollout**  
- Evaluation metrics:
  - Accuracy  
  - F1-score  
  - Pointing Game (interpretability metric)  
- A clean and reproducible codebase with clear configuration files  
- A final **PDF report** and a **presentation** summarizing the project workflow and results  

---

## 🔬 Experiments & Evaluation

Our experiments are designed to compare three major components:

1. **Baseline ViT-B/16 fine-tuning**  
2. **Prompt-Tuned ViT (VPT-Deep / VPT-Shallow)**  
3. **Explainability quality (Prompt-CAM & attention rollout)**

### **1️⃣ Experiment Setups**

| Experiment | Description | Status |
|-----------|-------------|--------|
| **E1 — Baseline ViT Training** | Full fine-tuning of ViT-B/16 on CUB-200 | ✅ Completed |
| **E2 — Prompt-Tuning (VPT-Deep)** | Add deep prompt tokens, freeze backbone | ⏳ In progress |
| **E3 — Prompt-Tuning (VPT-Shallow)** | Add prompts only to the first layer | ⏳ Planned |
| **E4 — Explainability Evaluation** | Generate Prompt-CAM & attention rollout | ⏳ Planned |
| **E5 — Pointing Game Metric** | Evaluate interpretability quality | ⏳ Planned |
| **E6 — Cross-dataset Generalization** | Evaluate CUB-trained model on Cars/Flowers | ⏳ Planned |

### **2️⃣ Evaluation Metrics**

We evaluate models on both **accuracy** and **interpretability**:

#### **Classification Metrics**
- **Top-1 Accuracy**
- **F1-score**
- **Confusion Matrix**

#### **Interpretability Metrics**
- **Pointing Game** (localization accuracy)
- **CAM heatmap quality** (qualitative)
- **Attention Rollout visualization**

### **3️⃣ Comparison Strategy**

We will compare:

| Model | Trainable Params | Expected Behavior |
|-------|------------------|------------------|
| **ViT-B/16 (full fine-tuning)** | ~86M | Highest accuracy, slow training |
| **VPT-Shallow** | <1% params | Lightweight, faster, stable |
| **VPT-Deep** | <1% params | Best for complex tasks |
| **No Prompts (frozen ViT)** | ~0 trainable | Weak baseline |

This comparison will show the **benefit of prompt tuning** vs full fine-tuning.

### **4️⃣ Datasets for Evaluation**

- **CUB-200-2011** → main dataset for all experiments  
- **Stanford Cars** → cross-dataset generalization test  
- **Oxford Flowers-102** → interpretability test (CAMs look very clear)

### **5️⃣ Deliverables per Experiment**

Each experiment will produce:

- Training logs  
- Validation curves  
- Best checkpoint  
- Visual explainability maps  
- Metric comparison tables  

This ensures full reproducibility and clarity in final reporting.

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