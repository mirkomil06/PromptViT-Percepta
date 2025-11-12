# 📚 Literature Review Summary — Week 2 (Asilbek Tashpulatov)

**Project:** PromptViT-Percepta  
**Course:** Computer Vision (Fall 2025)  
**Team:** Percepta  
**Instructor:** Dr. I. Atadjanov & Dr. B. Kiani  
**Week:** 2 (Oct 21–27)  
**Task:** Literature review & dataset preparation  

---

## 🧠 Overview

The goal of Week 2 was to review key academic papers that define the foundation of our project — *Prompt-Tuned Vision Transformers for Explainable Fine-Grained Recognition* — and to prepare datasets for training and evaluation.

This document summarizes the main findings from the four selected references and explains their direct relevance to our implementation.

---

## 📑 Literature Review Table

| # | Paper Title | Year | Authors / Source | Main Idea | Key Findings | Relevance to Project |
|:-:|--------------|------|------------------|------------|---------------|-----------------------|
| **1** | **An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale** | 2020 | Dosovitskiy et al., Google Brain, ICLR 2021 | Introduced **Vision Transformer (ViT)** — applies a standard Transformer directly to sequences of 16×16 image patches. | ViT pre-trained on large datasets (ImageNet-21k, JFT-300M) achieved SOTA results, outperforming CNNs with less compute. | Forms the **baseline** of our project. We fine-tune ViT-B/16 as our initial benchmark model. |
| **2** | **Visual Prompt Tuning (VPT)** | 2022 | Jia et al., Cornell & Meta AI, ECCV 2022 | Introduced **prompt tuning** — small learnable embeddings prepended to Transformer inputs, while freezing most weights. | VPT-Deep outperformed full fine-tuning in 20/24 tasks using <1 % of parameters. Works well even with limited data. | Core of our **parameter-efficient adaptation** approach; used for our prompt-tuned ViT experiments. |
| **3** | **Learning Deep Features for Discriminative Localization (CAM)** | 2016 | Zhou et al., CVPR 2016 (MIT & Microsoft Research) | Proposed **Class Activation Mapping (CAM)** for CNNs to visualize important regions influencing predictions. | Enabled interpretable localization of features using global average pooling. | Foundation for **Prompt-CAM**, which visualizes attention regions in our tuned ViT models. |
| **4** | **Transformer Interpretability Beyond Attention Visualization** | 2021 | Chefer et al., CVPR 2021 (The Hebrew University) | Developed **gradient-based relevance propagation** for Transformer explainability. | Produced more faithful visual explanations than raw attention maps. | Provides theoretical base for **Prompt-CAM** and attention rollout used in our explainability module. |
| **5** | **PromptViT-Percepta Project Proposal** | 2025 | Team Percepta (CAU) | Combines ViT, Prompt-Tuning, and CAM-based visualization for explainable fine-grained classification. | Defines complete pipeline: Baseline → Prompt-Tuning → Explainability → Evaluation. | Integrates prior works into one explainable, efficient framework. |

---

## 🔍 Summary Insights

- **Vision Transformers (ViT)** proved scalable and effective for large-scale image classification.  
- **Prompt-Tuning (VPT)** provides a lightweight, adaptable way to fine-tune ViTs with <1 % extra parameters.  
- **CAM and Transformer-CAM** techniques bridge the gap between accuracy and interpretability.  
- Combining these enables an **explainable, efficient ViT framework** suitable for fine-grained recognition.

---

## 🌸 Dataset Preparation

| Dataset | Classes | Images | Verified Split | Preprocessing Notes |
|----------|----------|---------|----------------|---------------------|
| **CUB-200-2011 (Birds)** | 200 | 11 788 | ✔ Train/Val/Test verified | Resized to 224×224, checked class balance |
| **Stanford Cars** | 196 | 16 185 | ✔ Train/Test verified | Labels normalized, minor imbalance handled |
| **Oxford Flowers-102** | 102 | 8 189 | ✔ Train/Val/Test verified | Cropped and normalized, used for FGV benchmark |

All datasets are publicly available and suitable for explainable fine-grained classification.

---

## ✅ Week 2 Progress Log

- [x] Reviewed and summarized 4 foundational papers.  
- [x] Verified and preprocessed datasets (CUB, Cars, Flowers).  
- [x] Documented baseline–prompt tuning–explainability pipeline flow.  
- [x] Prepared summary for integration into `ROADMAP.md`.  

---

> *Maintained by Asilbek Tashpulatov — Team Percepta, Central Asian University (Fall 2025)*  
