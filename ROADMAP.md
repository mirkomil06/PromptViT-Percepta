# 🗂️ ROADMAP — PromptViT-Percepta  

**Project:** Prompt-Tuned ViTs for Explainable Fine-Grained Recognition  
**Team:** Percepta  
**Course:** Computer Vision (Fall 2024)  
**Instructor:** Dr. I. Atadjanov & Dr. B. Kiani  

---

## 📅 Week-by-Week Milestones  

| Week | Dates | Milestone / Task | Deliverable | Owner(s) |
|------|--------|------------------|--------------|-----------|
| **W1** | Oct 14–21 | Team formation, topic confirmation, repo setup | GitHub repo with README & ROADMAP | All |
| **W2** | Oct 21–27 | Literature review + dataset preparation | Summary of 4–6 core papers; datasets verified (CUB/Cars/Flowers) | Asilbek |
| **W3** | Oct 28–Nov 3 | Baseline ViT-B/16 (full fine-tuning) | 60-epoch GPU training + logs + checkpoint | Mirkomil |
| **W4** | Nov 4–10 | Prompt-Tuning (VPT-Shallow & VPT-Deep) | Training scripts + configs + checkpoints | Muhammad & Mirkomil |
| **W5** | Nov 11–17 | Explainability (Prompt-CAM + Attention Rollout) | Heatmaps & analysis | Mirkomil |
| **W6** | Nov 18–24 | Evaluation & comparison | Accuracy, F1, Pointing Game, comparison tables | All |
| **W7** | Nov 25–Dec 1 | Report writing & slides | Final PDF + presentation | Asilbek |
| **W8** | Dec 2–8 | Final cleanup & project presentation | Code cleanup, visuals, final rehearsal | All |

---

## ✅ Weekly Progress Log  

Below is the detailed progress log for each week of development.

### **Week 1 (Oct 14–21)**
- [x] Formed Team Percepta  
- [x] Created GitHub repository  
- [x] Added README.md, ROADMAP.md, and project skeleton  

---

### **Week 2 (Oct 21–27)**
- [x] Reviewed 4 major papers:
  - ViT (2020)
  - Visual Prompt Tuning (2022)
  - CAM (2016)
  - Transformer Interpretability (2021)
- [x] Wrote literature review summary  
- [x] Prepared CUB-200, Stanford Cars, and Flowers-102 dataset structure  
- [x] Documented dataset sources + Kaggle links  
- [x] Finalized preprocessing plan  

📚 **Literature Review Summary**

| # | Paper | Year | Contribution |
|:-:|--------|------|---------------|
| 1 | Dosovitskiy et al., *ViT* | 2020 | Introduced Vision Transformer |
| 2 | Jia et al., *VPT* | 2022 | Efficient prompt-tuning for ViTs |
| 3 | Zhou et al., *CAM* | 2016 | Class Activation Maps |
| 4 | Chefer et al., *Transformer Interpretability* | 2021 | Relevance propagation for ViTs |

➡️ Full notes here: [`References/Literature_Review_Summary.md`](./References/Literature_Review_Summary.md)

---

### **Week 3 (Oct 28–Nov 3)**
- [x] Implemented **baseline ViT-B/16**  
- [x] Completed **60-epoch GPU training** on CUB-200  
- [x] Stored logs in `results/cub_baseline.txt`  
- [x] Saved checkpoint: `outputs/cub_baseline/best_model.pth`  
- [x] Implemented baseline inference script  

---

### **Week 4 (Nov 4–10)**
- [x] Implemented **VPT-Shallow**  
- [x] Implemented **VPT-Deep**  
- [x] Added configs:
  - `cub_vpt_shallow.yaml`
  - `cub_vpt_deep.yaml`
- [x] Added scripts:
  - `train_cub_vpt_shallow.py`
  - `train_cub_vpt_deep.py`
- [x] Trained both models on GPU (30 epochs)  
- [x] Logged results in:
  - `results/cub_vpt_shallow.txt`
  - `results/cub_vpt_deep.txt`
- [x] Saved checkpoints in:
  - `outputs/cub_vpt_shallow/`
  - `outputs/cub_vpt_deep/`

---

### **Week 5 (Nov 11–17)**
- [ ] Implement Prompt-CAM  
- [ ] Implement Attention Rollout  
- [ ] Generate first heatmap visualizations  
- [ ] Add explainability functions to scripts  

---

### **Week 6 (Nov 18–24)**
- [ ] Compute metrics:
  - Accuracy  
  - F1-score  
  - Pointing Game  
- [ ] Create comparison table (Baseline vs Shallow vs Deep)  
- [ ] Create visual grid of CAM/rollout results  

---

### **Week 7 (Nov 25–Dec 1)**
- [ ] Write final report (PDF)  
- [ ] Prepare final slide deck  
- [ ] Add final diagrams + heatmap figures  

---

### **Week 8 (Dec 2–8)**
- [ ] Final code cleanup  
- [ ] Add final visualizations to repo  
- [ ] Practice presentation  
- [ ] Submit final project  

---

## 👥 RACI Matrix  

| Task | Responsible | Accountable | Consulted | Informed |
|------|-------------|-------------|-----------|----------|
| Repo setup & documentation | Mirkomil | All | — | Instructor |
| Literature review | Asilbek | Mirkomil | Muhammad | Instructor |
| Baseline ViT training | Mirkomil | Muhammad | — | Team |
| Prompt-Tuning (Shallow/Deep) | Muhammad | Mirkomil | Asilbek | Team |
| Explainability (Prompt-CAM + Rollout) | Mirkomil | All | Muhammad | Instructor |
| Evaluation metrics | Muhammad | All | — | Instructor |
| Final report & presentation | Asilbek | All | — | Instructor |

---

## 🚀 Deliverables Summary  

- **Baseline ViT-B/16** (60-epoch training)  
- **Prompt-Tuned Models:** VPT-Shallow & VPT-Deep  
- **Explainability Toolbox:** CAM + Attention Rollout  
- **Evaluation Tables:** Accuracy, F1, Pointing Game  
- **Visual Outputs:** Heatmaps + attention maps  
- **Final PDF Report**  
- **Final Presentation Slides**  
- **Updated Weekly Roadmap**  

---

> *Maintained by Team Percepta — Central Asian University, Fall 2024*

