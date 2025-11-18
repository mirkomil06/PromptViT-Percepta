# 🗂️ ROADMAP — PromptViT-Percepta  

**Project:** Prompt-Tuned ViTs for Explainable Fine-Grained Recognition  
**Team:** Percepta  
**Course:** Computer Vision (Fall 2025)  
**Instructor:** Dr. I. Atadjanov & Dr. B. Kiani  

---

## 📅 Week-by-Week Milestones  

| Week | Dates | Milestone / Task | Deliverable | Owner(s) |
|------|--------|------------------|--------------|-----------|
| **W1** | Oct 14–21 | Team formation, topic confirmation, and repo setup | GitHub repo with README.md and ROADMAP.md created | All |
| **W2** | Oct 21–27 | Literature review and dataset preparation | Summary table of 5–10 related papers; dataset verified (CUB, Cars, Flowers) | Asilbek |
| **W3** | Oct 28–Nov 3 | Baseline ViT model setup | Fine-tuning ViT-B/16 on small split (CUB-200) | Mirkomil |
| **W4** | Nov 4–10 | Prompt-tuning implementation | Code for prompt embedding + model training | Muhammad |
| **W5** | Nov 11–17 | Explainability module (Prompt-CAM, Attention Rollout) | Visual heatmaps and interpretability results | Mirkomil |
| **W6** | Nov 18–24 | Evaluation & comparison | Metrics: Accuracy, F1, Pointing Game; results table vs baseline | All |
| **W7** | Nov 25–Dec 1 | Report writing & cleanup | Midterm proposal (PDF) finalized, slides prepared | Asilbek |
| **W8** | Dec 2–8 | Final refinements & presentation | Code cleanup, visual results upload, final presentation rehearsal | All |

---

## ✅ Weekly Progress Log  

Use this section to log updates as you work each week.  
Each update should include **3–6 short bullet points** about progress, challenges, or changes.

### Week 1 (Oct 14–21)
- [x] Formed Team Percepta  
- [x] Created GitHub repository  
- [x] Added README.md and ROADMAP.md  

### Week 2 (Oct 21–27)
- [x] Reviewed 4 core papers (ViT, Visual Prompt Tuning, CAM, Transformer Interpretability)
- [x] Created literature review summary table
- [x] Documented dataset sources and Kaggle download links
- [x] Downloaded and organized datasets locally (not uploaded due to size limits)
- [x] Prepared data preprocessing plan and documentation
#### 📚 Literature Review Summary
| # | Paper | Year | Contribution |
|:-:|--------|------|---------------|
| 1 | Dosovitskiy et al., *ViT* | 2020 | Baseline Vision Transformer model |
| 2 | Jia et al., *Visual Prompt Tuning* | 2022 | Efficient fine-tuning via prompts |
| 3 | Zhou et al., *CAM* | 2016 | Introduced class activation maps |
| 4 | Chefer et al., *Transformer Interpretability* | 2021 | Transformer explainability via relevance propagation |

➡️ [Full Literature Review Summary](References/Literature_Review_Summary.md)

### Week 3 (Oct 28–Nov 3)
- [x] Implemented ViT-B/16 baseline fine-tuning on CUB-200 (CPU)
- [x] Stored training logs and metrics in results/cub_baseline_cpu.txt
- [x] Saved model checkpoint: outputs/cub_baseline_cpu/best_model.pth
- [x] Created inference script and validated model on a sample image

### Week 4 (Nov 4–10)
- [ ] Add prompt-tuning modules and configuration  
- [ ] Validate training performance on smaller splits  

### Week 5 (Nov 11–17)
- [ ] Integrate Prompt-CAM and visualize attention heatmaps  
- [ ] Document example explanations  

### Week 6 (Nov 18–24)
- [ ] Run evaluations, generate comparison tables  
- [ ] Analyze interpretability–accuracy trade-offs  

### Week 7 (Nov 25–Dec 1)
- [ ] Write midterm proposal report (CV25_Proposal_Percepta.pdf)  
- [ ] Prepare PowerPoint presentation  

### Week 8 (Dec 2–8)
- [ ] Upload final visuals, weights, and report  
- [ ] Present project in class  

---

## 👥 RACI Matrix  

| Task | Responsible | Accountable | Consulted | Informed |
|------|--------------|--------------|------------|-----------|
| Repo setup & documentation | Mirkomil | All | — | Instructor |
| Literature review | Asilbek | Mirkomil | Muhammad | Instructor |
| Baseline ViT implementation | Mirkomil | Muhammad | — | Team |
| Prompt-tuning | Muhammad | Mirkomil | Asilbek | Team |
| Explainability visualization | Mirkomil | All | — | Instructor |
| Evaluation & metrics | Muhammad | All | — | Instructor |
| Report & presentation | Asilbek | All | — | Instructor |

---

## 🚀 Deliverables Summary  

- ✅ **Midterm Proposal:** 4–6 page PDF (CV25_Proposal_Percepta.pdf)  
- ✅ **Code Repository:** Working repo with README.md and ROADMAP.md  
- ✅ **Model Outputs:** Trained checkpoints and sample visualizations  
- ✅ **Presentation Slides:** Final midterm presentation (5–8 minutes)  
- ✅ **Weekly Logs:** Updated progress in ROADMAP.md  

---

> *Maintained by Team Percepta – Central Asian University, Fall 2025*
