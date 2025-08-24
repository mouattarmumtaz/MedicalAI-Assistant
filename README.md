# 🩺 MedicalAI Assistant

A demonstration of how **AI can assist in medical workflows** by combining:
- **Chest X-ray classification** using ResNet18
- **Report summarization** with transformers
- **Retrieval-Augmented Generation (RAG)** for medical guidelines

This project showcases how multimodal AI (image + text) can support clinical decision-making in a safe, explainable way.

---

## 🔗Purpose

This repository demonstrates the implementation of a **Medical AI Assistant** that:
- Processes and classifies chest X-rays into **Normal** or **Pneumonia**
- Summarizes radiology or clinical reports into concise findings
- Retrieves and answers clinical guideline questions with **RAG**
- Provides an **interactive Gradio UI** for end-to-end analysis

---

## 🔗Features

- **X-ray Classification**:  
  - ResNet18 backbone (ImageNet pretrained)  
  - Temperature scaling & test-time augmentation  
  - Confidence-based decision text  

- **Report Summarization**:  
  - Transformer summarizer (`t5-small`)  
  - Converts lengthy medical notes into concise summaries  

- **RAG Question Answering**:  
  - Uses FAISS + MiniLM embeddings  
  - Retrieves chunks from guideline PDFs / text files  
  - Flan-T5 used for context-aware answers  

- **Interactive Gradio UI**:  
  - Upload X-ray images  
  - Paste medical reports  
  - Ask free-text guideline queries  
  - Get structured results (classification, summary, guideline answer)  

---

## 🔗Architecture
![alt text](image.png)

---

## 🔗Prerequisites

- Python 3.10+  
- GPU with CUDA (optional, recommended)  
- Dependencies:  
  ```bash
  pip install torch torchvision transformers langchain langchain-community gradio scikit-learn pydicom faiss-cpu
---
## 🔗Setup Instructions

### 1. Clone & setup
```
git clone https://github.com/your-username/MedicalAI-Assistant.git
cd MedicalAI-Assistant
pip install -r requirements.txt
```
### 2. Prepare artifacts
- Place trained X-ray model in artifacts/xray_model.pth

- Place guidelines (guidelines.pdf or .txt) inside data/

- Optionally add Reports.csv for summarization testing

### 3. Run the application
```
jupyter notebook main.ipynb
```
### 4. Launch UI
The Gradio interface will be available at:

```
http://127.0.0.1:7860
```

---
## 🔗 Project Structure
```
MedicalAI-Assistant/
├── data/                # X-rays, reports, guidelines
├── artifacts/           # Model weights, FAISS index, summarizer
├── XRAYs_class.ipynb    # X-ray classification pipeline
├── reports.ipynb        # Report summarization
├── rag.ipynb            # RAG guideline QA
├── utils.ipynb          # Helper functions
├── main.ipynb           # Integrated Gradio UI
└── README.md            # Project documentation
```
---
## 🔗Configuration
### X-ray Classification
- Model: ResNet18 pretrained on ImageNet

- Classes: NORMAL, PNEUMONIA

- Uncertainty threshold: 0.55

### Report Summarization
- **Default model:** (`t5-small`) 

- Configurable via artifacts/summarizer/

### RAG Settings
- **Embedding Model:** all-MiniLM-L6-v2

- **Vector DB:** FAISS

- **Retriever:** Top-k = 3

- **LLM:** Flan-T5 (base or small)
---
## 🔗Example Workflow
1. Upload a chest X-ray image

2. Paste a medical report

3. Ask: “What is the recommended first-line treatment for pneumonia?”

4. Get:

- **X-ray result** → "Prediction: Pneumonia (confidence 0.82)"

- **Report summary** → concise structured findings

- **Guideline answer** → retrieved + evidence-based response
---
## 🔗Evaluation
This repository includes assets for evaluation of:

- X-ray classification accuracy (Confusion matrix, ROC-AUC, reports)

- Summarization quality (manual inspection or ROUGE metrics)

- RAG QA performance (precision of retrieved guideline chunks)
---
## 🔗Contributing

 Contributions are welcome! Please feel free to submit a Pull Request.

---
## 🔗Disclaimer
⚠️ This is a demonstration/research project only.
It is not intended for clinical use without regulatory approval.