# Boltzmann Semantic Score (BSS)

Official repository for:

**Boltzmann Semantic Score: A Semantic Metric for Evaluating Large Vision Models Using Large Language Models**  
_Ali Khajegili Mirabadi, Katherine Rich, Hossein Farahani, Ali Bashashati_  
_International Conference on Learning Representations (ICLR) 2025_

[📄 Paper (ICLR 2025)](https://arxiv.org/abs/PLACEHOLDER)  
[📁 LLM & LVM Feature Files (Google Drive)](https://drive.google.com/drive/folders/14Vw8pAsck-PlfcwbwRCl_EtE4lYqMtNJ?usp=sharing)

---

## 🔍 Overview

**Boltzmann Semantic Score (BSS)** is a novel metric for evaluating the **semantic alignment** between the representation spaces of Large Vision Models (LVMs) and Large Language Models (LLMs) using paired medical image-report datasets.

Unlike existing qualitative approaches, BSS offers a **quantitative**, **scalable**, and **expert-free** way to assess the semantic fidelity of LVMs.

---

## 🧠 Core Idea

For a dataset of paired images and medical reports:
- Use **LLMs** to create a structural representation of expert-written pathology reports
- Use **LVMs** to create an analogous structure from medical images
- Define **BSS** as the structural alignment between the two modalities using a Boltzmann-based similarity measure

---

## 🗂 Repository Structure

<pre lang="markdown"> ``` Boltzmann/ 
  ├── boltzmann_semantic_score/ # BSS computation core code
  ├── text_retrieval/ # Info retrieval code on LLM encodings
  ├── assets/ # all the required input and output files
      ├──files/ # all the required input files for running different codes
      ├──generated_files/ # all the outputs you generate with the codes in this package goes in this directory
  └── README.md # Project overview ``` </pre>
