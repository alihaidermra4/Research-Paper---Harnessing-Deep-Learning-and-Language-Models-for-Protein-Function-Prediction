# Research Paper - Harnessing Deep Learning and Language Models for Protein Function Prediction

# A CAFA5-Based Comparative Deep Learning Study
This repository contains the implementation and experiments from the research article:

“Harnessing Deep Learning and Language Models for Protein Function Prediction: A CAFA5-Based Study”
Ali Haider, Jamal Shah, Musadaq Mansoor, Omar Bin Samin
Published in Journal of Computing & Biomedical Informatics, Volume 10, Issue 1, 2025.

# 🚀 Project Overview

Protein function prediction plays a central role in drug discovery, genome annotation, and systems biology. However, traditional sequence-alignment and homology-based methods often fail to generalize, especially for proteins lacking close evolutionary relatives. This project addresses these limitations by exploring multiple Deep Learning architectures trained on state-of-the-art T5 protein embeddings.

This study conducts one of the most rigorous, side-by-side comparative evaluations of deep learning models on the CAFA5 dataset, using a uniform preprocessing pipeline and identical evaluation metrics.

# 🧬 Key Features

✔️ Uses CAFA5 dataset with 142,246 curated protein sequences
✔️ Incorporates T5-based protein language embeddings (1024-dim)
✔️ Evaluates multiple deep learning architectures under identical settings
✔️ Multi-label classification across 1,500 GO terms
✔️ Metrics: Binary Accuracy, Hamming Loss, AUC, ROC-AUC, Binary Cross-Entropy Loss
✔️ Reproducible TensorFlow/Keras implementations

# 🧠 Models Implemented

The project compares the performance of the following architectures:
-> GRU
-> LSTM
-> Bi-LSTM
-> Deep Neural Network (DNN)
-> Bi-LSTM + Attention Mechanism

Each model uses:

* T5 embeddings as input
* Sigmoid output for multi-label classification
* Binary cross-entropy loss
* Adam/AdamW optimizers
* Early stopping and dropout regularization

# 📊 Results Summary

A comparative performance snapshot from the study:

# Model	Loss	Binary Accuracy	AUC	Hamming Loss	ROC-AUC
GRU	0.0781	0.9767	0.8317	0.0233	0.8348
LSTM	0.0781	0.9771	0.8317	0.0229	0.8348
Bi-LSTM	0.0781	0.9755	0.8317	0.0245	0.8348
DNN	0.0638	0.9757	0.9180	0.0243	0.9239
Bi-LSTM + Attention	0.0638	0.9759	0.9180	0.0241	0.9239

# 🔍 Key Insight:
Bi-LSTM with Attention and DNN significantly outperform classical RNN-based architectures, especially in AUC and ROC-AUC. This demonstrates their superior ability to model long-range residue dependencies.

# 📂 Dataset Information

Source: CAFA5 Challenge

Format: FASTA sequences + GO annotations (BP, MF, CC)

Preprocessing:
* Removal of obsolete GO terms
* GO “true path rule” propagation
* Filtering to top 1,500 frequent GO terms

Final dataset: 142,246 protein embeddings

Embeddings are obtained using the T5 Protein Language Model from Rost Lab.

# 🔧 Methodology Workflow

* FASTA sequence collection & cleaning
* T5 Embedding Generation
* GO Hierarchy Expansion
* Multi-label Vector Construction
* Train/Validation/Test Split (80/10/10)
* Model Training & Evaluation

# 🛠 Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy, Pandas, Scikit-learn
* Matplotlib for visualization

# 📌 Conclusions

This work demonstrates that:

Deep architectures with attention mechanisms capture long-range dependencies in protein sequences more effectively.

T5 protein language embeddings significantly boost prediction performance.

Sequential RNN models are strong baselines, but attention-equipped models provide clearer biological discrimination.

Future directions include:

* Training with more epochs and larger hardware
* Exploring transformer-based models (ESM, ProtBERT)
* Integrating structural + evolutionary information
* Improving label dependency modeling and explainability

# 📄 Citation

If you use this work, please cite:

Ali Haider, Jamal Shah, Musadaq Mansoor, & Omar Bin Samin. (2025). Harnessing Deep Learning and Language Models for Protein Function Prediction: A CAFA5-Based Study. Journal of Computing & Biomedical Informatics, 10(01). Retrieved from https://jcbi.org/index.php/Main/article/view/1128
