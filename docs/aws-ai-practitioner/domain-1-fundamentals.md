# Domain 1: Fundamentals of AI and ML

30 focused MCQs for AWS AI Practitioner exam preparation.

---

## Core Concepts (Q1-10)

### Q1: What is Artificial Intelligence (AI)?

**A)** Only systems that learn from data  
**B)** Systems performing tasks requiring human-like intelligence  
**C)** Always requires neural networks

**Answer: B**

AI simulates human intelligence for decision-making, problem-solving, or perception. Can be rule-based (chess engine) or learning-based (ML). Doesn't always require learning or neural networks.

---

### Q2: What is Machine Learning (ML)?

**A)** Manually programming all rules  
**B)** Systems that learn patterns from data without explicit programming  
**C)** Only for image recognition

**Answer: B**

ML learns from training data to make predictions or decisions. Improves with more data. Includes supervised (labeled), unsupervised (unlabeled), and reinforcement learning (reward-based).

---

### Q3: What is Deep Learning?

**A)** Any machine learning algorithm  
**B)** ML using multi-layered neural networks  
**C)** Simple linear regression

**Answer: B**

Deep learning uses neural networks with multiple layers inspired by the brain. Excels with unstructured data (images, audio, text). Requires large datasets and computational power.

---

### Q4: Which AWS service is best for building custom ML models?

**A)** Amazon Lex  
**B)** Amazon SageMaker  
**C)** Amazon Comprehend

**Answer: B**

SageMaker provides complete ML workflow: build, train, deploy models. Includes notebooks, built-in algorithms, auto-tuning. Comprehend and Lex are pre-built AI services for specific tasks.

---

### Q5: What type of learning uses labeled training data?

**A)** Unsupervised learning  
**B)** Supervised learning  
**C)** Reinforcement learning

**Answer: B**

Supervised learning trains on labeled data (input-output pairs). Examples: spam detection (email → spam/not spam), image classification (image → category). Learns mapping from inputs to known outputs.

---

### Q6: What is unsupervised learning used for?

**A)** Classification with labels  
**B)** Finding patterns in unlabeled data  
**C)** Predicting specific outputs

**Answer: B**

Unsupervised learning discovers hidden patterns without labels. Common uses: customer segmentation (clustering), dimensionality reduction, anomaly detection. No predefined categories.

---

### Q7: Which service provides conversational AI interfaces?

**A)** Amazon Lex  
**B)** Amazon SageMaker  
**C)** Amazon Forecast

**Answer: A**

Lex builds chatbots and voice interfaces. Uses automatic speech recognition (ASR) and natural language understanding (NLU). Powers Alexa. No ML expertise needed.

---

### Q8: What does Amazon Rekognition do?

**A)** Text analysis  
**B)** Image and video analysis  
**C)** Time-series forecasting

**Answer: B**

Rekognition performs computer vision tasks: object/scene detection, facial analysis, text in images, content moderation, celebrity recognition. Deep learning-based, no ML expertise required.

---

### Q9: What is reinforcement learning?

**A)** Learning from labeled examples  
**B)** Finding clusters in data  
**C)** Learning through rewards and penalties

**Answer: C**

Reinforcement learning: agent learns by trial and error, receiving rewards for good actions. Examples: game playing (AlphaGo), robotics, resource optimization. No labeled training data needed.

---

### Q10: Which neural network type is best for images?

**A)** Recurrent Neural Networks (RNNs)  
**B)** Convolutional Neural Networks (CNNs)  
**C)** Simple linear models

**Answer: B**

CNNs excel at spatial data (images, video). Use convolutional layers to detect features (edges, shapes, objects). Common in facial recognition, medical imaging, object detection.

---

## ML Lifecycle (Q11-18)

### Q11: What is the first phase of the ML lifecycle?

**A)** Model training  
**B)** Business problem definition  
**C)** Deployment

**Answer: B**

Start with clear business problem and success metrics. Determine if ML is appropriate, what data exists, and expected ROI. Prevents building wrong solutions.

---

### Q12: What percentage of ML effort typically goes to data preparation?

**A)** 20%  
**B)** 50%  
**C)** 80%

**Answer: C**

Data collection, cleaning, labeling, and feature engineering consume most effort. Quality data is critical for model performance. Poor data = poor model.

---

### Q13: What is feature engineering?

**A)** Deploying models  
**B)** Creating meaningful input variables from raw data  
**C)** Monitoring models

**Answer: B**

Feature engineering transforms raw data into useful model inputs. Examples: "days since last login" from login timestamps, "total purchases" from transaction history. Improves model performance.

---

### Q14: What is the purpose of a validation dataset?

**A)** Train the model  
**B)** Tune hyperparameters and evaluate during training  
**C)** Final evaluation only

**Answer: B**

Validation set checks performance during training, tunes hyperparameters, prevents overfitting. Separate from test set (final evaluation) and training set (learning patterns).

---

### Q15: What is accuracy as a metric?

**A)** Only for regression  
**B)** Percentage of correct predictions overall  
**C)** Only for imbalanced data

**Answer: B**

Accuracy = correct predictions / total predictions. Simple but misleading with imbalanced classes (e.g., 99% non-fraud means "always predict non-fraud" = 99% accuracy but useless).

---

### Q16: What is precision in classification?

**A)** Of predicted positives, how many were correct  
**B)** Of actual positives, how many we caught  
**C)** Overall correctness

**Answer: A**

Precision = true positives / (true positives + false positives). Measures false alarm rate. High precision = few false positives. Important when false alarms are costly.

---

### Q17: What is recall in classification?

**A)** Of predicted positives, how many were correct  
**B)** Of actual positives, how many we caught  
**C)** Overall correctness

**Answer: B**

Recall = true positives / (true positives + false negatives). Measures how many real cases we find. High recall = few missed cases. Critical in fraud/disease detection.

---

### Q18: What is model drift?

**A)** Model training errors  
**B)** Model performance degrades due to data changes over time  
**C)** Hardware failures

**Answer: B**

Drift occurs when real-world data distributions change (e.g., COVID-19 changing shopping patterns). Model trained on old patterns fails on new data. Requires monitoring and retraining.

---

## Bias & Fairness (Q19-24)

### Q19: What is data collection bias?

**A)** Algorithm preferences  
**B)** Training data doesn't represent real-world population  
**C)** Hardware limitations

**Answer: B**

Training data unrepresentative of deployment population. Example: facial recognition trained on light skin tones fails on dark skin tones. Solution: diverse, representative training data.

---

### Q20: What AWS service detects bias in ML models?

**A)** Amazon SageMaker Clarify  
**B)** Amazon Forecast  
**C)** AWS CloudTrail

**Answer: A**

SageMaker Clarify analyzes datasets and models for bias, provides explainability (SHAP values), monitors drift. Detects pre-training and post-training bias across demographic groups.

---

### Q21: What is algorithmic bias?

**A)** Bad training data  
**B)** Algorithm design favors certain outcomes  
**C)** User input errors

**Answer: B**

Bias from algorithm choices: optimization targets, loss functions, feature selection. Example: optimizing for majority class ignores minorities. Solution: fairness-aware algorithms, balanced objectives.

---

### Q22: What is label bias?

**A)** Incorrect data types  
**B)** Human annotators' prejudices in training labels  
**C)** Missing data

**Answer: B**

Labels reflect human biases. Example: biased recruiters label resumes, model learns discrimination. Solution: multiple annotators, clear guidelines, diverse labeling teams.

---

### Q23: What is demographic parity?

**A)** Model accuracy  
**B)** Equal positive prediction rates across groups  
**C)** Training speed

**Answer: B**

Fairness metric: each demographic group gets positive outcome at same rate. Example: loan approval rates equal for all ethnicities. Trade-off with other fairness definitions.

---

### Q24: Which regulation requires "right to explanation"?

**A)** HIPAA  
**B)** GDPR  
**C)** PCI DSS

**Answer: B**

GDPR (Europe) mandates users can request explanations for automated decisions affecting them (credit, employment, insurance). Requires interpretable models or post-hoc explanations.

---

## Overfitting & Underfitting (Q25-28)

### Q25: What is overfitting?

**A)** Model too simple  
**B)** Model memorizes training data including noise  
**C)** Model trains too slowly

**Answer: B**

Overfitting: excellent training accuracy (95%+), poor test accuracy (< 75%). Model learns specific examples, not general patterns. Large training-test performance gap indicates overfitting.

---

### Q26: What is underfitting?

**A)** Model too complex  
**B)** Model too simple to capture patterns  
**C)** Model trains too fast

**Answer: B**

Underfitting: poor training accuracy, poor test accuracy. Model lacks capacity to learn patterns. Solution: increase complexity, add features, reduce regularization.

---

### Q27: How do you fix overfitting?

**A)** Train longer  
**B)** Add more training data or regularization  
**C)** Remove features randomly

**Answer: B**

Solutions: more training data (best), regularization (L1/L2), simplify model, cross-validation, early stopping, dropout (neural networks). Helps model generalize.

---

### Q28: What is regularization?

**A)** Making data uniform  
**B)** Penalizing model complexity to prevent overfitting  
**C)** Speeding up training

**Answer: B**

Regularization adds penalty for large weights/complexity. L1 (Lasso) forces some weights to zero. L2 (Ridge) shrinks all weights. Prevents memorization, encourages generalization.

---

## AWS Services & Practical (Q29-30)

### Q29: Which service provides time-series forecasting?

**A)** Amazon Forecast  
**B)** Amazon Lex  
**C)** Amazon Polly

**Answer: A**

Forecast uses ML for time-series predictions: demand forecasting, inventory planning, resource scheduling. Automatically selects best algorithm, handles missing data, supports related time-series.

---

### Q30: What is Amazon Comprehend used for?

**A)** Image recognition  
**B)** Natural language processing and text analysis  
**C)** Speech synthesis

**Answer: B**

Comprehend performs NLP: sentiment analysis, entity extraction, key phrase detection, language detection, topic modeling. Pre-trained, no ML expertise needed. Supports custom entity recognition.

---

## Advanced ML Concepts (Q31-38)

### Q31: What is transfer learning?

**A)** Moving models between servers  
**B)** Using pre-trained models as starting point for new tasks  
**C)** Translating models to different languages

**Answer: B**

Transfer learning: leverage knowledge from pre-trained model (e.g., ImageNet) for your task. Faster training, less data needed. Common in deep learning. Fine-tune final layers on your data.

---

### Q32: What is Amazon Personalize?

**A)** User authentication service  
**B)** ML service for real-time personalized recommendations  
**C)** Data storage service

**Answer: B**

Personalize creates recommendation systems (products, content, searches). Uses collaborative filtering and deep learning. No ML expertise needed. Examples: "Customers who bought X also bought Y".

---

### Q33: What is batch inference vs real-time inference?

**A)** Same thing  
**B)** Batch processes many requests at once, real-time serves individual requests immediately  
**C)** Batch is always faster

**Answer: B**

Batch: process large datasets offline (e.g., daily customer scoring). Real-time: immediate predictions per request (< 100ms, e.g., fraud detection). Batch cheaper, real-time for latency-sensitive.

---

### Q34: What is Amazon Fraud Detector?

**A)** General anomaly detection  
**B)** Managed service for fraud detection using ML  
**C)** Security firewall

**Answer: B**

Fraud Detector identifies suspicious activities: payment fraud, fake accounts, abuse. Uses your historical data, AWS fraud patterns. No ML expertise needed. Real-time scoring.

---

### Q35: What is ensemble learning?

**A)** Training one large model  
**B)** Combining multiple models for better predictions  
**C)** Distributed training

**Answer: B**

Ensemble combines predictions from multiple models. Reduces overfitting, improves accuracy. Examples: Random Forest (ensemble of decision trees), XGBoost (boosted trees). Wisdom of crowds.

---

### Q36: What is cross-validation?

**A)** Testing on separate dataset  
**B)** Splitting data into folds, training/testing on different combinations  
**C)** Validating user input

**Answer: B**

K-fold cross-validation: split data into K parts, train on K-1, test on 1, rotate. Ensures model generalizes. Detects overfitting. Common: 5-fold or 10-fold.

---

### Q37: What is Amazon Textract?

**A)** Text generation service  
**B)** Extract text and data from documents using ML  
**C)** Translation service

**Answer: B**

Textract performs OCR on documents, forms, tables. Extracts structured data (key-value pairs, tables). Works on PDFs, images. Use with Bedrock for document Q&A.

---

### Q38: What is the F1 score?

**A)** Training speed metric  
**B)** Harmonic mean of precision and recall  
**C)** Loss function

**Answer: B**

F1 = 2 × (precision × recall) / (precision + recall). Balances false positives and false negatives. Use when classes imbalanced or both precision/recall matter equally.

---

## Model Training & Deployment (Q39-45)

### Q39: What are hyperparameters?

**A)** Model outputs  
**B)** Settings configured before training (learning rate, layers)  
**C)** Training data features

**Answer: B**

Hyperparameters: learning rate, batch size, number of layers, regularization strength. Set before training (not learned from data). Tuning improves performance. SageMaker offers automatic tuning.

---

### Q40: What is Amazon Polly?

**A)** Survey service  
**B)** Text-to-speech service using deep learning  
**C)** Polling service

**Answer: B**

Polly converts text to lifelike speech. Multiple voices, languages. Neural TTS for natural sound. Use cases: accessibility, IVR, content creation. Pay per character.

---

### Q41: What is A/B testing in ML?

**A)** Testing two algorithms  
**B)** Comparing model versions in production with real traffic  
**C)** Code testing

**Answer: B**

A/B testing: route traffic to model A (champion) vs model B (challenger). Measure business metrics (conversion, revenue). Deploy winner. Reduces deployment risk.

---

### Q42: What is Amazon Transcribe?

**A)** Code transcription  
**B)** Speech-to-text service using ML  
**C)** Document translation

**Answer: B**

Transcribe converts audio/video to text. Supports real-time and batch. Features: speaker identification, custom vocabulary, timestamp generation. Use for meeting notes, subtitles, call analytics.

---

### Q43: What is feature scaling/normalization?

**A)** Adding more features  
**B)** Transforming features to similar ranges  
**C)** Removing features

**Answer: B**

Scaling: convert features to similar scale (e.g., 0-1 or mean=0, std=1). Prevents features with large values dominating. Important for distance-based algorithms (KNN, neural networks).

---

### Q44: What is Amazon Translate?

**A)** Code conversion  
**B)** Neural machine translation service  
**C)** Data transformation

**Answer: B**

Translate provides real-time language translation. Supports 75+ languages. Uses neural MT for fluency. Batch or real-time. Use cases: website localization, customer communication.

---

### Q45: What is early stopping?

**A)** Canceling training  
**B)** Stopping training when validation performance stops improving  
**C)** Starting training early

**Answer: B**

Early stopping: monitor validation loss during training, stop when it stops decreasing (starts increasing). Prevents overfitting. Saves time and resources.

---

## Additional AWS Services (Q46-50)

### Q46: What is Amazon SageMaker Autopilot?

**A)** Automatic deployment  
**B)** Automated ML (AutoML) for building models without code  
**C)** Automatic scaling

**Answer: B**

Autopilot automates ML workflow: data analysis, feature engineering, algorithm selection, hyperparameter tuning. Generates notebooks showing work. No ML expertise needed for baseline model.

---

### Q47: What is Amazon Augmented AI (A2I)?

**A)** AI training service  
**B)** Human review workflows for ML predictions  
**C)** Data augmentation

**Answer: B**

A2I adds human review to ML predictions. Route low-confidence predictions to humans. Use cases: content moderation, document review, quality control. HITL implementation.

---

### Q48: What is model monitoring in production?

**A)** Watching training progress  
**B)** Tracking model performance, drift, data quality in production  
**C)** Security monitoring

**Answer: B**

Monitor: prediction quality, latency, data drift, model drift. SageMaker Model Monitor automates this. Set alerts for degradation. Trigger retraining when needed.

---

### Q49: What is Amazon Lookout for Vision?

**A)** User authentication  
**B)** Detect visual defects in manufacturing using computer vision  
**C)** Video streaming

**Answer: B**

Lookout for Vision identifies product defects, damage, irregularities in images. Train with as few as 30 images. Use cases: quality control, manufacturing, assembly lines.

---

### Q50: What is gradient descent?

**A)** Data cleaning method  
**B)** Optimization algorithm that updates model weights to minimize loss  
**C)** Feature selection

**Answer: B**

Gradient descent: iteratively adjust model weights in direction that reduces error (loss). Learning rate controls step size. Foundation of neural network training. Variants: SGD, Adam.

---

## Exam Tips

**Key Concepts to Remember:**

1. **AI Hierarchy:** AI (broadest) → ML (learns from data) → Deep Learning (neural networks)
2. **Learning Types:** Supervised (labeled), Unsupervised (patterns), Reinforcement (rewards)
3. **ML Lifecycle:** Problem definition → Data prep (80% effort) → Training → Evaluation → Deployment → Monitoring → Retraining
4. **AWS Core Services:**
   - **SageMaker:** Build/train/deploy custom models
   - **Rekognition:** Images/video
   - **Comprehend:** Text/NLP
   - **Lex:** Chatbots
   - **Forecast:** Time-series
   - **Polly:** Text-to-speech
5. **Bias Types:** Data collection, algorithmic, label, measurement, aggregation
6. **Fairness:** SageMaker Clarify, demographic parity, GDPR compliance
7. **Overfitting vs Underfitting:** Train-test gap indicates overfitting, both poor = underfitting
8. **Metrics:** Accuracy (overall), Precision (false alarms), Recall (missed cases), F1 (balance)

**Study Focus:**

- Match AWS service to use case
- Identify overfitting/underfitting from scenarios
- Recognize bias types
- Understand when to use supervised vs unsupervised
- Know ML lifecycle phases
