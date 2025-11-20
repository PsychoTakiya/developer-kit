# Domain 1: Fundamentals of AI and ML

110 focused MCQs for AWS AI Practitioner exam preparation.

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

## Task Statement 1.1: AI/ML/DL Terms & Definitions (Q11-30)

### Q11: What is a neural network?

**A)** Database structure  
**C)** Computing system inspired by biological neurons with interconnected nodes  
**B)** Network security protocol

**Answer: C**

Neural network: computational model with layers of interconnected nodes (neurons). Input layer receives data, hidden layers process patterns, output layer produces predictions. Foundation of deep learning.

---

### Q12: What is computer vision?

**A)** Hardware displays  
**B)** AI field enabling machines to interpret and understand visual information  
**C)** Screen resolution technology

**Answer: B**

Computer vision: AI subfield for analyzing images/video. Tasks: object detection, facial recognition, scene understanding, OCR. Examples: Amazon Rekognition, autonomous vehicles, medical imaging analysis.

---

### Q13: What is Natural Language Processing (NLP)?

**A)** Network protocols  
**C)** AI field enabling machines to understand and generate human language  
**B)** Programming language

**Answer: C**

NLP: AI subfield for text/speech processing. Tasks: sentiment analysis, translation, question answering, text generation. Examples: Amazon Comprehend, chatbots, voice assistants, LLMs.

---

### Q14: What is a model in ML?

**A)** 3D design template  
**B)** Trained mathematical representation learned from data  
**C)** Hardware specification

**Answer: B**

Model: mathematical function learned from training data. Maps inputs to outputs (e.g., email → spam/not spam). Contains learned parameters (weights). Saved and deployed for inference on new data.

---

### Q15: What is an algorithm in AI/ML?

**A)** Final trained model  
**B)** Step-by-step procedure/recipe for learning patterns from data  
**C)** Data storage method

**Answer: B**

Algorithm: method for training models. Examples: decision trees, neural networks, linear regression, XGBoost. Different algorithms suit different data types and problems. SageMaker provides built-in algorithms.

---

### Q16: What is training in ML?

**A)** User education  
**C)** Process of learning patterns from data by adjusting model parameters  
**B)** Testing models

**Answer: C**

Training: feed data to algorithm, adjust weights to minimize prediction errors. Uses training dataset. Iterative process (epochs). Result: trained model ready for inference. Most compute-intensive phase.

---

### Q17: What is inferencing in ML?

**A)** Training phase  
**B)** Using trained model to make predictions on new unseen data  
**C)** Data collection

**Answer: B**

Inferencing (prediction): apply trained model to new inputs. Real-time (immediate predictions) or batch (process many at once). Lower compute than training. Deployed models serve inference requests.

---

### Q18: What is a Large Language Model (LLM)?

**A)** Large database  
**C)** Massive neural network trained on text to understand and generate language  
**B)** Code compiler

**Answer: C**

LLM: foundation model with billions/trillions of parameters trained on vast text corpus. Examples: GPT, Claude, Llama. Capabilities: text generation, translation, question answering, summarization. Accessed via Amazon Bedrock.

---

### Q19: What is tabular data?

**A)** Unstructured text  
**C)** Data organized in rows and columns with defined features  
**B)** Images

**Answer: C**

Tabular: structured data in tables (spreadsheets, databases). Rows = records, columns = features. Examples: customer demographics, sales transactions. Common ML algorithms: XGBoost, linear models, decision trees.

---

### Q20: What is time-series data?

**A)** Random data points  
**B)** Sequential data points indexed by time  
**C)** Static snapshots

**Answer: B**

Time-series: data collected at regular time intervals. Examples: stock prices, sensor readings, web traffic, sales over time. Requires specialized algorithms (ARIMA, LSTM, Prophet). Amazon Forecast handles time-series.

---

### Q21: What is image data?

**A)** Text files  
**C)** Visual data represented as pixel matrices  
**B)** Audio waveforms

**Answer: C**

Image data: 2D/3D arrays of pixels with color values. Unstructured. Requires CNNs for processing. Use cases: object detection, facial recognition, medical imaging. Services: Rekognition, Textract (OCR).

---

### Q22: What is text data?

**A)** Structured tables  
**B)** Unstructured sequences of characters and words  
**C)** Numerical data

**Answer: B**

Text data: unstructured strings (documents, emails, reviews, social media). Requires NLP/LLMs. Preprocessing: tokenization, embeddings. Use cases: sentiment analysis, classification, generation. Services: Comprehend, Bedrock.

---

### Q23: What is structured data?

**A)** Random unorganized information  
**C)** Data organized in predefined format with clear schema  
**B)** Images and videos

**Answer: C**

Structured: organized, labeled, easily searchable. Fixed schema (rows/columns). Examples: relational databases, CSV files, JSON with schema. Easier to analyze. Traditional ML algorithms work well.

---

### Q24: What is unstructured data?

**A)** Database tables  
**B)** Data without predefined format or organization  
**C)** Spreadsheets

**Answer: B**

Unstructured: no fixed format. Examples: text documents, images, audio, video, social media. Harder to analyze. Requires deep learning (CNNs, RNNs, Transformers). Represents ~80% of enterprise data.

---

### Q25: What is labeled data?

**A)** Data without annotations  
**C)** Training data with correct answers/annotations attached  
**B)** Encrypted data

**Answer: C**

Labeled: data with ground truth tags. Examples: images with object labels, emails marked spam/not spam. Required for supervised learning. Expensive to create. Services: SageMaker Ground Truth for labeling.

---

### Q26: What is unlabeled data?

**A)** Corrupted data  
**B)** Data without annotations or ground truth labels  
**C)** Encrypted data

**Answer: B**

Unlabeled: raw data without tags. Cheaper and more abundant than labeled. Used in unsupervised learning (clustering, anomaly detection) and semi-supervised learning. Can be labeled later if needed.

---

### Q27: What is the key difference between AI and ML?

**A)** No difference  
**C)** AI is broader (includes rule-based); ML specifically learns from data  
**B)** ML is broader than AI

**Answer: C**

AI: broad concept of machines performing intelligent tasks (includes rule-based systems like chess engines). ML: subset of AI that learns patterns from data without explicit programming. ML is one approach to achieve AI.

---

### Q28: What is the key difference between ML and Deep Learning?

**A)** No difference  
**B)** DL uses multi-layered neural networks; ML includes simpler algorithms  
**C)** DL cannot learn from data

**Answer: B**

ML: includes all learning algorithms (linear regression, decision trees, neural networks). Deep Learning: subset using multi-layered neural networks. DL requires more data/compute but excels with unstructured data (images, text, audio).

---

### Q29: What are characteristics of batch inference?

**A)** Real-time individual predictions  
**C)** Processes many requests together offline, cost-effective, higher latency  
**B)** Interactive responses

**Answer: C**

Batch: process large datasets at once (e.g., nightly scoring of all customers). Scheduled runs, higher throughput, lower cost per prediction. Use when immediate results not needed. SageMaker Batch Transform.

---

### Q30: What are characteristics of real-time inference?

**A)** Processes data once daily  
**B)** Immediate individual predictions with low latency (< 100ms)  
**C)** No infrastructure needed

**Answer: B**

Real-time: immediate predictions per request. Low latency (milliseconds). Always-on endpoint. Higher cost per prediction. Use when user waiting or time-sensitive (fraud detection, recommendations). SageMaker Endpoints.

---

## ML Lifecycle (Q31-38)

### Q31: What is the first phase of the ML lifecycle?

**A)** Model training  
**B)** Business problem definition  
**C)** Deployment

**Answer: B**

Start with clear business problem and success metrics. Determine if ML is appropriate, what data exists, and expected ROI. Prevents building wrong solutions.

---

### Q32: What percentage of ML effort typically goes to data preparation?

**A)** 20%  
**B)** 50%  
**C)** 80%

**Answer: C**

Data collection, cleaning, labeling, and feature engineering consume most effort. Quality data is critical for model performance. Poor data = poor model.

---

### Q33: What is feature engineering?

**A)** Deploying models  
**B)** Creating meaningful input variables from raw data  
**C)** Monitoring models

**Answer: B**

Feature engineering transforms raw data into useful model inputs. Examples: "days since last login" from login timestamps, "total purchases" from transaction history. Improves model performance.

---

### Q34: What is the purpose of a validation dataset?

**A)** Train the model  
**B)** Tune hyperparameters and evaluate during training  
**C)** Final evaluation only

**Answer: B**

Validation set checks performance during training, tunes hyperparameters, prevents overfitting. Separate from test set (final evaluation) and training set (learning patterns).

---

### Q35: What is accuracy as a metric?

**A)** Only for regression  
**B)** Percentage of correct predictions overall  
**C)** Only for imbalanced data

**Answer: B**

Accuracy = correct predictions / total predictions. Simple but misleading with imbalanced classes (e.g., 99% non-fraud means "always predict non-fraud" = 99% accuracy but useless).

---

### Q36: What is precision in classification?

**A)** Of predicted positives, how many were correct  
**B)** Of actual positives, how many we caught  
**C)** Overall correctness

**Answer: A**

Precision = true positives / (true positives + false positives). Measures false alarm rate. High precision = few false positives. Important when false alarms are costly.

---

### Q37: What is recall in classification?

**A)** Of predicted positives, how many were correct  
**B)** Of actual positives, how many we caught  
**C)** Overall correctness

**Answer: B**

Recall = true positives / (true positives + false negatives). Measures how many real cases we find. High recall = few missed cases. Critical in fraud/disease detection.

---

### Q38: What is model drift?

**A)** Model training errors  
**B)** Model performance degrades due to data changes over time  
**C)** Hardware failures

**Answer: B**

Drift occurs when real-world data distributions change (e.g., COVID-19 changing shopping patterns). Model trained on old patterns fails on new data. Requires monitoring and retraining.

---

## Bias & Fairness (Q39-44)

### Q39: What is data collection bias?

**A)** Algorithm preferences  
**B)** Training data doesn't represent real-world population  
**C)** Hardware limitations

**Answer: B**

Training data unrepresentative of deployment population. Example: facial recognition trained on light skin tones fails on dark skin tones. Solution: diverse, representative training data.

---

### Q40: What AWS service detects bias in ML models?

**A)** Amazon SageMaker Clarify  
**B)** Amazon Forecast  
**C)** AWS CloudTrail

**Answer: A**

SageMaker Clarify analyzes datasets and models for bias, provides explainability (SHAP values), monitors drift. Detects pre-training and post-training bias across demographic groups.

---

### Q41: What is algorithmic bias?

**A)** Bad training data  
**B)** Algorithm design favors certain outcomes  
**C)** User input errors

**Answer: B**

Bias from algorithm choices: optimization targets, loss functions, feature selection. Example: optimizing for majority class ignores minorities. Solution: fairness-aware algorithms, balanced objectives.

---

### Q42: What is label bias?

**A)** Incorrect data types  
**B)** Human annotators' prejudices in training labels  
**C)** Missing data

**Answer: B**

Labels reflect human biases. Example: biased recruiters label resumes, model learns discrimination. Solution: multiple annotators, clear guidelines, diverse labeling teams.

---

### Q43: What is demographic parity?

**A)** Model accuracy  
**B)** Equal positive prediction rates across groups  
**C)** Training speed

**Answer: B**

Fairness metric: each demographic group gets positive outcome at same rate. Example: loan approval rates equal for all ethnicities. Trade-off with other fairness definitions.

---

### Q44: Which regulation requires "right to explanation"?

**A)** HIPAA  
**B)** GDPR  
**C)** PCI DSS

**Answer: B**

GDPR (Europe) mandates users can request explanations for automated decisions affecting them (credit, employment, insurance). Requires interpretable models or post-hoc explanations.

---

## Overfitting & Underfitting (Q45-48)

### Q45: What is overfitting?

**A)** Model too simple  
**B)** Model memorizes training data including noise  
**C)** Model trains too slowly

**Answer: B**

Overfitting: excellent training accuracy (95%+), poor test accuracy (< 75%). Model learns specific examples, not general patterns. Large training-test performance gap indicates overfitting.

---

### Q46: What is underfitting?

**A)** Model too complex  
**B)** Model too simple to capture patterns  
**C)** Model trains too fast

**Answer: B**

Underfitting: poor training accuracy, poor test accuracy. Model lacks capacity to learn patterns. Solution: increase complexity, add features, reduce regularization.

---

### Q47: How do you fix overfitting?

**A)** Train longer  
**B)** Add more training data or regularization  
**C)** Remove features randomly

**Answer: B**

Solutions: more training data (best), regularization (L1/L2), simplify model, cross-validation, early stopping, dropout (neural networks). Helps model generalize.

---

### Q48: What is regularization?

**A)** Making data uniform  
**B)** Penalizing model complexity to prevent overfitting  
**C)** Speeding up training

**Answer: B**

Regularization adds penalty for large weights/complexity. L1 (Lasso) forces some weights to zero. L2 (Ridge) shrinks all weights. Prevents memorization, encourages generalization.

---

## AWS Services & Practical (Q49-50)

### Q49: Which service provides time-series forecasting?

**A)** Amazon Forecast  
**B)** Amazon Lex  
**C)** Amazon Polly

**Answer: A**

Forecast uses ML for time-series predictions: demand forecasting, inventory planning, resource scheduling. Automatically selects best algorithm, handles missing data, supports related time-series.

---

### Q50: What is Amazon Comprehend used for?

**A)** Image recognition  
**B)** Natural language processing and text analysis  
**C)** Speech synthesis

**Answer: B**

Comprehend performs NLP: sentiment analysis, entity extraction, key phrase detection, language detection, topic modeling. Pre-trained, no ML expertise needed. Supports custom entity recognition.

---

## Advanced ML Concepts (Q51-58)

### Q51: What is transfer learning?

**A)** Moving models between servers  
**B)** Using pre-trained models as starting point for new tasks  
**C)** Translating models to different languages

**Answer: B**

Transfer learning: leverage knowledge from pre-trained model (e.g., ImageNet) for your task. Faster training, less data needed. Common in deep learning. Fine-tune final layers on your data.

---

### Q52: What is Amazon Personalize?

**A)** User authentication service  
**B)** ML service for real-time personalized recommendations  
**C)** Data storage service

**Answer: B**

Personalize creates recommendation systems (products, content, searches). Uses collaborative filtering and deep learning. No ML expertise needed. Examples: "Customers who bought X also bought Y".

---

### Q53: What is batch inference vs real-time inference?

**A)** Same thing  
**B)** Batch processes many requests at once, real-time serves individual requests immediately  
**C)** Batch is always faster

**Answer: B**

Batch: process large datasets offline (e.g., daily customer scoring). Real-time: immediate predictions per request (< 100ms, e.g., fraud detection). Batch cheaper, real-time for latency-sensitive.

---

### Q54: What is Amazon Fraud Detector?

**A)** General anomaly detection  
**B)** Managed service for fraud detection using ML  
**C)** Security firewall

**Answer: B**

Fraud Detector identifies suspicious activities: payment fraud, fake accounts, abuse. Uses your historical data, AWS fraud patterns. No ML expertise needed. Real-time scoring.

---

### Q55: What is ensemble learning?

**A)** Training one large model  
**B)** Combining multiple models for better predictions  
**C)** Distributed training

**Answer: B**

Ensemble combines predictions from multiple models. Reduces overfitting, improves accuracy. Examples: Random Forest (ensemble of decision trees), XGBoost (boosted trees). Wisdom of crowds.

---

### Q56: What is cross-validation?

**A)** Testing on separate dataset  
**B)** Splitting data into folds, training/testing on different combinations  
**C)** Validating user input

**Answer: B**

K-fold cross-validation: split data into K parts, train on K-1, test on 1, rotate. Ensures model generalizes. Detects overfitting. Common: 5-fold or 10-fold.

---

### Q57: What is Amazon Textract?

**A)** Text generation service  
**B)** Extract text and data from documents using ML  
**C)** Translation service

**Answer: B**

Textract performs OCR on documents, forms, tables. Extracts structured data (key-value pairs, tables). Works on PDFs, images. Use with Bedrock for document Q&A.

---

### Q58: What is the F1 score?

**A)** Training speed metric  
**B)** Harmonic mean of precision and recall  
**C)** Loss function

**Answer: B**

F1 = 2 × (precision × recall) / (precision + recall). Balances false positives and false negatives. Use when classes imbalanced or both precision/recall matter equally.

---

## Model Training & Deployment (Q59-65)

### Q59: What are hyperparameters?

**A)** Model outputs  
**B)** Settings configured before training (learning rate, layers)  
**C)** Training data features

**Answer: B**

Hyperparameters: learning rate, batch size, number of layers, regularization strength. Set before training (not learned from data). Tuning improves performance. SageMaker offers automatic tuning.

---

### Q60: What is Amazon Polly?

**A)** Survey service  
**B)** Text-to-speech service using deep learning  
**C)** Polling service

**Answer: B**

Polly converts text to lifelike speech. Multiple voices, languages. Neural TTS for natural sound. Use cases: accessibility, IVR, content creation. Pay per character.

---

### Q61: What is A/B testing in ML?

**A)** Testing two algorithms  
**B)** Comparing model versions in production with real traffic  
**C)** Code testing

**Answer: B**

A/B testing: route traffic to model A (champion) vs model B (challenger). Measure business metrics (conversion, revenue). Deploy winner. Reduces deployment risk.

---

### Q62: What is Amazon Transcribe?

**A)** Code transcription  
**B)** Speech-to-text service using ML  
**C)** Document translation

**Answer: B**

Transcribe converts audio/video to text. Supports real-time and batch. Features: speaker identification, custom vocabulary, timestamp generation. Use for meeting notes, subtitles, call analytics.

---

### Q63: What is feature scaling/normalization?

**A)** Adding more features  
**B)** Transforming features to similar ranges  
**C)** Removing features

**Answer: B**

Scaling: convert features to similar scale (e.g., 0-1 or mean=0, std=1). Prevents features with large values dominating. Important for distance-based algorithms (KNN, neural networks).

---

### Q64: What is Amazon Translate?

**A)** Code conversion  
**B)** Neural machine translation service  
**C)** Data transformation

**Answer: B**

Translate provides real-time language translation. Supports 75+ languages. Uses neural MT for fluency. Batch or real-time. Use cases: website localization, customer communication.

---

### Q65: What is early stopping?

**A)** Canceling training  
**B)** Stopping training when validation performance stops improving  
**C)** Starting training early

**Answer: B**

Early stopping: monitor validation loss during training, stop when it stops decreasing (starts increasing). Prevents overfitting. Saves time and resources.

---

## Additional AWS Services (Q66-70)

### Q66: What is Amazon SageMaker Autopilot?

**A)** Automatic deployment  
**B)** Automated ML (AutoML) for building models without code  
**C)** Automatic scaling

**Answer: B**

Autopilot automates ML workflow: data analysis, feature engineering, algorithm selection, hyperparameter tuning. Generates notebooks showing work. No ML expertise needed for baseline model.

---

### Q67: What is Amazon Augmented AI (A2I)?

**A)** AI training service  
**B)** Human review workflows for ML predictions  
**C)** Data augmentation

**Answer: B**

A2I adds human review to ML predictions. Route low-confidence predictions to humans. Use cases: content moderation, document review, quality control. HITL implementation.

---

### Q68: What is model monitoring in production?

**A)** Watching training progress  
**B)** Tracking model performance, drift, data quality in production  
**C)** Security monitoring

**Answer: B**

Monitor: prediction quality, latency, data drift, model drift. SageMaker Model Monitor automates this. Set alerts for degradation. Trigger retraining when needed.

---

### Q69: What is Amazon Lookout for Vision?

**A)** User authentication  
**B)** Detect visual defects in manufacturing using computer vision  
**C)** Video streaming

**Answer: B**

Lookout for Vision identifies product defects, damage, irregularities in images. Train with as few as 30 images. Use cases: quality control, manufacturing, assembly lines.

---

### Q70: What is gradient descent?

**A)** Data cleaning method  
**B)** Optimization algorithm that updates model weights to minimize loss  
**C)** Feature selection

**Answer: B**

Gradient descent: iteratively adjust model weights in direction that reduces error (loss). Learning rate controls step size. Foundation of neural network training. Variants: SGD, Adam.

---

## Task Statement 1.2: Practical Use Cases & Applications (Q71-90)

### Q71: When does AI/ML provide the most value?

**A)** Only for large enterprises  
**B)** Automating repetitive tasks, scaling decisions, augmenting human judgment  
**C)** Replacing all human workers

**Answer: B**

AI/ML value: automate high-volume repetitive tasks (classification, routing), scale decisions beyond human capacity (millions of transactions), augment expertise (medical diagnosis assistance), personalize at scale (recommendations).

---

### Q72: When is AI/ML NOT appropriate?

**A)** Always appropriate for every problem  
**C)** When deterministic outcomes required, cost exceeds benefit, insufficient data  
**B)** Never appropriate

**Answer: C**

Avoid AI/ML when: need guaranteed specific outcome (safety-critical systems), development cost > business value, < 100 training examples, simple rules suffice, explainability legally required but not possible.

---

### Q73: What ML technique is used for predicting continuous numerical values?

**A)** Classification  
**B)** Regression  
**C)** Clustering

**Answer: B**

Regression: predict numbers (house prices, sales revenue, temperature, customer lifetime value). Outputs continuous values. Algorithms: linear regression, XGBoost, neural networks. Example: forecasting demand.

---

### Q74: What ML technique is used for categorizing data into predefined classes?

**A)** Regression  
**C)** Classification  
**B)** Clustering

**Answer: C**

Classification: assign labels/categories (spam/not spam, fraud/legitimate, product categories). Outputs discrete classes. Algorithms: logistic regression, decision trees, neural networks. Requires labeled training data.

---

### Q75: What ML technique groups similar data without predefined labels?

**A)** Regression  
**B)** Classification  
**C)** Clustering

**Answer: C**

Clustering: discover natural groupings in data (customer segments, document topics, anomaly detection). Unsupervised learning. Algorithms: K-means, DBSCAN, hierarchical clustering. No labels needed.

---

### Q76: What is a real-world computer vision application?

**A)** Sentiment analysis  
**C)** Quality control defect detection in manufacturing  
**B)** Speech recognition

**Answer: C**

CV applications: manufacturing defect detection (Lookout for Vision), medical image analysis (X-ray diagnosis), autonomous vehicles (object detection), retail (cashierless stores), security (facial recognition), content moderation.

---

### Q77: What is a real-world NLP application?

**A)** Image classification  
**B)** Document summarization and sentiment analysis  
**C)** Video analysis

**Answer: B**

NLP applications: customer feedback sentiment analysis (Comprehend), document summarization, chatbots (Lex), email classification, contract analysis, language translation (Translate), entity extraction from text.

---

### Q78: What is a real-world speech recognition application?

**A)** Image tagging  
**C)** Medical transcription and call center analytics  
**B)** Fraud detection

**Answer: C**

Speech recognition: medical dictation (Transcribe Medical), call center quality monitoring, meeting transcription, voice assistants, accessibility features (subtitles), IVR systems (Lex + Transcribe).

---

### Q79: What is a real-world recommendation system application?

**A)** Fraud detection  
**B)** Personalized product and content recommendations  
**C)** Image recognition

**Answer: B**

Recommendations (Personalize): e-commerce product suggestions ("customers who bought X also bought Y"), streaming content (Netflix-style), news articles, job postings, ad targeting, email campaigns.

---

### Q80: What is a real-world fraud detection application?

**A)** Product recommendations  
**C)** Payment fraud and fake account detection  
**B)** Language translation

**Answer: C**

Fraud detection: payment transaction screening (Fraud Detector), account takeover detection, fake review identification, insurance claim fraud, identity verification, bot detection on websites.

---

### Q81: What is a real-world forecasting application?

**A)** Image classification  
**B)** Demand forecasting and inventory planning  
**C)** Text summarization

**Answer: B**

Forecasting (Forecast): retail demand prediction, inventory optimization, workforce planning, energy consumption forecasting, financial planning, website traffic prediction, resource allocation.

---

### Q82: How do you select the right ML technique?

**A)** Always use neural networks  
**C)** Match problem type: regression for numbers, classification for categories, clustering for grouping  
**B)** Random selection

**Answer: C**

Selection criteria: regression (predict prices, revenue), classification (spam detection, diagnostics), clustering (segmentation), time-series (forecasting), NLP (text), CV (images), recommendations (personalization). Match data type and business goal.

---

### Q83: What AWS service capability does SageMaker provide?

**A)** Only pre-trained models  
**B)** Build, train, and deploy custom ML models  
**C)** Only speech services

**Answer: B**

SageMaker: complete ML platform for custom models. Notebooks (development), built-in algorithms (XGBoost, k-means), training jobs (distributed), hyperparameter tuning, model deployment (endpoints), monitoring (Model Monitor).

---

### Q84: What AWS service capability does Transcribe provide?

**A)** Image analysis  
**C)** Automatic speech recognition (speech-to-text)  
**B)** Text generation

**Answer: C**

Transcribe: convert audio/video to text. Real-time and batch. Features: speaker identification (diarization), custom vocabulary, timestamps, punctuation, medical/call analytics variants, PII redaction.

---

### Q85: What AWS service capability does Translate provide?

**A)** Speech recognition  
**B)** Neural machine translation between languages  
**C)** Image translation

**Answer: B**

Translate: real-time text translation. 75+ languages, neural MT for fluency, batch translation, custom terminology (domain-specific), formality control, automatic language detection.

---

### Q86: What AWS service capability does Comprehend provide?

**A)** Image recognition  
**C)** NLP: sentiment, entities, key phrases, language detection  
**B)** Video analysis

**Answer: C**

Comprehend: text analysis at scale. Sentiment analysis (positive/negative/neutral), entity extraction (people, places, dates), key phrase detection, language identification, topic modeling, PII detection, custom entity recognition.

---

### Q87: What AWS service capability does Lex provide?

**A)** Image classification  
**B)** Build conversational chatbots and voice interfaces  
**C)** Data storage

**Answer: B**

Lex: conversational AI platform. Build chatbots (text/voice), automatic speech recognition (ASR), natural language understanding (NLU), intent recognition, slot filling, multi-turn conversations, integrates with Lambda.

---

### Q88: What AWS service capability does Polly provide?

**A)** Speech-to-text  
**C)** Text-to-speech with lifelike voices  
**B)** Translation

**Answer: C**

Polly: convert text to natural speech. 60+ voices, 30+ languages, neural TTS (high quality), SSML support (pronunciation control), speech marks (lip-sync), real-time streaming.

---

### Q89: What use case matches Rekognition?

**A)** Translating documents  
**B)** Facial recognition and content moderation in images/videos  
**C)** Time-series forecasting

**Answer: B**

Rekognition use cases: facial verification (access control), celebrity recognition, content moderation (inappropriate images), PPE detection (safety compliance), text in images (OCR), object/scene detection.

---

### Q90: What is the cost-benefit consideration for AI/ML?

**A)** AI/ML always saves money  
**C)** Development + infrastructure cost must be less than business value gained  
**B)** Cost doesn't matter

**Answer: C**

Cost-benefit: calculate development cost (data labeling, engineering time) + operational cost (inference, infrastructure) vs business value (revenue increase, cost savings, efficiency gains). ROI must be positive. Start with pre-trained services for lower cost.

---

## Task Statement 1.3: ML Development Lifecycle & MLOps (Q91-110)

### Q91: What is Exploratory Data Analysis (EDA)?

**A)** Final model testing  
**C)** Initial investigation to understand patterns, distributions, anomalies in data  
**B)** Model deployment

**Answer: C**

EDA: visualize distributions, identify outliers, check correlations, understand data quality. Tools: histograms, scatter plots, summary statistics. Informs feature engineering. Use SageMaker Data Wrangler for visual EDA.

---

### Q92: What is data pre-processing in ML pipeline?

**A)** After model training  
**B)** Cleaning, transforming, normalizing data before training  
**C)** Final deployment step

**Answer: B**

Pre-processing: handle missing values, remove duplicates, normalize/scale features, encode categorical variables, split train/validation/test sets. Critical for model quality. SageMaker Data Wrangler automates 300+ transformations.

---

### Q93: What AWS service helps with visual data preparation?

**A)** Amazon Rekognition  
**C)** Amazon SageMaker Data Wrangler  
**B)** Amazon Comprehend

**Answer: C**

Data Wrangler: visual interface for data preparation. Import from S3/Redshift/Athena, apply 300+ built-in transformations, generate code automatically, get data quality insights. Reduces data prep time from weeks to minutes.

---

### Q94: What is SageMaker Feature Store used for?

**A)** Model marketplace  
**B)** Centralized repository for storing and managing ML features  
**C)** Image storage

**Answer: B**

Feature Store: central location for feature definitions. Online store (low-latency inference), offline store (training). Ensures training-serving consistency, enables feature reuse across teams, supports time-travel queries.

---

### Q95: What is hyperparameter tuning?

**A)** Data cleaning  
**C)** Optimizing model configuration settings to improve performance  
**B)** Feature selection

**Answer: C**

Hyperparameter tuning: systematically search for best learning rate, batch size, number of layers, regularization strength. SageMaker automatic tuning tries combinations, selects best based on validation metrics. Improves accuracy.

---

### Q96: What is the purpose of model evaluation?

**A)** Deploy immediately  
**B)** Assess performance using metrics before production deployment  
**C)** Collect more data

**Answer: B**

Evaluation: test model on holdout test set. Measure metrics (accuracy, precision, recall, F1, AUC). Compare against baseline and business requirements. Decide if model ready for production. Critical quality gate.

---

### Q97: What is SageMaker Model Monitor used for?

**A)** Training models  
**C)** Continuous monitoring of deployed model quality and drift  
**B)** Data labeling

**Answer: C**

Model Monitor: detects data drift (input distribution changes), model drift (accuracy degradation), bias drift, data quality issues. Automatic baseline creation, scheduled checks, CloudWatch alerts. Triggers retraining when needed.

---

### Q98: Where can you source ML models from?

**A)** Only train from scratch  
**C)** Open source pre-trained models, AWS model zoo, custom training, foundation models  
**B)** Only AWS services

**Answer: C**

Model sources: pre-trained models (Hugging Face, TensorFlow Hub), AWS services (SageMaker JumpStart, Bedrock FMs), custom training (SageMaker), open source frameworks. Choose based on use case, data, customization needs.

---

### Q99: What is a managed API deployment method?

**A)** Self-hosting on EC2  
**B)** Fully managed inference endpoint via AWS service API  
**C)** Manual server management

**Answer: B**

Managed API: AWS handles infrastructure (SageMaker endpoints, Bedrock API, AI services). Serverless, auto-scaling, pay-per-use. Benefits: no ops overhead, automatic updates, built-in monitoring. Example: Bedrock API calls.

---

### Q100: What is a self-hosted API deployment method?

**A)** Only SageMaker endpoints  
**C)** Deploy model on own infrastructure (EC2, ECS, Lambda) with custom API  
**B)** Third-party hosting only

**Answer: C**

Self-hosted: deploy on EC2/ECS/EKS/Lambda, manage infrastructure, create custom API. Full control over environment, optimizations, costs. More operational overhead. Use when need specific configurations or on-premises deployment.

---

### Q101: What is MLOps?

**A)** Manual ML workflows  
**B)** Practices for automating and standardizing ML lifecycle operations  
**C)** Only model training

**Answer: B**

MLOps: applying DevOps principles to ML. Automates: experimentation tracking, reproducible training, CI/CD pipelines, model versioning, monitoring, retraining. Goal: production-ready, scalable, maintainable ML systems. SageMaker Pipelines implements MLOps.

---

### Q102: What is the purpose of SageMaker Pipelines?

**A)** Data storage  
**C)** Orchestrate and automate end-to-end ML workflows  
**B)** Only model training

**Answer: C**

Pipelines: define ML workflow as code (data prep → training → evaluation → deployment). Automates execution, tracks lineage, enables versioning, supports CI/CD. Ensures reproducibility and scalability. JSON or Python SDK.

---

### Q103: What is model versioning in MLOps?

**A)** Naming models randomly  
**B)** Tracking model versions with metadata for reproducibility  
**C)** Deleting old models

**Answer: B**

Model versioning: track each model iteration with version number, training data, hyperparameters, metrics, approval status. SageMaker Model Registry manages versions. Enables rollback, comparison, governance, audit trails.

---

### Q104: What is technical debt in ML?

**A)** Financial debt  
**C)** Accumulated quick fixes and manual processes that hinder scalability  
**B)** Model accuracy

**Answer: C**

Technical debt: shortcuts like hardcoded values, manual steps, lack of testing, no monitoring. Causes: time pressure, changing requirements. Mitigation: automation (Pipelines), testing, documentation, refactoring. MLOps reduces debt.

---

### Q105: What does production readiness mean for ML models?

**A)** Model trained successfully  
**C)** Model meets performance, scalability, monitoring, and governance requirements  
**B)** High training accuracy only

**Answer: C**

Production readiness: model meets SLAs (latency, throughput), monitored for drift, has rollback plan, documented, tested, secure, compliant. Includes: endpoint auto-scaling, logging, alerting, A/B testing capability.

---

### Q106: What is AUC (Area Under ROC Curve)?

**A)** Training speed metric  
**B)** Measures classification model's ability to distinguish between classes  
**C)** Loss function

**Answer: B**

AUC: plots true positive rate vs false positive rate. Range: 0-1 (1 = perfect, 0.5 = random). Good for imbalanced classes. Higher = better discrimination. Common threshold: AUC > 0.7 acceptable, > 0.8 good.

---

### Q107: What business metric evaluates ML model ROI?

**A)** Only accuracy  
**C)** Revenue increase, cost savings, customer satisfaction improvement  
**B)** Training time

**Answer: C**

Business metrics: revenue impact (sales lift), cost reduction (automation savings), customer metrics (retention, satisfaction, NPS), efficiency gains (time saved). Compare model benefit vs development/operational cost. ROI = (Benefit - Cost) / Cost.

---

### Q108: What is model re-training?

**A)** Initial training  
**B)** Updating model with new data to maintain accuracy over time  
**C)** Hyperparameter tuning

**Answer: B**

Re-training: periodically train on fresh data when performance degrades (drift detected) or new patterns emerge. Schedule: monthly, quarterly, or trigger-based (monitoring alerts). Automated via SageMaker Pipelines.

---

### Q109: What is experimentation in MLOps?

**A)** Random testing  
**C)** Systematic tracking of model experiments with parameters and results  
**B)** Final deployment

**Answer: C**

Experimentation: track every training run with hyperparameters, data versions, metrics. Compare experiments to find best model. SageMaker Experiments automatically logs runs. Enables reproducibility and knowledge sharing.

---

### Q110: What is the purpose of SageMaker Model Registry?

**A)** Store training data  
**B)** Version control and governance for ML models  
**C)** Feature engineering

**Answer: B**

Model Registry: catalog of model versions with metadata (metrics, approval status, lineage). Supports approval workflows (dev → staging → production), audit trails, rollback. Central governance for model lifecycle management.

---

## Exam Tips

**Key Concepts to Remember:**

1. **AI Hierarchy:** AI (broadest) → ML (learns from data) → Deep Learning (neural networks)
2. **Learning Types:** Supervised (labeled), Unsupervised (patterns), Reinforcement (rewards)
3. **ML Lifecycle:** Problem definition → Data prep (80% effort) → Training → Evaluation → Deployment → Monitoring → Retraining
4. **ML Pipeline Components:**
   - **Data Collection:** Gather training data
   - **EDA:** Understand patterns, distributions, anomalies
   - **Pre-processing:** Clean, transform, normalize (Data Wrangler)
   - **Feature Engineering:** Create meaningful variables (Feature Store)
   - **Model Training:** Learn patterns from data
   - **Hyperparameter Tuning:** Optimize model settings
   - **Evaluation:** Test performance (accuracy, AUC, F1)
   - **Deployment:** Serve predictions (managed or self-hosted API)
   - **Monitoring:** Track drift, quality (Model Monitor)
   - **Re-training:** Update with new data
5. **AWS Core Services:**
   - **SageMaker:** Build/train/deploy custom models
   - **Data Wrangler:** Visual data preparation
   - **Feature Store:** Centralized feature management
   - **Pipelines:** ML workflow automation (MLOps)
   - **Model Monitor:** Drift detection
   - **Model Registry:** Version control, governance
   - **Rekognition:** Images/video
   - **Comprehend:** Text/NLP
   - **Lex:** Chatbots
   - **Forecast:** Time-series
   - **Polly:** Text-to-speech
   - **Transcribe:** Speech-to-text
   - **Translate:** Language translation
6. **Model Sources:**
   - Open source pre-trained (Hugging Face, TensorFlow Hub)
   - AWS (SageMaker JumpStart, Bedrock)
   - Custom training (SageMaker)
7. **Deployment Methods:**
   - **Managed API:** SageMaker endpoints, Bedrock, AI services (serverless, auto-scaling)
   - **Self-hosted:** EC2, ECS, Lambda (full control, more ops)
8. **MLOps Fundamentals:**
   - **Experimentation:** Track runs, parameters, metrics
   - **Reproducibility:** Versioning, pipelines, lineage
   - **Scalability:** Automated workflows, distributed training
   - **Technical Debt:** Avoid manual processes, automate testing
   - **Production Readiness:** Performance SLAs, monitoring, rollback
   - **Model Re-training:** Scheduled or triggered by drift
9. **Bias Types:** Data collection, algorithmic, label, measurement, aggregation
10. **Fairness:** SageMaker Clarify, demographic parity, GDPR compliance
11. **Overfitting vs Underfitting:** Train-test gap indicates overfitting, both poor = underfitting
12. **Performance Metrics:**
    - **Accuracy:** Overall correctness
    - **Precision:** False alarm rate
    - **Recall:** Missed cases
    - **F1:** Balance precision/recall
    - **AUC:** Classification discrimination ability (0.8+ good)
13. **Business Metrics:**
    - **ROI:** (Benefit - Cost) / Cost
    - **Revenue:** Sales lift, customer value
    - **Cost:** Savings from automation
    - **Customer:** Satisfaction, retention, NPS
14. **ML Techniques:**
    - **Regression:** Predict numbers (prices, revenue)
    - **Classification:** Categorize (spam, fraud)
    - **Clustering:** Group similar items (segmentation)
15. **Real-World Applications:**
    - **CV:** Manufacturing defects, medical imaging, security
    - **NLP:** Sentiment analysis, summarization, chatbots
    - **Speech:** Transcription, call analytics, voice assistants
    - **Recommendations:** Products, content, personalization
    - **Fraud:** Payment screening, account security
    - **Forecasting:** Demand, inventory, resource planning

**Study Focus:**

- Match AWS service to use case
- Identify overfitting/underfitting from scenarios
- Recognize bias types
- Understand when to use supervised vs unsupervised
- Know ML lifecycle phases and pipeline components
- Select appropriate ML technique (regression/classification/clustering)
- Recognize when AI/ML is NOT appropriate
- Map real-world applications to AWS services
- Understand MLOps concepts (versioning, pipelines, monitoring)
- Know AWS services for each pipeline stage (Data Wrangler, Feature Store, Pipelines, Model Monitor, Model Registry)
- Distinguish managed vs self-hosted deployment
- Understand performance metrics (accuracy, AUC, F1) and business metrics (ROI, cost, revenue)
