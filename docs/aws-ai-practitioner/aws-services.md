# AWS AI/ML Services

50 focused MCQs for AWS AI Practitioner exam preparation.

---

## Amazon SageMaker (Q1-10)

### Q1: What is Amazon SageMaker?

**A)** Only for model deployment  
**B)** Fully managed ML platform for build, train, deploy  
**C)** Data storage service

**Answer: B**

SageMaker: end-to-end ML platform with notebooks, built-in algorithms, training jobs, hosting. Scales automatically, integrates with AWS ecosystem. Central service for custom ML workflows.

---

### Q2: What is SageMaker Studio?

**A)** Video editing tool  
**B)** Web-based IDE for ML workflows  
**C)** Music production

**Answer: B**

Studio: integrated development environment for ML. Single interface for notebooks, experiments, pipelines, model registry, debugger. Collaborative workspace for data science teams.

---

### Q3: What is SageMaker Ground Truth?

**A)** Data validation  
**B)** Data labeling service with human and ML workers  
**C)** Model testing

**Answer: B**

Ground Truth: build labeled datasets using human annotators or active learning. Reduces labeling costs up to 70%. Supports images, text, video. Built-in and custom labeling workflows.

---

### Q4: What is SageMaker Autopilot?

**A)** Auto-deployment  
**B)** AutoML that automatically builds, trains, tunes models  
**C)** Auto-scaling

**Answer: B**

Autopilot: AutoML for tabular data. Automatically explores algorithms (XGBoost, linear, deep learning), tunes hyperparameters, generates explainable notebooks. No code required for baseline models.

---

### Q5: What is SageMaker Clarify?

**A)** Model compression  
**B)** Bias detection and model explainability  
**C)** Data cleaning

**Answer: B**

Clarify: detect bias in data and models across demographics. Provides SHAP values for explainability. Monitors deployed models for drift. Supports fairness metrics, feature importance analysis.

---

### Q6: What is SageMaker Model Monitor?

**A)** Video monitoring  
**B)** Continuous monitoring of model quality in production  
**C)** Network monitoring

**Answer: B**

Model Monitor: detects data drift, model drift, bias drift, feature attribution drift. Automatic baseline creation, scheduled monitoring, CloudWatch integration. Alerts on quality degradation.

---

### Q7: What is SageMaker Pipelines?

**A)** Data pipelines only  
**B)** CI/CD for ML workflows  
**C)** Network pipelines

**Answer: B**

Pipelines: orchestrate ML workflows (data prep → training → evaluation → deployment). Automates retraining, versioning. JSON/Python SDK. Integrates with Model Registry for governance.

---

### Q8: What is SageMaker Feature Store?

**A)** Marketplace  
**B)** Centralized repository for ML features  
**C)** App store

**Answer: B**

Feature Store: manage, share, reuse features across teams. Online store (low-latency), offline store (training). Ensures training-serving consistency. Time-travel queries for point-in-time correctness.

---

### Q9: What is SageMaker Neo?

**A)** New SageMaker version  
**B)** Optimize models for edge devices  
**C)** Neural network only

**Answer: B**

Neo: compile models to run 2x faster on edge devices (IoT, mobile). Supports TensorFlow, PyTorch, MXNet. Optimizes for CPU, GPU, specific hardware. Deploy with AWS IoT Greengrass.

---

### Q10: What is SageMaker Batch Transform?

**A)** Real-time inference  
**B)** Offline batch predictions on large datasets  
**C)** Data transformation only

**Answer: B**

Batch Transform: run inference on S3 datasets without persistent endpoint. Cost-effective for periodic predictions (daily scoring). No infrastructure management, automatic scaling.

---

## Amazon Bedrock (Q11-18)

### Q11: What is Amazon Bedrock?

**A)** Database service  
**B)** Fully managed service for foundation models  
**C)** Container service

**Answer: B**

Bedrock: access foundation models (Claude, Llama, Titan, Jurassic) via API. No infrastructure management. Customize with fine-tuning, RAG. Serverless, pay-per-use pricing.

---

### Q12: What are Bedrock Agents?

**A)** Human agents  
**B)** Autonomous agents that execute multi-step tasks  
**C)** Sales agents

**Answer: B**

Agents: orchestrate foundation models to complete complex tasks. Use tools (APIs, Lambda), retrieve knowledge, reason through steps. Break user requests into actions automatically.

---

### Q13: What is Bedrock Knowledge Bases?

**A)** Training data  
**B)** RAG implementation with vector storage  
**C)** Documentation

**Answer: B**

Knowledge Bases: implement RAG without managing infrastructure. Connect to S3, chunk documents, create embeddings, store in vector DB (OpenSearch, Pinecone). Retrieves context for prompts.

---

### Q14: What are Bedrock Guardrails?

**A)** Physical barriers  
**B)** Safety controls for content filtering  
**C)** Network security

**Answer: B**

Guardrails: filter harmful content, PII, sensitive topics. Denied topics, content filters, word filters. Apply to inputs and outputs. Configurable thresholds for toxicity, hate, violence.

---

### Q15: What is Bedrock model evaluation?

**A)** Manual testing  
**B)** Automated comparison of model performance  
**C)** User surveys

**Answer: B**

Model evaluation: compare models on your tasks (accuracy, robustness, toxicity). Built-in datasets or custom. Metrics: ROUGE, BLEU, toxicity scores. Choose best model objectively.

---

### Q16: What is Bedrock Provisioned Throughput?

**A)** Free tier  
**B)** Reserved model capacity for consistent performance  
**C)** Network bandwidth

**Answer: B**

Provisioned Throughput: purchase reserved model units for predictable latency/throughput. Lower cost for high-volume use. Commitment-based pricing vs on-demand. Ideal for production workloads.

---

### Q17: What is Bedrock fine-tuning?

**A)** Adjusting prompts  
**B)** Customizing models with your data  
**C)** Performance tuning

**Answer: B**

Fine-tuning: adapt foundation models to your domain/style. Provide labeled examples, model trains on your data. Creates private custom model. Available for select models (Titan, Claude).

---

### Q18: What is Bedrock Playground?

**A)** Testing environment  
**B)** Interactive console to experiment with models  
**C)** Game platform

**Answer: B**

Playground: web interface to test prompts, compare models, adjust parameters (temperature, top-p). No code required. Export successful prompts to code. Quick experimentation.

---

## Computer Vision Services (Q19-25)

### Q19: What is Amazon Rekognition?

**A)** Audio analysis  
**B)** Image and video analysis using deep learning  
**C)** Text analysis

**Answer: B**

Rekognition: detect objects, faces, text, scenes, activities in images/videos. Facial analysis (age, emotion), face comparison, celebrity recognition. Content moderation, PPE detection.

---

### Q20: What is Rekognition Custom Labels?

**A)** Pre-trained only  
**B)** Train custom computer vision models  
**C)** Text labels

**Answer: B**

Custom Labels: build object/scene detection models with your images. AutoML approach, as few as 10 images. No ML expertise. Use cases: brand detection, defect detection.

---

### Q21: What is Amazon Textract?

**A)** Text generation  
**B)** Extract text and data from documents  
**C)** Translation

**Answer: B**

Textract: OCR with structure understanding. Extract text, tables, forms (key-value pairs). Identifies document layouts. Use for invoice processing, form digitization, document analysis.

---

### Q22: What is Textract AnalyzeExpense?

**A)** Cost analysis  
**B)** Extract data from invoices and receipts  
**C)** Expense reporting

**Answer: B**

AnalyzeExpense: specialized for financial documents. Extracts vendor, date, line items, totals. Pre-trained for invoice/receipt formats. Faster than general AnalyzeDocument for expenses.

---

### Q23: What is Amazon Lookout for Vision?

**A)** Security cameras  
**B)** Visual defect detection in manufacturing  
**C)** Eye health

**Answer: B**

Lookout for Vision: detect product defects using computer vision. Train with as few as 30 images. Identifies anomalies, damage, irregularities. Use cases: quality control, assembly verification.

---

### Q24: What is AWS Panorama?

**A)** Photo stitching  
**B)** Computer vision at edge with existing cameras  
**C)** VR headsets

**Answer: B**

Panorama: run CV models on IP cameras at edge. Low-latency, on-premises processing. Use cases: retail analytics, worker safety, manufacturing quality. Panorama Appliance or SDK.

---

### Q25: What is Amazon Monitron?

**A)** System monitoring  
**B)** Equipment monitoring for predictive maintenance  
**C)** Network monitoring

**Answer: B**

Monitron: end-to-end system for equipment health. Sensors detect vibration, temperature. ML detects anomalies. Predicts failures before they occur. Use cases: motors, fans, pumps.

---

## Natural Language Processing (Q26-33)

### Q26: What is Amazon Comprehend?

**A)** Compression service  
**B)** NLP service for text analysis  
**C)** Understanding training

**Answer: B**

Comprehend: extract insights from text. Sentiment analysis, entity extraction, key phrases, language detection, topic modeling, PII detection. Pre-trained, no ML expertise needed.

---

### Q27: What is Comprehend Medical?

**A)** General medical app  
**B)** Extract medical information from clinical text  
**C)** Appointment scheduling

**Answer: B**

Comprehend Medical: NLP for healthcare. Extracts conditions, medications, dosages, treatments, protected health information (PHI). Understands medical terminology. HIPAA-eligible.

---

### Q28: What is Amazon Translate?

**A)** Code translation  
**B)** Neural machine translation service  
**C)** Audio translation

**Answer: B**

Translate: real-time text translation. 75+ languages. Neural MT for natural translations. Batch or real-time. Custom terminology for domain-specific terms. Formality control.

---

### Q29: What is Amazon Transcribe?

**A)** Code transcription  
**B)** Speech-to-text service  
**C)** Document transcription

**Answer: B**

Transcribe: convert audio to text. Real-time or batch. Speaker identification, custom vocabulary, automatic punctuation, timestamps. Medical and call analytics variants. Multiple languages.

---

### Q30: What is Transcribe Call Analytics?

**A)** Phone billing  
**B)** Analyze customer service calls  
**C)** Network analysis

**Answer: B**

Call Analytics: transcribe calls plus sentiment, categories, interruptions, talk speed. Post-call analytics for quality monitoring. Redact PII. Integrate with contact centers.

---

### Q31: What is Amazon Polly?

**A)** Polling service  
**B)** Text-to-speech service  
**C)** Survey tool

**Answer: B**

Polly: convert text to lifelike speech. Neural TTS for natural sound. 60+ voices, 30+ languages. SSML for pronunciation control. Newsreader style, conversational style.

---

### Q32: What is Amazon Kendra?

**A)** General search  
**B)** Intelligent enterprise search using ML  
**C)** E-commerce search

**Answer: B**

Kendra: ML-powered search for documents, FAQs, knowledge bases. Natural language queries. Understands context, ranks by relevance. Connectors for S3, SharePoint, databases. Document ranking.

---

### Q33: What is Amazon Lex?

**A)** Legal service  
**B)** Build conversational interfaces (chatbots)  
**C)** Text editor

**Answer: B**

Lex: create chatbots and voice bots. Automatic speech recognition (ASR), natural language understanding (NLU). Powers Alexa. Intents, slots, fulfillment. Multi-channel deployment.

---

## Forecasting & Recommendations (Q34-39)

### Q34: What is Amazon Forecast?

**A)** Weather prediction  
**B)** Time-series forecasting service  
**C)** Cost forecasting only

**Answer: B**

Forecast: ML for time-series predictions. Demand forecasting, inventory planning, resource allocation. Auto-selects algorithms (ARIMA, DeepAR, Prophet). Handles missing data, seasonality.

---

### Q35: What is Amazon Personalize?

**A)** User profiles  
**B)** Real-time personalization and recommendations  
**C)** Email personalization

**Answer: B**

Personalize: recommendation engine as a service. Collaborative filtering, deep learning. User-item interactions, metadata. Real-time recommendations, similar items, personalized rankings. No ML expertise.

---

### Q36: What are Personalize recipes?

**A)** Cooking instructions  
**B)** Pre-configured algorithms for different use cases  
**C)** Data formats

**Answer: B**

Recipes: algorithm templates for specific scenarios. User-Personalization (general), Similar-Items (related products), Personalized-Ranking (re-rank items). Choose based on use case.

---

### Q37: What is Amazon Fraud Detector?

**A)** General anomaly detection  
**B)** Managed fraud detection service  
**C)** Network security

**Answer: B**

Fraud Detector: identify suspicious activities. Payment fraud, account takeover, fake accounts. Uses your historical data + AWS fraud expertise. Real-time risk scoring. Rule-based + ML.

---

### Q38: What is Amazon DevOps Guru?

**A)** Developer training  
**B)** ML-powered operational insights  
**C)** Code review

**Answer: B**

DevOps Guru: detect operational issues using ML. Analyzes CloudWatch metrics, logs, events. Identifies anomalies, predicts problems, recommends fixes. Proactive issue detection for applications.

---

### Q39: What is Amazon Lookout for Metrics?

**A)** Dashboard service  
**B)** Automated anomaly detection in metrics  
**C)** KPI tracking only

**Answer: B**

Lookout for Metrics: detect anomalies in business metrics (revenue, users, transactions). Connects to S3, RDS, Redshift. ML-based anomaly detection. Diagnoses root causes across dimensions.

---

## Edge & IoT AI Services (Q40-44)

### Q40: What is AWS DeepLens?

**A)** Camera lens  
**B)** Deep learning-enabled video camera  
**C)** Magnification tool

**Answer: B**

DeepLens: camera with onboard GPU for running CV models at edge. Pre-trained models (object detection, face detection). Deploy custom SageMaker models. Learning device for CV applications.

---

### Q41: What is AWS DeepRacer?

**A)** Car racing game  
**B)** Autonomous race car for learning RL  
**C)** Speed testing

**Answer: B**

DeepRacer: 1/18th scale race car for reinforcement learning. Train models in simulator, deploy to physical car. Learn RL concepts hands-on. Competitions, leagues, virtual racing.

---

### Q42: What is AWS IoT Greengrass ML Inference?

**A)** Cloud ML only  
**B)** Run ML models on edge IoT devices  
**C)** Green energy

**Answer: B**

Greengrass ML: deploy SageMaker models to edge devices. Local inference without cloud latency. Supports TensorFlow, MXNet. Updates models OTA. Use cases: industrial, robotics, smart cameras.

---

### Q43: What is Amazon Lookout for Equipment?

**A)** Equipment shopping  
**B)** Predictive maintenance for industrial equipment  
**C)** Inventory management

**Answer: B**

Lookout for Equipment: detect abnormal equipment behavior. Sensor data analysis (vibration, pressure, temperature). ML predicts failures. Use cases: manufacturing, energy, utilities.

---

### Q44: What is AWS Panorama Appliance?

**A)** Camera system  
**B)** Hardware for running CV at edge  
**C)** Network appliance

**Answer: B**

Panorama Appliance: hardware device connecting to IP cameras. Runs multiple CV models simultaneously. Low-latency on-premises processing. Deploy custom SageMaker models. For retail, manufacturing.

---

## Additional AI Services (Q45-50)

### Q45: What is Amazon CodeGuru?

**A)** Teaching platform  
**B)** ML-powered code reviews and performance recommendations  
**C)** Code generator

**Answer: B**

CodeGuru: automated code reviews (Reviewer) and runtime performance analysis (Profiler). Detects bugs, security issues, inefficiencies. Java, Python. Reduces manual review time.

---

### Q46: What is Amazon HealthLake?

**A)** Fitness app  
**B)** Store, transform, analyze health data at scale  
**C)** Medical devices

**Answer: B**

HealthLake: HIPAA-eligible FHIR data store. Transforms unstructured medical data (notes, PDFs) into structured FHIR format. Integrates Comprehend Medical. Query, analyze health data at scale.

---

### Q47: What is AWS DeepComposer?

**A)** Video editing  
**B)** Music composition using ML  
**C)** Document composition

**Answer: B**

DeepComposer: create music with generative AI. Input melody, AI generates accompaniment. Learn generative models (GANs, Transformers, ARs). Keyboard hardware or virtual. Educational tool.

---

### Q48: What is Amazon Augmented AI (A2I)?

**A)** AR/VR service  
**B)** Human review workflows for ML predictions  
**C)** Training augmentation

**Answer: B**

A2I: build human review into ML workflows. Route low-confidence predictions to humans. Pre-built workflows for Textract, Rekognition. Custom workflows with SageMaker. Quality control, compliance.

---

### Q49: What is Amazon Omics?

**A)** General analytics  
**B)** Store and analyze genomic and biological data  
**C)** Business intelligence

**Answer: B**

Omics: purpose-built for genomics, transcriptomics, proteomics. Sequence stores, variant stores, annotation stores. Scalable analysis of biological data. Accelerates research, drug discovery.

---

### Q50: What is AWS HealthScribe?

**A)** Medical transcription  
**B)** Clinical documentation from patient-clinician conversations  
**C)** Prescription writing

**Answer: B**

HealthScribe: generate clinical notes from audio. Transcribes conversations, extracts medical terms, summarizes. SOAP notes format. HIPAA-eligible. Reduces documentation burden for clinicians.

---

## Exam Tips

**Key Concepts to Remember:**

1. **SageMaker Family:**

   - **Studio:** IDE for ML
   - **Autopilot:** AutoML
   - **Clarify:** Bias/explainability
   - **Ground Truth:** Data labeling
   - **Feature Store:** Feature management
   - **Pipelines:** ML workflows

2. **Bedrock Components:**

   - Foundation models via API
   - **Agents:** Multi-step task execution
   - **Knowledge Bases:** RAG implementation
   - **Guardrails:** Content filtering

3. **Computer Vision:**

   - **Rekognition:** Images/videos
   - **Textract:** Document OCR
   - **Lookout for Vision:** Defect detection
   - **Panorama:** Edge CV

4. **NLP Services:**

   - **Comprehend:** Text analysis
   - **Translate:** Language translation
   - **Transcribe:** Speech-to-text
   - **Polly:** Text-to-speech
   - **Lex:** Chatbots
   - **Kendra:** Intelligent search

5. **Business AI:**

   - **Forecast:** Time-series
   - **Personalize:** Recommendations
   - **Fraud Detector:** Fraud detection

6. **Service Selection:**
   - Custom models → SageMaker
   - Foundation models → Bedrock
   - Pre-trained specific task → AI service (Rekognition, Comprehend, etc.)

**Study Focus:**

- Match service to use case
- Know which services are fully managed vs require customization
- Understand service integrations (A2I + Textract, Neo + IoT Greengrass)
- Remember specialized variants (Comprehend Medical, Transcribe Call Analytics)
- Know HIPAA-eligible services (Comprehend Medical, HealthLake, HealthScribe)
