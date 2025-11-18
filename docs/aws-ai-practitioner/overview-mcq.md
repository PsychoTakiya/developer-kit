# AWS AI Practitioner - MCQ Practice

50 focused MCQs covering AWS AI/ML fundamentals, services, and architecture concepts.

---

## AWS AI/ML Basics (Q1-10)

### Q1: What is AWS AI/ML?

**A)** Only for data scientists  
**B)** Suite of AI/ML services for building intelligent applications  
**C)** Replacement for traditional programming

**Answer: B**

AWS AI/ML includes pre-trained services (Rekognition, Comprehend) and ML platforms (SageMaker, Bedrock). Enables developers to add AI without deep ML expertise. Pay-as-you-go, serverless options available.

---

### Q2: What distinguishes AI from ML?

**A)** AI always requires neural networks  
**B)** AI is broader concept, ML is subset that learns from data  
**C)** They are identical

**Answer: B**

AI: systems performing human-like tasks (can be rule-based). ML: systems learning patterns from data without explicit programming. Deep Learning: ML using multi-layered neural networks.

---

### Q3: Which AWS service requires NO ML expertise?

**A)** Amazon SageMaker  
**B)** Amazon Rekognition  
**C)** Custom neural networks

**Answer: B**

Pre-trained AI services (Rekognition, Polly, Transcribe, Translate, Comprehend, Textract) need no ML knowledge. SageMaker requires ML skills for custom models. Simple API calls for common tasks.

---

### Q4: What type of data does Deep Learning excel at?

**A)** Structured spreadsheets only  
**B)** Unstructured data like images, audio, text  
**C)** Small datasets

**Answer: B**

Deep Learning uses neural networks for unstructured data. CNNs for images/video, RNNs for sequences, Transformers for language. Requires large datasets and computational power. Best accuracy trade-off.

---

### Q5: When should you use pre-trained AI services?

**A)** Always for every use case  
**B)** Common tasks like translation, OCR, sentiment analysis  
**C)** Never, always build custom

**Answer: B**

Use pre-trained when: common use case, fast deployment needed, limited ML expertise. Use SageMaker when: custom business problem, proprietary data, need model control. Cost-effective starting point.

---

### Q6: What is Amazon Rekognition used for?

**A)** Text translation  
**B)** Image and video analysis  
**C)** Time-series forecasting

**Answer: B**

Rekognition: detect objects, faces, text in images/videos. Facial analysis, content moderation, celebrity recognition, PPE detection. Deep learning-based computer vision. No training required.

---

### Q7: What does Amazon Comprehend do?

**A)** Compress data  
**B)** Natural language processing and text analysis  
**C)** Image recognition

**Answer: B**

Comprehend: NLP for sentiment analysis, entity extraction, key phrases, language detection, topic modeling, PII detection. Pre-trained for common text tasks. No ML expertise needed.

---

### Q8: Which service converts speech to text?

**A)** Amazon Polly  
**B)** Amazon Transcribe  
**C)** Amazon Translate

**Answer: B**

Transcribe: speech-to-text conversion. Real-time or batch. Speaker identification, custom vocabulary, timestamps. Polly does opposite (text-to-speech). Medical and call analytics variants available.

---

### Q9: What is Amazon Textract?

**A)** Text generation  
**B)** Extract text and data from documents  
**C)** Text translation

**Answer: B**

Textract: OCR with structure understanding. Extracts text, tables, forms (key-value pairs). Works on PDFs, images. Identifies document layouts. Specialized AnalyzeExpense for invoices/receipts.

---

### Q10: What is the AI hierarchy from broadest to narrowest?

**A)** ML → DL → AI  
**B)** AI → ML → Deep Learning  
**C)** DL → AI → ML

**Answer: B**

AI (broadest): human-like intelligence, any approach. ML (subset): learns from data. Deep Learning (narrowest): multi-layer neural networks. Each level builds on previous.

---

## Amazon SageMaker (Q11-20)

### Q11: What is Amazon SageMaker?

**A)** Only for deployment  
**B)** Fully managed ML platform for build, train, deploy  
**C)** Pre-trained models only

**Answer: B**

SageMaker: end-to-end ML platform with notebooks, built-in algorithms, training jobs, hosting, monitoring. Scales automatically, managed infrastructure. For custom ML models and workflows.

---

### Q12: What is SageMaker Studio?

**A)** Music production  
**B)** Web-based IDE for ML workflows  
**C)** Training service only

**Answer: B**

Studio: integrated development environment for ML. Single interface for notebooks, experiments, pipelines, model registry. Collaborative workspace, no server setup. Visual workflow designer included.

---

### Q13: What is SageMaker Ground Truth?

**A)** Testing framework  
**B)** Data labeling service with human and ML workers  
**C)** Model validation

**Answer: B**

Ground Truth: build labeled datasets using human annotators or active learning. Reduces labeling costs up to 70%. Supports images, text, video. Built-in and custom workflows.

---

### Q14: What is SageMaker Autopilot?

**A)** Auto-deployment only  
**B)** AutoML that builds, trains, tunes models automatically  
**C)** Auto-scaling service

**Answer: B**

Autopilot: AutoML for tabular data. Automatically explores algorithms (XGBoost, linear, deep learning), tunes hyperparameters, generates explainable notebooks. No code required for baseline models.

---

### Q15: What does SageMaker Clarify do?

**A)** Data cleaning  
**B)** Bias detection and model explainability  
**C)** Performance optimization

**Answer: B**

Clarify: detect bias in data/models across demographics. Provides SHAP values for explainability. Monitors deployed models for drift. Supports fairness metrics, feature importance analysis.

---

### Q16: What is SageMaker Model Monitor?

**A)** Video monitoring  
**B)** Continuous monitoring of model quality in production  
**C)** Cost monitoring

**Answer: B**

Model Monitor: detects data drift, model drift, bias drift, feature attribution drift. Automatic baseline creation, scheduled monitoring, CloudWatch integration. Alerts on quality degradation.

---

### Q17: What is SageMaker Pipelines?

**A)** Data pipelines only  
**B)** CI/CD for ML workflows  
**C)** Network configuration

**Answer: B**

Pipelines: orchestrate ML workflows (data prep → training → evaluation → deployment). Automates retraining, versioning. JSON/Python SDK. Integrates with Model Registry for governance.

---

### Q18: What is SageMaker Feature Store?

**A)** App marketplace  
**B)** Centralized repository for ML features  
**C)** Model storage

**Answer: B**

Feature Store: manage, share, reuse features across teams. Online store (low-latency), offline store (training). Ensures training-serving consistency. Time-travel queries for point-in-time correctness.

---

### Q19: What is SageMaker Neo?

**A)** New version of SageMaker  
**B)** Optimize models for edge devices  
**C)** Neural network designer

**Answer: B**

Neo: compile models to run 2x faster on edge devices (IoT, mobile). Supports TensorFlow, PyTorch, MXNet. Optimizes for CPU, GPU, specific hardware. Deploy with IoT Greengrass.

---

### Q20: What is SageMaker Batch Transform?

**A)** Real-time predictions  
**B)** Offline batch predictions on large datasets  
**C)** Data transformation only

**Answer: B**

Batch Transform: run inference on S3 datasets without persistent endpoint. Cost-effective for periodic predictions (daily scoring). No infrastructure management, automatic scaling.

---

## Amazon Bedrock (Q21-28)

### Q21: What is Amazon Bedrock?

**A)** Database service  
**B)** Fully managed service for foundation models via API  
**C)** Infrastructure service

**Answer: B**

Bedrock: access foundation models (Claude, Llama, Titan, Jurassic) via API. No infrastructure management. Customize with fine-tuning, RAG. Serverless, pay-per-token pricing.

---

### Q22: What are Bedrock Agents?

**A)** Human support agents  
**B)** Autonomous agents executing multi-step tasks  
**C)** Monitoring agents

**Answer: B**

Agents: orchestrate foundation models to complete complex tasks. Use tools (APIs, Lambda), retrieve knowledge, reason through steps. Break user requests into actions automatically.

---

### Q23: What is Bedrock Knowledge Bases?

**A)** Training datasets  
**B)** RAG implementation with vector storage  
**C)** Documentation library

**Answer: B**

Knowledge Bases: implement RAG without managing infrastructure. Connect to S3, chunk documents, create embeddings, store in vector DB (OpenSearch, Pinecone). Retrieves context for prompts.

---

### Q24: What are Bedrock Guardrails?

**A)** Physical security  
**B)** Safety controls for content filtering  
**C)** Network firewall

**Answer: B**

Guardrails: filter harmful content, PII, sensitive topics. Denied topics, content filters, word filters. Apply to inputs and outputs. Configurable thresholds for toxicity, hate, violence.

---

### Q25: What is Bedrock Provisioned Throughput?

**A)** Free usage tier  
**B)** Reserved model capacity for consistent performance  
**C)** Network bandwidth

**Answer: B**

Provisioned Throughput: purchase reserved model units for predictable latency/throughput. Lower cost for high-volume use. Commitment-based vs on-demand pricing. Ideal for production workloads.

---

### Q26: When should you use Bedrock vs SageMaker?

**A)** Always use Bedrock  
**B)** Bedrock for generative AI, SageMaker for custom ML  
**C)** They're interchangeable

**Answer: B**

Bedrock: generative AI, quick deployment, no training data, foundation models. SageMaker: custom models, proprietary data, full control, traditional ML tasks. Can use both together.

---

### Q27: What is fine-tuning in Bedrock?

**A)** Prompt adjustment  
**B)** Customizing models with your data  
**C)** Performance optimization

**Answer: B**

Fine-tuning: adapt foundation models to your domain/style. Provide labeled examples, model trains on your data. Creates private custom model. Available for select models (Titan, Claude).

---

### Q28: What is Bedrock Playground?

**A)** Testing sandbox  
**B)** Interactive console to experiment with models  
**C)** Game platform

**Answer: B**

Playground: web interface to test prompts, compare models, adjust parameters (temperature, top-p). No code required. Export successful prompts to code. Quick experimentation before implementation.

---

## ML Workflow & Architecture (Q29-38)

### Q29: What percentage of ML effort goes to data preparation?

**A)** 20%  
**B)** 50%  
**C)** 80%

**Answer: C**

Data preparation (collection, cleaning, labeling, feature engineering) consumes 80% of ML effort. Quality data critical for model performance. Poor data = poor model regardless of algorithm.

---

### Q30: What is feature engineering?

**A)** Hardware design  
**B)** Creating meaningful input variables from raw data  
**C)** Model deployment

**Answer: B**

Feature engineering: transform raw data into useful model inputs. Examples: "days since last login" from timestamps, "total purchases" from transactions. Improves model performance significantly.

---

### Q31: What is hyperparameter tuning?

**A)** Tuning after deployment  
**B)** Optimizing model settings before training  
**C)** Data cleaning

**Answer: B**

Hyperparameter tuning: optimize learning rate, batch size, layers, regularization before training. Not learned from data, set manually or automatically (SageMaker Autopilot). Improves model accuracy.

---

### Q32: What is model drift?

**A)** Training errors  
**B)** Performance degradation due to data changes over time  
**C)** Hardware issues

**Answer: B**

Model drift: real-world data distributions change (COVID changing shopping patterns). Model trained on old patterns fails on new data. Requires monitoring and retraining periodically.

---

### Q33: What is the purpose of A/B testing in ML?

**A)** Code testing  
**B)** Comparing model versions in production with real traffic  
**C)** Testing two algorithms

**Answer: B**

A/B testing: route traffic to model A (champion) vs model B (challenger). Measure business metrics (conversion, revenue). Deploy winner. Reduces deployment risk with real data.

---

### Q34: What is RAG (Retrieval-Augmented Generation)?

**A)** Regular AI generation  
**B)** Combining retrieval from knowledge base with generation  
**C)** Random answer generation

**Answer: B**

RAG: retrieve relevant documents from knowledge base, pass as context to LLM for generation. Reduces hallucinations, grounds responses in facts. Bedrock Knowledge Bases implements this automatically.

---

### Q35: What is the purpose of VPC endpoints for AI services?

**A)** Public internet access  
**B)** Private connection to AWS services without internet  
**C)** VPN connection

**Answer: B**

VPC endpoints: access Bedrock, SageMaker privately within VPC. No internet exposure. Reduces attack surface, meets compliance (data doesn't leave private network). Interface or gateway endpoints.

---

### Q36: What is Step Functions used for in ML?

**A)** Walking tracking  
**B)** Orchestrating multi-step ML workflows  
**C)** Step-by-step tutorials

**Answer: B**

Step Functions: orchestrate ML pipelines (extract → classify → analyze → store). Handles errors, retries, parallel execution. Visual workflow designer. Coordinates multiple Lambda/AI services.

---

### Q37: What is the purpose of SageMaker Model Registry?

**A)** Domain registration  
**B)** Version control and governance for ML models  
**C)** User registration

**Answer: B**

Model Registry: track model versions (v1.0, v1.1), compare performance metrics, manage approvals (dev/staging/prod). Audit trail of deployments. Governance and reproducibility for teams.

---

### Q38: What is serverless inference in SageMaker?

**A)** No servers involved  
**B)** Pay-per-request model hosting without persistent endpoints  
**C)** Free hosting

**Answer: B**

Serverless inference: deploy models without provisioning instances. Pay only for compute time during predictions. Auto-scales to zero. Ideal for sporadic traffic, variable workloads. Reduces costs.

---

## Security & Best Practices (Q39-46)

### Q39: What is the shared responsibility model for AI?

**A)** AWS handles everything  
**B)** AWS secures infrastructure, customer secures data/models  
**C)** Customer handles everything

**Answer: B**

AWS: physical security, infrastructure, managed service security. Customer: data encryption, access control, model security, responsible AI. Clear boundary enables shared security accountability.

---

### Q40: What is IAM's role in AI security?

**A)** Model training  
**B)** Identity and access management for AI services  
**C)** Data storage

**Answer: B**

IAM: controls who can access SageMaker, Bedrock, AI services. Uses policies, roles, permissions. Principle of least privilege: grant minimum access needed. Prevents unauthorized model/data access.

---

### Q41: What is AWS KMS used for in AI?

**A)** Model deployment  
**B)** Encryption key management for data and models  
**C)** Performance monitoring

**Answer: B**

KMS: manages encryption keys for data at rest (S3, EBS), in transit (TLS). Encrypts training data, models, predictions. Supports customer-managed keys (CMK) for control.

---

### Q42: What should you monitor in production ML models?

**A)** Training progress only  
**B)** Drift, latency, accuracy, data quality  
**C)** Code quality

**Answer: B**

Monitor: data drift (input changes), model drift (accuracy degradation), prediction latency, data quality issues. SageMaker Model Monitor automates this. Set alerts, trigger retraining when needed.

---

### Q43: What is prompt injection in AI?

**A)** SQL injection  
**B)** Malicious prompts to bypass AI safety controls  
**C)** Network attack

**Answer: B**

Prompt injection: crafted inputs to make LLM ignore instructions, reveal system prompts, generate harmful content. Mitigate: input validation, Bedrock Guardrails, output filtering, monitoring.

---

### Q44: How do you handle PII in AI applications?

**A)** Ignore it  
**B)** Detect and redact before processing  
**C)** Store in plain text

**Answer: B**

Detect PII with Comprehend or Bedrock Guardrails. Redact before processing, logging, training. GDPR/HIPAA requirement. Use data masking for testing. Prevents data leaks and compliance violations.

---

### Q45: What is the purpose of CloudTrail for AI?

**A)** Model training logs  
**B)** Audit trail of API calls and user actions  
**C)** Application logs

**Answer: B**

CloudTrail: logs all AWS API calls (who accessed what, when, from where). Essential for compliance audits, security investigations, governance. Track SageMaker/Bedrock usage, model deployments.

---

### Q46: What is X-Ray used for in AI applications?

**A)** Image analysis  
**B)** Distributed tracing and debugging  
**C)** Security scanning

**Answer: B**

X-Ray: trace requests through AI application (API Gateway → Lambda → SageMaker → DynamoDB). Identify latency bottlenecks, errors. Performance optimization for complex workflows.

---

## Cost Optimization (Q47-50)

### Q47: What is Spot Instance for SageMaker training?

**A)** Premium instance  
**B)** Discounted compute (up to 90% savings) with interruption risk  
**C)** Dedicated instance

**Answer: B**

Spot Instances: use spare AWS capacity at steep discount (70-90% off). Can be interrupted with 2-min notice. Ideal for training jobs (can resume). Set max_wait > max_run.

---

### Q48: How does caching reduce AI costs?

**A)** Stores physical equipment  
**B)** Stores AI responses to avoid repeated API calls  
**C)** Clears memory

**Answer: B**

Cache Bedrock responses for repeated queries (ElastiCache, DynamoDB). Reduces token costs, improves latency. Set appropriate TTL (Time To Live). Most effective for common questions/prompts.

---

### Q49: What is S3 Intelligent-Tiering?

**A)** Manual data organization  
**B)** Automatic data movement to cheaper storage based on access  
**C)** Fast storage only

**Answer: B**

Intelligent-Tiering: automatically moves objects between access tiers based on usage. Frequent → Infrequent → Archive → Deep Archive. Reduces storage costs without performance impact. Ideal for ML datasets.

---

### Q50: When is batch processing more cost-effective?

**A)** Real-time critical applications  
**B)** Non-urgent, large-scale predictions  
**C)** Never

**Answer: B**

Batch processing: SageMaker Batch Transform for offline predictions. No persistent endpoint costs. Process large datasets periodically (daily scoring). Much cheaper than real-time for non-urgent workloads.

---

## Exam Tips

**Key Concepts to Remember:**

1. **Service Selection:**

   - Pre-trained AI → Common tasks, no ML expertise
   - SageMaker → Custom models, full control
   - Bedrock → Generative AI, foundation models

2. **AI Hierarchy:** AI (broadest) → ML (learns from data) → Deep Learning (neural networks)

3. **SageMaker Components:**

   - Studio (IDE), Autopilot (AutoML), Clarify (bias), Ground Truth (labeling)
   - Pipelines (CI/CD), Feature Store (features), Model Monitor (drift)

4. **Bedrock Features:**

   - Multiple foundation models via API
   - Agents (multi-step), Knowledge Bases (RAG), Guardrails (safety)

5. **Security:**

   - Shared responsibility model
   - IAM (access), KMS (encryption), VPC (network isolation)
   - CloudTrail (audit), Guardrails (content filtering)

6. **Cost Optimization:**
   - Spot Instances (training)
   - Serverless/Batch (inference)
   - Caching, Intelligent-Tiering

**Study Focus:**

- Match service to use case scenario
- Understand when to use Bedrock vs SageMaker
- Know security best practices (encryption, IAM, VPC)
- Cost optimization techniques for different workloads
- ML workflow steps and monitoring
