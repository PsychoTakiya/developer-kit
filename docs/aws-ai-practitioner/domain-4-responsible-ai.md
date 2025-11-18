# Domain 4: Guidelines for Responsible AI

30 focused MCQs for AWS AI Practitioner exam preparation.

---

## Foundational Concepts (Q1-10)

### Q1: What is the primary goal of responsible AI?

**A)** Maximize model accuracy  
**B)** Ensure AI systems are fair, transparent, and beneficial  
**C)** Reduce computational costs

**Answer: B**

Responsible AI ensures systems don't harm users, are transparent in decisions, and benefit society. Accuracy alone isn't sufficient if the model is biased or opaque.

---

### Q2: Which AWS service helps detect bias in ML models?

**A)** Amazon SageMaker Clarify  
**B)** Amazon Comprehend  
**C)** AWS CloudTrail

**Answer: A**

SageMaker Clarify analyzes datasets and models for bias, provides explainability reports using SHAP, and monitors for drift.

---

### Q3: What does model explainability mean?

**A)** Model runs faster  
**B)** Understanding why a model made specific predictions  
**C)** Model uses less memory

**Answer: B**

Explainability helps users understand decision logic, crucial for trust, debugging, and regulatory compliance (GDPR, FCRA).

---

### Q4: Which is NOT a key principle of responsible AI?

**A)** Fairness  
**B)** Profit maximization  
**C)** Accountability

**Answer: B**

Responsible AI prioritizes fairness, transparency, accountability, privacy, and safety—not profit at the expense of ethics.

---

### Q5: What is data bias?

**A)** Incorrect model predictions  
**B)** Training data doesn't represent real-world diversity  
**C)** Model runs on biased hardware

**Answer: B**

Data bias occurs when training data is unrepresentative (e.g., facial recognition trained only on one ethnicity), leading to unfair outcomes.

---

### Q6: Which regulation requires "right to explanation" for automated decisions?

**A)** HIPAA  
**B)** GDPR  
**C)** SOC 2

**Answer: B**

GDPR (Europe) mandates users can request explanations for significant automated decisions affecting them (loans, employment).

---

### Q7: What is algorithmic bias?

**A)** Hardware malfunction  
**B)** Algorithm favors certain outcomes regardless of data  
**C)** Slow model training

**Answer: B**

Algorithmic bias stems from model design choices (optimization, loss functions) that systematically favor certain groups or outcomes.

---

### Q8: Which metric measures fairness across demographic groups?

**A)** Accuracy  
**B)** Demographic parity  
**C)** Latency

**Answer: B**

Demographic parity ensures equal positive prediction rates across groups (e.g., loan approval rates equal for all ethnicities).

---

### Q9: What is model drift?

**A)** Model predictions change over time due to data changes  
**B)** Model loses accuracy during training  
**C)** Model uses more memory

**Answer: A**

Drift occurs when data distributions or relationships change (e.g., COVID-19 changing shopping patterns), degrading model performance.

---

### Q10: Which is a transparent (interpretable) model?

**A)** Deep neural network  
**B)** Decision tree  
**C)** Ensemble of 100 models

**Answer: B**

Decision trees show clear IF-THEN paths. Neural networks are "black boxes" requiring post-hoc explanation methods.

---

## Bias Detection & Mitigation (Q11-18)

### Q11: What causes label bias?

**A)** Insufficient training data  
**B)** Human annotators' prejudices in labels  
**C)** Hardware limitations

**Answer: B**

Label bias occurs when humans labeling data inject their biases (e.g., biased resume screeners labeling candidates).

---

### Q12: How can you reduce data collection bias?

**A)** Train longer  
**B)** Ensure diverse, representative training data  
**C)** Use more GPUs

**Answer: B**

Collect data across demographics, geographies, and scenarios. Oversample underrepresented groups if needed.

---

### Q13: What is aggregation bias?

**A)** Combining multiple models  
**B)** One model doesn't work well for diverse subgroups  
**C)** Data storage inefficiency

**Answer: B**

Aggregation bias: one-size-fits-all models fail for groups with different patterns (e.g., diabetes risk varies by ethnicity).

---

### Q14: Which technique helps explain individual predictions?

**A)** SHAP values  
**B)** Increasing training data  
**C)** Using faster hardware

**Answer: A**

SHAP (SHapley Additive exPlanations) shows each feature's contribution to specific predictions, enabling local explainability.

---

### Q15: What is a counterfactual explanation?

**A)** Explaining past predictions  
**B)** Showing what changes would alter the outcome  
**C)** Listing all model features

**Answer: B**

Counterfactuals answer: "If X changed to Y, prediction would be Z" (e.g., "If debt ratio dropped to 40%, loan approved").

---

### Q16: How does SageMaker Clarify help with bias?

**A)** Speeds up training  
**B)** Detects bias in data and models, provides explanations  
**C)** Reduces model size

**Answer: B**

Clarify analyzes pre-training data bias, post-training model bias, generates SHAP explanations, and monitors for drift.

---

### Q17: What is evaluation bias?

**A)** Test data doesn't match real-world deployment  
**B)** Model trains too slowly  
**C)** Using wrong loss function

**Answer: A**

Evaluation bias: testing on clean data but deploying in noisy environments (e.g., speech recognition tested on clear audio).

---

### Q18: Which regularization technique prevents overfitting?

**A)** L2 (Ridge) regularization  
**B)** Adding more features  
**C)** Removing test data

**Answer: A**

L2 penalizes large weights, smoothing the model. Prevents memorizing noise. L1 (Lasso) also works by forcing some weights to zero.

---

## Transparency & Governance (Q19-24)

### Q19: What should model documentation include?

**A)** Only accuracy metrics  
**B)** Training data, limitations, intended use, biases  
**C)** Just the model file

**Answer: B**

Complete documentation covers data sources, known limitations, fairness metrics, intended use cases, and potential biases.

---

### Q20: Why is model versioning important?

**A)** Faster inference  
**B)** Track changes, enable rollback, audit decisions  
**C)** Reduce storage costs

**Answer: B**

Versioning (e.g., SageMaker Model Registry) enables auditing which model made decisions, rollback if issues arise, and compliance.

---

### Q21: What is the purpose of A/B testing ML models?

**A)** Train models faster  
**B)** Compare new model against current in production  
**C)** Reduce bias

**Answer: B**

A/B testing routes traffic to champion (current) and challenger (new) models, comparing performance before full deployment.

---

### Q22: What is a human-in-the-loop (HITL) system?

**A)** Fully automated AI  
**B)** Humans review/override AI decisions  
**C)** Manual data entry

**Answer: B**

HITL systems have humans review uncertain predictions or all decisions in high-stakes scenarios (medical diagnosis, loan rejections).

---

### Q23: Which AWS service helps with model monitoring?

**A)** Amazon SageMaker Model Monitor  
**B)** Amazon S3  
**C)** AWS Lambda

**Answer: A**

Model Monitor tracks prediction quality, data drift, model drift, and bias over time, alerting when performance degrades.

---

### Q24: What is the purpose of model cards?

**A)** Store model files  
**B)** Document model details for transparency  
**C)** Improve accuracy

**Answer: B**

Model cards provide standardized documentation: intended use, training data, performance metrics, limitations, and fairness evaluations.

---

## Privacy & Security (Q25-27)

### Q25: What is differential privacy?

**A)** Encrypting model weights  
**B)** Adding noise to protect individual data privacy  
**C)** Using different models for different users

**Answer: B**

Differential privacy adds mathematical noise ensuring individual records can't be identified, even if attackers know the rest of the dataset.

---

### Q26: Why should you encrypt ML training data?

**A)** Improve accuracy  
**B)** Protect sensitive information from unauthorized access  
**C)** Speed up training

**Answer: B**

Encryption (at rest with KMS, in transit with TLS) protects PII, health records, financial data from breaches during storage/transfer.

---

### Q27: What is federated learning?

**A)** Training one large model  
**B)** Training models on decentralized data without sharing raw data  
**C)** Using multiple AWS regions

**Answer: B**

Federated learning trains models across devices/locations without centralizing data, preserving privacy (e.g., smartphone keyboard predictions).

---

## Practical Application (Q28-30)

### Q28: When should you use an interpretable model over a black box?

**A)** When accuracy is the only concern  
**B)** In regulated industries requiring explainability  
**C)** When training data is large

**Answer: B**

Use interpretable models (linear, decision trees) in finance, healthcare, legal domains where regulations demand transparency.

---

### Q29: What indicates a model is overfitting?

**A)** High training accuracy, low test accuracy  
**B)** Low training accuracy, low test accuracy  
**C)** Equal training and test accuracy

**Answer: A**

Overfitting: model memorizes training data (95%+ accuracy) but fails on new data (< 75% accuracy). Large train-test gap is the indicator.

---

### Q30: Which strategy helps ensure responsible AI deployment?

**A)** Deploy once and forget  
**B)** Continuous monitoring, retraining, bias audits  
**C)** Maximize model complexity

**Answer: B**

Responsible AI requires ongoing monitoring for drift/bias, regular retraining with new data, periodic fairness audits, and human oversight.

---

## Advanced Fairness & Ethics (Q31-38)

### Q31: What is disparate impact in AI?

**A)** Hardware performance difference  
**B)** AI system disproportionately harms specific groups  
**C)** Training speed variation

**Answer: B**

Disparate impact: policy/model appears neutral but disadvantages protected groups. Example: height requirement excludes women disproportionately. Test: 80% rule (selection rate < 80% = potential issue).

---

### Q32: What is the confusion matrix?

**A)** Training errors  
**B)** Table showing true positives, false positives, true negatives, false negatives  
**C)** Model architecture

**Answer: B**

Confusion matrix visualizes classification performance. Rows: actual, columns: predicted. Enables calculating precision, recall, accuracy, F1. Essential for fairness analysis across groups.

---

### Q33: What is equal opportunity fairness?

**A)** Same accuracy for all  
**B)** Equal true positive rates across groups  
**C)** Equal dataset size

**Answer: B**

Equal opportunity: qualified individuals from all groups have equal chance of positive outcome. TPR (recall) equal across demographics. Example: qualified loan applicants approved equally regardless of race.

---

### Q34: What is representational harm?

**A)** Underrepresentation in data  
**B)** AI reinforces stereotypes or demeaning portrayals  
**C)** Missing features

**Answer: B**

Representational harm: system reinforces stereotypes (e.g., "CEO" image search shows only men, translation adds gender bias). Damages dignity even without material harm. Mitigate with diverse training data.

---

### Q35: What is fairness through awareness?

**A)** Ignoring protected attributes  
**B)** Explicitly using protected attributes to ensure fair outcomes  
**C)** Random predictions

**Answer: B**

Fairness through awareness: include protected attributes to monitor/correct for bias. Paradox: ignoring gender/race can worsen bias. Better: track, measure, adjust for fairness.

---

### Q36: What is AI red teaming?

**A)** Using red colors in UI  
**B)** Deliberately testing AI for vulnerabilities, biases, harmful outputs  
**C)** Team structure

**Answer: B**

Red teaming: adversarial testing to find model weaknesses (jailbreaks, prompt injections, bias triggers, harmful outputs). Critical before production. AWS supports red teaming for Bedrock.

---

### Q37: What is contextual fairness?

**A)** Same rules everywhere  
**B)** Fairness requirements vary by domain/use case  
**C)** Context windows

**Answer: B**

Contextual fairness: appropriate fairness metric depends on domain. Medical: maximize true positives (recall). Advertising: equal exposure (demographic parity). No universal fairness definition.

---

### Q38: What is AI impact assessment?

**A)** Performance benchmarks  
**B)** Analyzing potential societal/ethical impacts before deployment  
**C)** Cost analysis

**Answer: B**

Impact assessment: evaluate AI's effects on stakeholders, society, environment. Identify risks (bias, job displacement, privacy). Plan mitigations. Required by EU AI Act.

---

## Privacy & Data Protection (Q39-44)

### Q39: What is data minimization?

**A)** Compressing data  
**B)** Collecting only necessary data for specific purpose  
**C)** Deleting all data

**Answer: B**

Data minimization (GDPR principle): collect minimum data needed. Reduces privacy risk, storage costs, compliance burden. Example: don't collect SSN if name/email sufficient.

---

### Q40: What is purpose limitation?

**A)** Limited features  
**B)** Using data only for stated purpose  
**C)** Time limits

**Answer: B**

Purpose limitation (GDPR): use data only for declared purpose. Can't repurpose without consent. Example: marketing data can't be used for credit decisions without new consent.

---

### Q41: What is anonymization vs pseudonymization?

**A)** Same thing  
**B)** Anonymization irreversible, pseudonymization reversible with key  
**C)** Both fully reversible

**Answer: B**

Anonymization: remove identifiers permanently (can't re-identify). Pseudonymization: replace identifiers with codes (can reverse with key). GDPR encourages pseudonymization for flexibility.

---

### Q42: What is homomorphic encryption?

**A)** Standard encryption  
**B)** Computing on encrypted data without decrypting  
**C)** Password hashing

**Answer: B**

Homomorphic encryption: perform calculations on encrypted data, get encrypted result. Decrypt final result. Enables privacy-preserving ML. Computationally expensive but improving.

---

### Q43: What is the right to erasure (right to be forgotten)?

**A)** Backup deletion  
**B)** Users can request deletion of their personal data  
**C)** Cache clearing

**Answer: B**

Right to erasure (GDPR Article 17): individuals can request deletion of personal data. Exceptions: legal obligations, public interest. Design systems to support deletion (data lineage tracking).

---

### Q44: What is consent management in AI?

**A)** User agreements  
**B)** Tracking and honoring user data processing permissions  
**C)** Admin permissions

**Answer: B**

Consent management: record user consent for data use, respect preferences, allow withdrawal. Required for GDPR compliance. Granular consent (specific purposes) better than blanket.

---

## Governance & Compliance (Q45-50)

### Q45: What is AI governance framework?

**A)** Software framework  
**B)** Policies, processes, roles for responsible AI development  
**C)** Model architecture

**Answer: B**

AI governance: establishes accountability, decision rights, risk management, ethics review. Includes: AI ethics board, review processes, standards, training. Enterprise-wide structure.

---

### Q46: What is algorithmic accountability?

**A)** Accounting software  
**B)** Ability to explain and justify AI decisions, assign responsibility  
**C)** Financial audits

**Answer: B**

Algorithmic accountability: organizations responsible for AI outcomes. Must explain decisions, remediate harms, assign ownership. Requires documentation, monitoring, clear governance.

---

### Q47: What is AI ethics review board?

**A)** Software testing team  
**B)** Cross-functional group evaluating AI projects for ethical risks  
**C)** External auditors

**Answer: B**

Ethics board: diverse team (legal, ethics, tech, domain experts) reviewing AI projects. Approve, require changes, or reject based on risk assessment. Gate before production.

---

### Q48: What is responsible disclosure?

**A)** Financial reporting  
**B)** Communicating AI limitations and risks to users  
**C)** Security vulnerabilities

**Answer: B**

Responsible disclosure: inform users about AI capabilities, limitations, risks, data use. Transparency builds trust. Include in UI, docs, terms. Example: "AI-generated, may contain errors".

---

### Q49: What is algorithmic recourse?

**A)** Algorithm updates  
**B)** Users can challenge/appeal AI decisions  
**C)** Training retries

**Answer: B**

Algorithmic recourse: provide mechanism to contest AI decisions. Users should know how to appeal, path to human review. Required for fairness. Example: loan denial appeals process.

---

### Q50: What is continuous ethical monitoring?

**A)** One-time audit  
**B)** Ongoing tracking of fairness, bias, societal impact  
**C)** Security monitoring only

**Answer: B**

Continuous monitoring: regularly assess bias metrics, user complaints, societal changes affecting fairness. Not one-time—iterative. SageMaker Model Monitor supports ongoing bias tracking.

---

## Exam Tips

**Key Concepts to Remember:**

1. **Bias Types:** Data collection, algorithmic, label, measurement, aggregation, evaluation
2. **AWS Tools:** SageMaker Clarify (bias/explainability), Model Monitor (drift), Model Registry (versioning)
3. **Explainability:** SHAP (feature contributions), LIME (local approximations), counterfactuals (what-if)
4. **Regulations:** GDPR (right to explanation), FCRA (credit decisions), HIPAA (healthcare)
5. **Fairness Metrics:** Demographic parity, equal opportunity, predictive parity
6. **Privacy:** Differential privacy (noise), encryption (KMS), federated learning (decentralized)
7. **Model Issues:** Overfitting (memorization), underfitting (too simple), drift (data changes)
8. **Governance:** Documentation, versioning, A/B testing, HITL, model cards

**Study Focus:**

- Identify bias types from scenarios
- Match AWS services to use cases
- Understand when interpretability is required
- Know mitigation strategies for each bias type
