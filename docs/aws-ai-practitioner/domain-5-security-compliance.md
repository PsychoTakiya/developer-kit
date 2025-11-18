# Domain 5: Security, Compliance and Governance for AI Solutions

30 focused MCQs for AWS AI Practitioner exam preparation.

---

## AI Security Fundamentals (Q1-8)

### Q1: What is the shared responsibility model for AI on AWS?

**A)** AWS handles everything  
**B)** AWS secures infrastructure, customer secures data and models  
**C)** Customer handles everything

**Answer: B**

AWS: physical security, infrastructure, managed service security. Customer: data encryption, access control, model security, responsible AI practices. Clear boundary enables shared security.

---

### Q2: What is AWS IAM's role in AI security?

**A)** Data storage  
**B)** Identity and access management for AI services  
**C)** Model training

**Answer: B**

IAM controls who can access SageMaker, Bedrock, AI services. Uses policies, roles, permissions. Principle of least privilege: grant minimum access needed. Prevents unauthorized model/data access.

---

### Q3: What is AWS KMS used for in AI?

**A)** Model deployment  
**B)** Encryption key management for data and models  
**C)** Performance monitoring

**Answer: B**

KMS manages encryption keys for data at rest (S3, EBS), in transit (TLS). Encrypts training data, models, predictions. Supports customer-managed keys (CMK) for control.

---

### Q4: What is VPC endpoint for AI services?

**A)** Public internet access  
**B)** Private connection to AWS services without internet  
**C)** VPN connection

**Answer: B**

VPC endpoints: access Bedrock, SageMaker privately within VPC. No internet exposure. Reduces attack surface, meets compliance (data doesn't leave private network). Interface or gateway endpoints.

---

### Q5: What is AWS CloudTrail for AI governance?

**A)** Model training logs  
**B)** Audit trail of API calls and user actions  
**C)** Error logs

**Answer: B**

CloudTrail logs all AWS API calls: who accessed what, when, from where. Essential for compliance audits, security investigations, governance. Track SageMaker/Bedrock usage, model deployments.

---

### Q6: What is data encryption in transit?

**A)** Storing encrypted data  
**B)** Encrypting data while moving between locations  
**C)** Deleting data

**Answer: B**

Encryption in transit: TLS/SSL protects data moving between client-service, service-service. Prevents interception. AWS AI services use TLS 1.2+ by default. Critical for sensitive data.

---

### Q7: What is model versioning for security?

**A)** Backup copies  
**B)** Tracking model changes for audit and rollback  
**C)** Performance optimization

**Answer: B**

Model versioning (SageMaker Model Registry): track which model version deployed when, by whom. Enables rollback if issues, compliance audits, reproducibility. Essential for governance.

---

### Q8: What is AWS PrivateLink?

**A)** Private networking between VPCs and AWS services  
**B)** Public DNS service  
**C)** VPN service

**Answer: A**

PrivateLink: secure private connectivity to AWS services (Bedrock, SageMaker) without internet, NAT, gateways. Traffic stays on AWS network. Higher security, better performance.

---

## Data Protection & Privacy (Q9-15)

### Q9: What is data residency in AI?

**A)** Data storage duration  
**B)** Geographic location where data is stored/processed  
**C)** Data format

**Answer: B**

Data residency: keeping data in specific regions for legal/compliance reasons (GDPR, data sovereignty). Choose AWS regions carefully. Some Bedrock models available in specific regions only.

---

### Q10: What is PII detection in AI?

**A)** Performance monitoring  
**B)** Identifying personally identifiable information in data  
**C)** Image recognition

**Answer: B**

PII: names, SSN, emails, addresses. Detect with Comprehend, Bedrock Guardrails. Redact before processing, logging, or training. GDPR/HIPAA requirement. Prevents data leaks.

---

### Q11: What is data masking?

**A)** Data deletion  
**B)** Hiding sensitive data with modified values  
**C)** Data compression

**Answer: B**

Data masking: replace sensitive values with fake but realistic data (e.g., real SSN → fake SSN format). Use for testing, development. Preserves data utility while protecting privacy.

---

### Q12: What is Amazon Macie?

**A)** ML training service  
**B)** Automated data security and privacy service  
**C)** Image recognition

**Answer: B**

Macie uses ML to discover, classify, protect sensitive data in S3. Finds PII, credentials, financial data. Alerts on exposure risks. Helps GDPR/HIPAA compliance.

---

### Q13: What is data lifecycle management?

**A)** Active directory management  
**B)** Policies for data creation, storage, archival, deletion  
**C)** Network lifecycle

**Answer: B**

Lifecycle policies: move old training data to cheaper storage (S3 Glacier), delete after retention period. Reduces costs, meets compliance (data retention limits). Automate with S3 Lifecycle.

---

### Q14: What is AWS Secrets Manager for AI?

**A)** Model encryption  
**B)** Securely storing API keys, credentials, secrets  
**C)** Data anonymization

**Answer: B**

Secrets Manager: store database passwords, API keys, tokens. Rotate automatically. Use in SageMaker, Lambda for AI apps. Don't hardcode secrets in code/notebooks.

---

### Q15: What is Amazon GuardDuty?

**A)** Access control service  
**B)** Threat detection service monitoring malicious activity  
**C)** Firewall service

**Answer: B**

GuardDuty: continuous threat detection using ML. Monitors AWS accounts, workloads, data access. Alerts on unauthorized access, compromised instances, suspicious API calls. Security layer for AI infrastructure.

---

## Compliance & Regulations (Q16-22)

### Q16: What is HIPAA compliance in AI?

**A)** General security standard  
**B)** US healthcare data protection regulation  
**C)** European privacy law

**Answer: B**

HIPAA: protects patient health information (PHI). AI systems handling health data must use HIPAA-eligible services (SageMaker, Bedrock with BAA), encrypt data, audit access, limit retention.

---

### Q17: What is a Business Associate Agreement (BAA)?

**A)** Partnership contract  
**B)** Contract for HIPAA compliance with service providers  
**C)** Employee agreement

**Answer: B**

BAA: legal contract with AWS for HIPAA-covered accounts. Required before processing PHI. Not all AWS services HIPAA-eligible—check documentation. SageMaker, Bedrock support BAA.

---

### Q18: What is SOC 2 compliance?

**A)** European standard  
**B)** Trust service criteria for security, availability, confidentiality  
**C)** Healthcare standard

**Answer: B**

SOC 2: audits security controls across five trust principles (security, availability, processing integrity, confidentiality, privacy). AWS AI services SOC 2 compliant. Demonstrates operational security.

---

### Q19: What is data localization?

**A)** Data compression  
**B)** Keeping data within specific country/region borders  
**C)** Data translation

**Answer: B**

Data localization laws: some countries require data stored/processed within borders (China, Russia, EU). Choose AWS regions accordingly. Affects Bedrock model availability, SageMaker deployment.

---

### Q20: What is the GDPR right to access?

**A)** Service access  
**B)** Individuals can request copy of their personal data  
**C)** Admin access

**Answer: B**

Right to access (GDPR Article 15): individuals request what personal data you hold. AI systems must support data retrieval, export. Design data architecture for searchability by individual.

---

### Q21: What is ISO 27001?

**A)** Healthcare standard  
**B)** International information security management standard  
**C)** Payment card standard

**Answer: B**

ISO 27001: globally recognized security standard. AWS AI services certified. Covers risk management, security controls, continuous improvement. Demonstrates systematic security approach.

---

### Q22: What is PCI DSS for AI?

**A)** General security  
**B)** Payment card data security standard  
**C)** Privacy regulation

**Answer: B**

PCI DSS: protects credit card data. AI systems processing payments must comply: encrypt cardholder data, restrict access, monitor networks. AWS offers PCI DSS compliant services.

---

## Model Security (Q23-27)

### Q23: What is model theft/extraction attack?

**A)** Physical theft  
**B)** Stealing model weights through repeated queries  
**C)** Data breach

**Answer: B**

Model extraction: attacker queries model repeatedly to reverse-engineer it. Mitigate: rate limiting, query monitoring, detect unusual patterns, use authentication. Protect IP.

---

### Q24: What is prompt injection attack?

**A)** SQL injection  
**B)** Malicious prompts to bypass AI safety controls  
**C)** Network attack

**Answer: B**

Prompt injection: crafted inputs to make LLM ignore instructions, reveal system prompts, generate harmful content. Mitigate: input validation, Bedrock Guardrails, output filtering, monitoring.

---

### Q25: What is model poisoning?

**A)** Corrupting training data to compromise model  
**B)** Slow training  
**C)** Overfitting

**Answer: A**

Model poisoning: inject malicious data during training to create backdoors or bias. Mitigate: validate training data sources, monitor data quality, secure data pipeline, limit data sources.

---

### Q26: What is adversarial attack on AI?

**A)** DDoS attack  
**B)** Inputs designed to fool model predictions  
**C)** Password attack

**Answer: B**

Adversarial examples: slightly modified inputs that cause wrong predictions (e.g., image with imperceptible noise misclassified). Mitigate: adversarial training, input validation, ensemble models.

---

### Q27: What is model watermarking?

**A)** Data labeling  
**B)** Embedding identifiers in models to prove ownership  
**C)** Image watermarking

**Answer: B**

Model watermarking: embed unique patterns in model weights proving ownership. Detect unauthorized use. Emerging technique for IP protection. Some foundation model providers use this.

---

## Governance & Best Practices (Q28-30)

### Q28: What is separation of duties in AI?

**A)** Team organization  
**B)** Different people handle training, deployment, approval  
**C)** Work schedules

**Answer: B**

Separation of duties: prevent single person controlling entire ML pipeline. Example: data scientist trains, MLOps deploys, manager approves. Reduces fraud, errors, unauthorized changes.

---

### Q29: What is AWS Config for AI governance?

**A)** Configuration files  
**B)** Service tracking resource configurations and compliance  
**C)** Network configuration

**Answer: B**

AWS Config: tracks resource configurations (SageMaker endpoints, S3 buckets), evaluates compliance rules, alerts on violations. Ensures governance policies enforced. History for audits.

---

### Q30: What is incident response plan for AI?

**A)** User support  
**B)** Procedures for handling security breaches, model failures  
**C)** Development workflow

**Answer: B**

Incident response: documented plan for AI security incidents (data breach, model poisoning, bias detected). Include: detection, containment, remediation, communication, post-mortem. Test regularly.

---

## Advanced Security Controls (Q31-37)

### Q31: What is AWS Shield for AI applications?

**A)** Data encryption service  
**B)** DDoS protection service  
**C)** Access control service

**Answer: B**

Shield: protects against distributed denial-of-service (DDoS) attacks. Standard (free) for all, Advanced (paid) for SageMaker endpoints, API Gateway. Prevents availability disruptions.

---

### Q32: What is AWS WAF for AI?

**A)** Firewall for web applications and APIs  
**B)** Network firewall  
**C)** Database firewall

**Answer: A**

WAF: web application firewall filtering HTTP/HTTPS traffic. Protect API Gateway, SageMaker endpoints from SQL injection, XSS, rate-based attacks. Create custom rules for prompt injection patterns.

---

### Q33: What is AWS Certificate Manager (ACM)?

**A)** Certification training  
**B)** SSL/TLS certificate provisioning and management  
**C)** Compliance certification

**Answer: B**

ACM: provision, manage, deploy SSL/TLS certificates for encryption in transit. Free for AWS services. Auto-renewal prevents expiration. Use with custom domain endpoints for SageMaker, API Gateway.

---

### Q34: What is multi-factor authentication (MFA) for AI?

**A)** Multiple users  
**B)** Additional authentication factor beyond password  
**C)** Multiple models

**Answer: B**

MFA: require second factor (token, SMS, biometric) for AWS account access. Critical for root account, privileged IAM users. Prevents credential theft attacks. Hardware or virtual MFA devices.

---

### Q35: What is AWS Organizations for AI governance?

**A)** Team management  
**B)** Central management of multiple AWS accounts  
**C)** Data organization

**Answer: B**

Organizations: manage multiple AWS accounts centrally. Apply service control policies (SCPs) across accounts. Separate dev/test/prod environments. Consolidated billing. Enforce security baselines for AI workloads.

---

### Q36: What is AWS Security Hub?

**A)** Training portal  
**B)** Centralized security and compliance view across AWS  
**C)** Network hub

**Answer: B**

Security Hub: aggregates security findings from GuardDuty, Macie, Inspector, Config. Provides compliance scores (CIS, PCI DSS). Single dashboard for AI infrastructure security posture.

---

### Q37: What is AWS Systems Manager for AI?

**A)** ML system design  
**B)** Operational management for AWS resources  
**C)** System architecture

**Answer: B**

Systems Manager: operational data management, automation. Parameter Store (secure config), Session Manager (secure access), Patch Manager. Manage SageMaker instances, store model configs securely.

---

## Audit & Monitoring (Q38-43)

### Q38: What is AWS CloudWatch for AI monitoring?

**A)** Time tracking  
**B)** Metrics, logs, alarms for AWS resources  
**C)** Code review

**Answer: B**

CloudWatch: monitor SageMaker endpoint latency, invocations, errors. Create alarms for anomalies. Collect logs from Lambda, Bedrock. Dashboards for real-time visibility. Essential for production AI.

---

### Q39: What is log retention policy?

**A)** Employee retention  
**B)** Duration to keep logs before deletion  
**C)** Backup frequency

**Answer: B**

Log retention: comply with regulations (SOC 2: 90 days, PCI DSS: 1 year, HIPAA: 6 years). Configure CloudWatch Logs, S3 lifecycle policies. Balance compliance costs with legal requirements.

---

### Q40: What is AWS Audit Manager?

**A)** Team audits  
**B)** Automated evidence collection for compliance audits  
**C)** Financial audits

**Answer: B**

Audit Manager: continuously collect evidence for compliance (GDPR, HIPAA, SOC 2). Pre-built frameworks. Maps AWS usage to compliance controls. Reduces manual audit preparation time.

---

### Q41: What is anomaly detection in AI monitoring?

**A)** Bug detection  
**B)** Identifying unusual patterns in metrics or behavior  
**C)** Code analysis

**Answer: B**

Anomaly detection: CloudWatch, GuardDuty use ML to detect unusual API calls, traffic patterns, model predictions. Alerts on potential security incidents or model drift. Baseline normal behavior.

---

### Q42: What is AWS X-Ray for AI applications?

**A)** Image analysis  
**B)** Distributed tracing and debugging service  
**C)** Security scanning

**Answer: B**

X-Ray: trace requests through AI application components (API Gateway → Lambda → SageMaker → DynamoDB). Identify latency bottlenecks, errors. Performance optimization and debugging for complex workflows.

---

### Q43: What is immutable audit trail?

**A)** Unchangeable record of events  
**B)** Temporary logs  
**C)** Editable history

**Answer: A**

Immutable trail: CloudTrail logs to S3 with Object Lock, MFA delete. Prevents tampering with audit evidence. Required for high-compliance environments (financial, healthcare). Proves chain of custody.

---

## Data Security & Access (Q44-50)

### Q44: What is AWS Lake Formation for AI data?

**A)** Lake creation  
**B)** Secure data lake with fine-grained access control  
**C)** Water management

**Answer: B**

Lake Formation: centralized governance for data lakes. Column-level security, row-level filtering. Integrate with SageMaker for secure training data access. Simplifies data access management at scale.

---

### Q45: What is data classification?

**A)** ML classification  
**B)** Labeling data by sensitivity level  
**C)** Organizing folders

**Answer: B**

Data classification: label as Public, Internal, Confidential, Restricted. Different security controls per level. Macie auto-classifies S3 data. Drives encryption, access, retention policies.

---

### Q46: What is AWS Glue DataBrew for data security?

**A)** Data visualization  
**B)** Visual data preparation with PII redaction  
**C)** Database service

**Answer: B**

DataBrew: no-code data preparation. Detect and redact PII, mask sensitive data. Profile data quality. Prepare clean training data for SageMaker. Visual interface for data engineers.

---

### Q47: What is resource tagging for AI governance?

**A)** Metadata labels for AWS resources  
**B)** Code comments  
**C)** Image tagging

**Answer: A**

Tags: key-value pairs (Environment=Production, Project=Fraud-Detection, Owner=DataScience). Track costs, enforce policies, organize resources. Required tags via AWS Organizations SCPs. Critical for multi-account governance.

---

### Q48: What is least privilege principle?

**A)** Minimum cost  
**B)** Grant only minimum permissions needed  
**C)** Fewest users

**Answer: B**

Least privilege: IAM policies with minimum necessary permissions. Start with deny-all, add specific allows. Regular access reviews. Prevents lateral movement in breaches. SageMaker execution roles should be scoped.

---

### Q49: What is AWS PrivateLink for third-party AI services?

**A)** Public API access  
**B)** Private connectivity to SaaS providers  
**C)** VPN service

**Answer: B**

PrivateLink endpoints: access third-party AI services (model providers, data vendors) privately. No internet exposure. Provider provisions endpoint in your VPC. Secure integration without public IPs.

---

### Q50: What is break-glass access?

**A)** Emergency access procedure  
**B)** Security breach  
**C)** Permanent access

**Answer: A**

Break-glass: emergency admin access for critical incidents. Highly audited, time-limited, require multiple approvals. Example: CloudTrail readonly role for security investigations. Document all break-glass usage for compliance.

---

## Exam Tips

**Key Concepts to Remember:**

1. **Shared Responsibility:** AWS = infrastructure, Customer = data/models/responsible AI
2. **Access Control:** IAM (who), policies (what), roles (temp credentials), least privilege
3. **Encryption:** At rest (KMS), in transit (TLS), customer-managed keys (CMK)
4. **Network Security:** VPC endpoints (private), PrivateLink (secure), no internet exposure
5. **Compliance Standards:**
   - **GDPR:** European privacy, right to explanation/erasure
   - **HIPAA:** US healthcare, BAA required
   - **SOC 2:** Security/availability audits
   - **PCI DSS:** Payment card data
6. **Data Protection:** PII detection (Comprehend, Guardrails), masking, Macie (discovery)
7. **AI-Specific Threats:**
   - Prompt injection (malicious prompts)
   - Model extraction (query-based theft)
   - Model poisoning (training data corruption)
   - Adversarial attacks (fooling predictions)
8. **Governance:** CloudTrail (audit), Config (compliance), versioning (SageMaker Registry)
9. **Services Map:**
   - **Security:** IAM, KMS, VPC, PrivateLink, Secrets Manager
   - **Monitoring:** CloudTrail, GuardDuty, Macie, CloudWatch
   - **Compliance:** Config, AWS Artifact (compliance reports)

**Study Focus:**

- Identify appropriate security control for scenario
- Match compliance requirement to regulation
- Recognize AI-specific attack types
- Understand encryption: when KMS vs TLS
- Know which services support HIPAA/PCI DSS
- Governance best practices (separation of duties, versioning, audit trails)
