# Domain 2: Fundamentals of Generative AI

110 focused MCQs for AWS AI Practitioner exam preparation.

---

## Task 2.1: Basic Concepts of Generative AI (Q1-25)

### Q1: What is Generative AI?

**A)** AI that only classifies data  
**B)** AI that creates new content (text, images, code)  
**C)** AI for data analysis only

**Answer: B**

Generative AI creates new content rather than just analyzing existing data. Learns patterns from training data to generate novel outputs: text (GPT), images (Stable Diffusion), code (CodeWhisperer).

---

### Q2: How does Generative AI differ from traditional ML?

**A)** Uses smaller models  
**B)** Creates content vs predicts/classifies  
**C)** Requires less data

**Answer: B**

Traditional ML: predict, classify, analyze (spam filter, fraud detection). Generative AI: create, generate, synthesize (chatbots, image generation). GenAI uses foundation models with billions of parameters.

---

### Q3: What is a foundation model?

**A)** Small task-specific model  
**B)** Large pre-trained model adaptable to many tasks  
**C)** Rule-based system

**Answer: B**

Foundation models: billions of parameters, trained on massive diverse datasets, multi-task capable. Can be adapted via prompting or fine-tuning without retraining from scratch (GPT, Claude, Llama).

---

### Q4: Which AWS service provides access to foundation models?

**A)** Amazon SageMaker  
**B)** Amazon Bedrock  
**C)** Amazon Rekognition

**Answer: B**

Bedrock offers API access to foundation models (Claude, Titan, Llama, Stable Diffusion) without managing infrastructure. Pay-per-use, no model training required. SageMaker is for custom models.

---

### Q5: What architecture powers most modern LLMs?

**A)** Decision trees  
**B)** Transformers with self-attention  
**C)** Linear regression

**Answer: B**

Transformers use self-attention to understand context and relationships. Process all tokens simultaneously (parallelizable). Foundation of GPT, Claude, BERT. Enables long-range dependencies in text.

---

### Q6: What is Amazon Titan?

**A)** Third-party model only  
**B)** AWS-developed foundation models  
**C)** Hardware accelerator

**Answer: B**

Titan: Amazon's foundation model family. Titan Text (generation), Titan Image (image generation/editing), Titan Embeddings (vector representations). Cost-effective, AWS-native integration.

---

### Q7: What is Amazon CodeWhisperer?

**A)** Code deployment tool  
**B)** AI coding companion with real-time suggestions  
**C)** Code repository

**Answer: B**

CodeWhisperer provides AI-powered code suggestions, completions, security scanning. Trained on billions of lines of code. Supports multiple languages. Integrated with IDEs.

---

### Q8: What does "emergent capabilities" mean in foundation models?

**A)** Planned features  
**B)** Capabilities arising from scale not explicitly programmed  
**C)** Hardware improvements

**Answer: B**

As models scale (more parameters, data), new abilities emerge: reasoning, chain-of-thought, few-shot learning. Not explicitly trained for these tasks, they arise from scale and complexity.

---

### Q9: What is the purpose of RLHF in foundation models?

**A)** Speed up training  
**B)** Align model outputs with human preferences  
**C)** Reduce model size

**Answer: B**

RLHF (Reinforcement Learning from Human Feedback): trains models to be helpful, harmless, honest. Humans rate outputs, model learns preferences. Critical for safe, useful AI assistants.

---

### Q10: Which model type is best for text-to-image generation?

**A)** BERT  
**B)** GPT-4  
**C)** Stable Diffusion

**Answer: C**

Stable Diffusion, DALL-E, Titan Image generate images from text prompts. Diffusion models iteratively denoise random noise guided by text. GPT focuses on text, BERT on understanding.

---

### Q11: What are tokens in the context of language models?

**A)** Security credentials  
**C)** Smallest units of text that models process (words, subwords, characters)  
**B)** API keys

**Answer: C**

Tokens: text chunks models read/write. English: ~1 token = 4 characters or 0.75 words. "Hello world" = 2 tokens. Models have token limits (context window). More tokens = higher cost.

---

### Q12: What is tokenization?

**A)** Encryption  
**B)** Converting text into tokens that models can process  
**C)** User authentication

**Answer: B**

Tokenization: breaking text into tokens using specific algorithm (e.g., BPE, WordPiece). Each model has its own tokenizer. Important for understanding context limits and pricing.

---

### Q13: What are multi-modal foundation models?

**A)** Models with multiple layers  
**C)** Models that process multiple data types (text, image, audio, video)  
**B)** Models with multiple parameters

**Answer: C**

Multi-modal: handle text + images + audio/video. Examples: Claude 3 (text + images), GPT-4V (vision), Gemini. Enables tasks like image captioning, visual Q&A, document analysis with images.

---

### Q14: What is the attention mechanism in transformers?

**A)** User focus tracking  
**B)** Mechanism that weighs importance of different input tokens  
**C)** Memory optimization

**Answer: B**

Attention: allows model to focus on relevant parts of input. Self-attention computes relationships between all tokens. "The cat sat on the mat" - model learns "it" refers to "cat". Core of transformer architecture.

---

### Q15: What is a use case for video generation with GenAI?

**A)** Database queries  
**C)** Creating marketing videos, training content, personalized video messages  
**B)** Data analysis

**Answer: C**

Video generation: create videos from text prompts or images. Use cases: product demos, educational content, social media clips, personalized video ads. AWS partnerships with models like Runway.

---

### Q16: What is a use case for audio generation with GenAI?

**A)** Image compression  
**B)** Creating voices for audiobooks, podcasts, voice assistants, music  
**C)** Text analysis

**Answer: B**

Audio generation: text-to-speech, music creation, voice cloning. Use cases: accessibility (audiobooks), customer service (IVR), content creation (podcasts), personalized voice messages. Example: Amazon Polly for TTS.

---

### Q17: How does GenAI enable language translation?

**A)** Dictionary lookup  
**C)** Understanding context and generating natural translations preserving meaning  
**B)** Word replacement

**Answer: C**

LLMs understand linguistic nuances, idioms, context. Better than traditional translation: handles slang, maintains tone, considers cultural context. Use cases: real-time chat translation, document localization, multilingual support.

---

### Q18: What is a recommendation engine use case for GenAI?

**A)** Load balancing  
**B)** Generating personalized product/content recommendations with explanations  
**C)** Network routing

**Answer: B**

GenAI enhances recommendations: generates natural language explanations ("Based on your interest in sci-fi..."), creates personalized descriptions, understands nuanced preferences. Goes beyond traditional collaborative filtering.

---

### Q19: What is content summarization with GenAI?

**A)** Data compression  
**C)** Generating concise summaries of long documents preserving key points  
**B)** Keyword extraction

**Answer: C**

Summarization: condense articles, reports, meetings into brief summaries. Use cases: executive summaries, news digests, meeting notes, legal document review. Can specify length, style, audience.

---

### Q20: What is data selection in the FM lifecycle?

**A)** Choosing storage type  
**B)** Curating high-quality, diverse training data  
**C)** Database queries

**Answer: B**

Data selection: collect representative, unbiased data for training. Quality > quantity. Consider diversity (languages, domains), filter toxic content, ensure licensing. Critical for model capabilities and fairness.

---

### Q21: What is model selection in the FM lifecycle?

**A)** Choosing hardware  
**C)** Selecting appropriate pre-trained model based on use case requirements  
**B)** Picking training algorithm

**Answer: C**

Model selection: choose FM based on task (text/image/code), performance needs, cost, latency, context window. Bedrock offers: Claude (reasoning), Titan (cost-effective), Llama (open-source), Stable Diffusion (images).

---

### Q22: What is pre-training in the FM lifecycle?

**A)** User training  
**B)** Training model on massive general dataset (trillions of tokens)  
**C)** Fine-tuning

**Answer: B**

Pre-training: unsupervised learning on vast data (web text, books, code). Teaches language patterns, world knowledge, reasoning. Expensive (millions of dollars, months). Users typically use pre-trained models, not train from scratch.

---

### Q23: What is deployment in the FM lifecycle?

**A)** Training models  
**C)** Making model available via API or application integration  
**B)** Data collection

**Answer: C**

Deployment: host model for inference. Options: Bedrock API (serverless), SageMaker (custom hosting), edge deployment. Consider: latency, throughput, cost, security, compliance.

---

### Q24: What is the feedback loop in the FM lifecycle?

**A)** Audio feedback  
**B)** Collecting user responses to improve model via fine-tuning or RLHF  
**C)** Error logs

**Answer: B**

Feedback loop: monitor model performance, collect user ratings/corrections, retrain/fine-tune. RLHF uses human feedback to align model behavior. Continuous improvement cycle: deploy → monitor → collect feedback → update.

---

### Q25: What is customer service automation with GenAI?

**A)** IVR menus  
**C)** AI agents handling support tickets, chat, email with natural language  
**B)** Call routing

**Answer: C**

GenAI customer service: understand complex questions, access knowledge bases (RAG), provide personalized answers, escalate to humans when needed. Use cases: 24/7 support, ticket triage, FAQ automation. Example: Amazon Lex + Bedrock.

---

### Q26: What is the difference between LLMs and SLMs?

**A)** No difference  
**B)** LLMs have billions of parameters; SLMs have millions for edge/specific tasks  
**C)** SLMs are always better

**Answer: B**

LLMs (Large Language Models): billions of parameters (GPT-4, Claude), high accuracy, expensive, slow. SLMs (Small Language Models): millions of parameters, faster, cheaper, run on edge devices, specialized tasks. Choose based on requirements.

---

### Q27: What is knowledge cutoff in foundation models?

**A)** Security feature  
**C)** Date after which model has no training data/knowledge  
**B)** Token limit

**Answer: C**

Knowledge cutoff: models trained on data up to specific date. Post-cutoff events unknown. Example: Claude trained until April 2024 doesn't know events from June 2024. Use RAG or fine-tuning for current information.

---

### Q28: Why is context length important in foundation models?

**A)** Affects training time  
**B)** Determines how much text model can process in one request  
**C)** Controls model accuracy

**Answer: B**

Context length: total tokens (input + output) model handles. Longer context = more conversation history, larger documents, better understanding. 8K tokens = ~6K words. Trade-off: longer context = slower, costlier inference.

---

### Q29: What distinguishes Claude from other foundation models?

**A)** Only does classification  
**C)** Anthropic model emphasizing safety, long context (200K tokens), strong reasoning  
**B)** Text-only model

**Answer: C**

Claude (Anthropic): constitutional AI for safety, 200K token context window, strong analytical reasoning, multi-modal (text + images), multiple versions (Opus, Sonnet, Haiku). Available on Bedrock.

---

### Q30: What is Meta Llama?

**A)** Closed-source only  
**B)** Open-source foundation model family for research and commercial use  
**C)** AWS proprietary model

**Answer: B**

Llama (Meta): open-source LLMs (Llama 2, Llama 3), various sizes (7B-70B parameters), commercial-friendly license, community fine-tuned versions. Available on Bedrock, SageMaker. Cost-effective alternative to closed models.

---

## Task 2.2: Capabilities & Limitations of GenAI (Q31-45)

### Q31: What is a key advantage of GenAI's adaptability?

**A)** Requires complete retraining for new tasks  
**C)** Can handle diverse tasks without task-specific training  
**B)** Only works for one domain

**Answer: C**

Adaptability: foundation models generalize across tasks via prompting/few-shot learning. Same model handles summarization, Q&A, translation without retraining. Reduces development time and infrastructure complexity.

---

### Q32: How does GenAI provide responsiveness advantages?

**A)** Slower than traditional systems  
**B)** Updates require months of retraining  
**C)** Quickly incorporates new information via prompts or RAG

**Answer: C**

Responsiveness: update knowledge instantly with RAG (no retraining), adjust behavior via prompt changes, deploy new capabilities rapidly. Traditional ML requires data collection, retraining, redeployment for updates.

---

### Q33: What simplicity advantage does GenAI offer?

**A)** Requires complex rule systems  
**B)** Natural language interfaces reduce technical barriers  
**C)** Needs specialized programming

**Answer: B**

Simplicity: interact via natural language (prompts), no complex APIs or rule engines. Business users can iterate prompts without engineering. Reduces time-to-value and democratizes AI access.

---

### Q34: What is nondeterminism in GenAI?

**A)** Predictable outputs  
**C)** Same input may produce different outputs across runs  
**B)** Fixed responses

**Answer: C**

Nondeterminism: probabilistic sampling causes variation. Same prompt → different responses. Challenging for testing, compliance, reproducibility. Mitigate: low temperature, set seed values, deterministic sampling.

---

### Q35: What is the interpretability challenge with GenAI?

**A)** Models explain every decision  
**B)** Difficult to understand why specific outputs were generated  
**C)** Fully transparent reasoning

**Answer: B**

Interpretability: billions of parameters create "black box". Hard to explain why model chose specific words/decisions. Critical for regulated industries (finance, healthcare). Mitigation: chain-of-thought prompting, confidence scores.

---

### Q36: What type of inaccuracy issue affects GenAI?

**A)** Perfect accuracy always  
**C)** May generate plausible-sounding but incorrect information  
**B)** Only mathematical errors

**Answer: C**

Inaccuracy: models confidently state falsehoods, outdated info (training cutoff), misinterpret context. Especially problematic for factual domains. Mitigation: RAG, fact-checking, human review, lower temperature.

---

### Q37: What compliance factor is critical for model selection?

**A)** Model color  
**B)** Data residency and regulatory requirements (GDPR, HIPAA)  
**C)** Marketing materials

**Answer: B**

Compliance: ensure model hosting meets regulations (data sovereignty, privacy, audit trails). Healthcare → HIPAA, EU → GDPR. Check: where data processed, encryption, access logs. Bedrock offers region-specific deployment.

---

### Q38: How do performance requirements influence model selection?

**A)** Always pick largest model  
**C)** Balance latency, throughput, accuracy needs with model size  
**B)** Ignore speed requirements

**Answer: C**

Performance trade-offs: large models (Claude Opus) = high accuracy, slow, expensive. Small models (Haiku, Titan Lite) = fast, cheap, lower accuracy. Real-time chat needs low latency; batch analysis tolerates higher latency.

---

### Q39: What constraint affects model selection for edge deployment?

**A)** Unlimited resources  
**B)** Model size and memory footprint  
**C)** Cloud-only operation

**Answer: B**

Edge constraints: limited memory, compute, power on devices. Requires smaller quantized models, efficient architectures. Bedrock targets cloud; edge needs SageMaker Edge or mobile-optimized models.

---

### Q40: What is conversion rate as a GenAI business metric?

**A)** Currency exchange  
**C)** Percentage of users taking desired action after AI interaction  
**B)** Data transformation

**Answer: C**

Conversion rate: users completing purchase, signup, download after GenAI engagement. Example: 5% of chatbot users → buyers. Measures AI's impact on business goals. Compare AI vs non-AI conversion.

---

### Q41: What is Average Revenue Per User (ARPU) for GenAI apps?

**A)** Total company revenue  
**B)** Revenue divided by number of AI-using customers  
**C)** Development cost

**Answer: B**

ARPU: revenue per user leveraging GenAI features. Tracks monetization effectiveness. Example: users with AI recommendations spend $50/month vs $30 without. Demonstrates ROI of GenAI investment.

---

### Q42: How is cross-domain performance measured for GenAI?

**A)** Single task accuracy only  
**C)** Model effectiveness across multiple use cases/industries  
**B)** Hardware compatibility

**Answer: C**

Cross-domain: evaluate model on diverse tasks (customer service, content generation, analysis). Foundation models excel here vs task-specific models. Measure: accuracy across 5+ domains, reusability, prompt success rate.

---

### Q43: What is Customer Lifetime Value (CLV) in GenAI context?

**A)** One-time purchase  
**B)** Total revenue from customer over entire relationship, enhanced by AI  
**C)** Support cost

**Answer: B**

CLV: predict long-term customer value. GenAI increases CLV via: personalization (retention), upsell recommendations, improved support (satisfaction). Track: CLV of AI-engaged users vs control group.

---

### Q44: How do you measure efficiency gains from GenAI?

**A)** Ignore time savings  
**C)** Time/cost reduction for tasks vs traditional methods  
**B)** Only count errors

**Answer: C**

Efficiency: compare GenAI vs baseline. Metrics: time saved (summary in 30s vs 10min manual), cost per task, automation rate (% tickets auto-resolved), employee productivity (tasks/hour). Quantify ROI.

---

### Q45: What accuracy metric is important for GenAI business value?

**A)** Speed only  
**B)** Correctness of outputs relative to ground truth/human judgment  
**C)** File size

**Answer: B**

Accuracy: measure output correctness for use case. Methods: human evaluation (% acceptable responses), automated benchmarks, task-specific metrics (BLEU for translation, ROUGE for summarization). Track over time, set SLAs.

---

### Q41: What is model drift in GenAI?

**A)** Physical movement  
**C)** Model performance degradation over time as real-world data changes  
**B)** Faster inference

**Answer: C**

Model drift: accuracy declines when real-world patterns shift from training data. Example: language trends change, new topics emerge. Monitor performance metrics, retrain/fine-tune periodically, use RAG for current data. Critical for production systems.

---

### Q42: What computational cost challenge affects GenAI?

**A)** No cost impact  
**B)** Large models require expensive GPUs, high memory, significant power  
**C)** Free to run

**Answer: B**

Computational costs: training FMs costs millions (GPU clusters, weeks/months), inference expensive (per-token pricing adds up at scale). Mitigation: use managed services (Bedrock), smaller models, caching, batch processing.

---

### Q43: What bias and fairness issue can GenAI models have?

**A)** Models are always fair  
**C)** Training data biases reflected in outputs (gender, race, cultural stereotypes)  
**B)** No bias possible

**Answer: C**

Bias: models learn from internet data containing human biases. May generate stereotypical content, unfair decisions. Mitigation: diverse training data, bias detection tools, human review, guardrails, regular audits. Critical for fair AI.

---

### Q44: What output consistency challenge exists with GenAI?

**A)** Always identical outputs  
**B)** Same prompt can produce varying quality/format across runs  
**C)** Perfect consistency

**Answer: B**

Consistency challenge: probabilistic nature causes variation. Testing hard, SLAs difficult, user experience inconsistent. Mitigation: low temperature, prompt templates, output parsing/validation, multiple generations with voting, deterministic mode.

---

### Q45: What regulatory compliance challenge do GenAI applications face?

**A)** No regulations apply  
**C)** Explainability requirements, data privacy laws, industry-specific rules  
**B)** Automatic compliance

**Answer: C**

Compliance challenges: GDPR (right to explanation), HIPAA (healthcare privacy), financial regulations (audit trails), AI Act (EU). Solutions: invocation logging, human-in-loop, explainable AI techniques, data governance, regional deployment.

---

## Task 2.3: AWS Infrastructure & Technologies for GenAI (Q46-65)

### Q46: What is Amazon SageMaker JumpStart?

**A)** Database service  
**B)** Hub for pre-trained models and quick deployment  
**C)** Networking tool

**Answer: B**

SageMaker JumpStart: 1-click deployment of pre-trained models (text, vision, embedding). Includes foundation models, fine-tuning capabilities, example notebooks. Quick start for ML/GenAI without infrastructure setup.

---

### Q47: What is PartyRock?

**A)** Music service  
**C)** Code-free GenAI app builder (Amazon Bedrock Playground)  
**B)** Database tool

**Answer: C**

PartyRock: no-code platform to build GenAI apps using Bedrock. Create chatbots, generators, workflows with drag-and-drop. Learn prompt engineering, prototype ideas. Free experimentation environment.

---

### Q48: How does Amazon Bedrock lower the barrier to entry for GenAI?

**A)** Requires ML expertise  
**B)** No infrastructure management, pay-per-use API access  
**C)** Needs custom training

**Answer: B**

Bedrock accessibility: API access to foundation models (no servers), pay only for tokens used (no upfront investment), pre-built models (no training required), integrated with AWS (familiar tools). Developers can start in minutes.

---

### Q49: What speed-to-market advantage do AWS GenAI services provide?

**A)** Months of setup required  
**C)** Pre-built models and APIs enable rapid prototyping and deployment  
**B)** Must train from scratch

**Answer: C**

Speed advantage: deploy models in days vs months, use pre-trained FMs (skip training), API integration (minimal code), managed infrastructure (no DevOps delay). Iterate quickly, launch MVPs fast.

---

### Q50: How is AWS GenAI cost-effective for businesses?

**A)** Requires massive upfront investment  
**B)** Pay-per-use pricing, no infrastructure costs, shared models  
**C)** Must buy hardware

**Answer: B**

Cost-effectiveness: no GPU infrastructure investment, pay only for tokens used, no model training costs (use pre-trained), scale up/down instantly, share costs across AWS customer base. Lower TCO vs self-hosting.

---

### Q51: What efficiency advantage do managed AWS GenAI services offer?

**A)** Manual infrastructure management  
**C)** Focus on application logic, not infrastructure/model maintenance  
**B)** Requires dedicated ML team

**Answer: C**

Efficiency: AWS manages model hosting, scaling, updates, patches. Teams focus on prompts, RAG, application features. Reduce operational overhead, reallocate resources to innovation vs maintenance.

---

### Q52: How do AWS GenAI services help meet business objectives faster?

**A)** Slow iteration cycles  
**B)** Rapid experimentation and deployment of AI features  
**C)** Requires regulatory approval

**Answer: B**

Business agility: test hypotheses quickly (A/B testing prompts), deploy new features rapidly (API calls), pivot based on feedback, scale successful features. Faster time-to-value, competitive advantage.

---

### Q53: What security benefit does AWS infrastructure provide for GenAI?

**A)** Data stored unencrypted  
**C)** Encryption at rest/in transit, VPC isolation, IAM access controls  
**B)** Public access by default

**Answer: C**

Security: data encrypted (AES-256 at rest, TLS in transit), VPC endpoints (private connectivity), IAM policies (fine-grained permissions), network isolation. Bedrock doesn't store customer prompts/responses by default.

---

### Q54: What compliance certifications does AWS GenAI infrastructure support?

**A)** No certifications  
**B)** HIPAA, GDPR, SOC, ISO, FedRAMP compliance  
**C)** Only local regulations

**Answer: B**

Compliance: AWS infrastructure certified for major frameworks (HIPAA for healthcare, GDPR for EU privacy, SOC 2 for security, ISO 27001, FedRAMP for government). Inherit compliance from AWS platform.

---

### Q55: What is the AWS shared responsibility model for GenAI?

**A)** AWS responsible for everything  
**C)** AWS secures infrastructure; customer secures data, prompts, access  
**B)** Customer responsible for everything

**Answer: C**

Shared responsibility: AWS manages model infrastructure, patching, availability. Customer manages: data classification, prompt content, access controls, guardrails configuration, output validation. Both share security duties.

---

### Q56: How does AWS ensure AI safety in GenAI services?

**A)** No safety measures  
**B)** Guardrails, content filters, responsible AI guidelines, safety tools  
**C)** Relies only on models

**Answer: B**

AI safety: Bedrock Guardrails filter harmful content, responsible AI tools detect bias, toxicity detection, PII redaction capabilities, prompt injection protection. AWS AI Service Cards document model behavior.

---

### Q57: What data residency benefit does AWS provide for GenAI?

**A)** Data stored globally by default  
**C)** Choose specific AWS regions for data processing and storage  
**B)** No control over location

**Answer: C**

Data sovereignty: deploy Bedrock in specific regions (EU, US, Asia), data never leaves chosen region, meets data localization laws, regional compliance (GDPR in Europe). Control where prompts/data processed.

---

### Q58: What cost tradeoff exists between on-demand and provisioned throughput?

**A)** No difference  
**B)** On-demand: flexible, higher per-token cost; Provisioned: committed, lower cost at scale  
**C)** Provisioned is always more expensive

**Answer: B**

Cost tradeoff: On-demand pays per token (flexible, no commitment, good for variable workloads). Provisioned reserves capacity (cheaper per token at high volume, requires commitment, predictable cost). Choose based on usage patterns.

---

### Q59: How does regional coverage affect GenAI cost and performance?

**A)** No impact  
**C)** Closer regions reduce latency; availability varies; pricing differs by region  
**B)** All regions identical

**Answer: C**

Regional tradeoffs: nearby regions = lower latency (better UX), not all models in all regions (availability constraint), pricing varies (US typically cheaper than Asia/Europe). Balance cost, latency, compliance needs.

---

### Q60: What responsiveness tradeoff exists with model size selection?

**A)** All models same speed  
**B)** Larger models = higher accuracy but slower/costlier; smaller = faster/cheaper but lower quality  
**C)** Smaller models always better

**Answer: B**

Performance tradeoff: Claude Opus (accurate, slow, expensive) vs Haiku (fast, cheap, less accurate). Real-time chat needs fast models; complex analysis justifies slow models. Balance quality requirements vs latency/cost.

---

### Q61: How does AWS PrivateLink enhance GenAI security?

**A)** Public internet access  
**C)** Private connectivity between VPC and Bedrock without internet exposure  
**B)** Slower performance

**Answer: C**

PrivateLink: connect to Bedrock via private AWS network (no internet traversal), keeps traffic within AWS backbone, meets compliance for sensitive data, reduces attack surface. Traffic never exposed publicly.

---

### Q62: What is the advantage of AWS IAM for GenAI access control?

**A)** No access controls  
**B)** Fine-grained permissions, role-based access, audit trails  
**C)** Everyone has full access

**Answer: B**

IAM benefits: control who invokes models (user/role permissions), restrict by model/action, temporary credentials (STS), CloudTrail logs all access (audit compliance), integrate with existing identity systems.

---

### Q63: How does CloudWatch support GenAI monitoring?

**A)** No monitoring available  
**C)** Track invocations, latency, errors, tokens, costs with metrics/alarms  
**B)** Only error logs

**Answer: C**

CloudWatch monitoring: track model invocations per second, latency percentiles (p50, p99), error rates, token usage, throttling. Set alarms for anomalies, dashboard for operations, log analysis for debugging.

---

### Q64: What redundancy benefit does AWS provide for GenAI applications?

**A)** Single point of failure  
**B)** Multi-AZ deployments, automatic failover, high availability SLAs  
**C)** No backup systems

**Answer: B**

Redundancy: Bedrock deployed across availability zones (AZs), automatic failover between AZs, 99.9%+ uptime SLA, inference profiles for cross-region routing. Minimize downtime, ensure business continuity.

---

### Q65: How do cost allocation tags help manage GenAI expenses?

**A)** Increase costs  
**C)** Tag resources by project/team/environment to track spending  
**B)** No cost visibility

**Answer: C**

Cost management: tag Bedrock usage by department, project, environment (dev/prod), cost center. AWS Cost Explorer shows spending by tag, enable chargeback, identify optimization opportunities, budget by team.

---

## Prompt Engineering (Q66-73)

### Q66: What is prompt engineering?

**A)** Training new models  
**B)** Crafting inputs to guide model outputs  
**C)** Hardware optimization

**Answer: B**

Prompt engineering: designing effective instructions for GenAI models. Same model, different prompts = different quality results. No code changes needed, just better questions/instructions.

---

### Q67: What is zero-shot prompting?

**A)** Asking without examples  
**B)** Always providing examples  
**C)** Using empty prompts

**Answer: A**

Zero-shot: direct task request without examples ("Summarize this article"). Works for common tasks models saw during training. Simplest approach, use first.

---

### Q68: What is few-shot prompting?

**A)** Short prompts only  
**B)** Providing examples before the task  
**C)** Using multiple models

**Answer: B**

Few-shot: include 2-5 examples showing desired format/pattern before actual task. Model learns from examples. Better consistency and handles unusual formats. More tokens used.

---

### Q69: What is chain-of-thought prompting?

**A)** Multiple separate prompts  
**B)** Asking model to show reasoning steps  
**C)** Connecting models together

**Answer: B**

Chain-of-thought: request step-by-step reasoning ("Let's think step by step"). Improves accuracy on complex reasoning, math, analysis. Makes model's logic transparent.

---

### Q70: What is the purpose of temperature in generation?

**A)** Hardware cooling  
**B)** Controls randomness/creativity of outputs  
**C)** Training speed

**Answer: B**

Temperature (0-1): Low (0.1-0.3) = deterministic, consistent, factual. High (0.7-0.9) = creative, diverse, varied. Use low for facts, high for creative writing.

---

### Q71: What should a well-structured prompt include?

**A)** Only the question  
**B)** Role, task, context, format, constraints  
**C)** Just keywords

**Answer: B**

Effective prompts specify: role (persona), clear task, relevant context, desired format, constraints (length, tone, what to avoid). Explicit instructions = better results.

---

### Q72: What is prompt chaining?

**A)** Long single prompt  
**B)** Breaking complex tasks into sequential prompts  
**C)** Repeating same prompt

**Answer: B**

Prompt chaining: multi-step process where each prompt builds on previous output. Example: extract facts → summarize → generate questions. Higher quality for complex tasks.

---

### Q73: How do you reduce hallucinations in responses?

**A)** Use higher temperature  
**B)** Provide context, use low temperature, request citations  
**C)** Use longer prompts

**Answer: B**

Reduce hallucinations: provide relevant context (RAG), lower temperature (0.1-0.3), explicit instructions ("use only provided information"), request citations, validate facts.

---

## RAG & Advanced Techniques (Q74-81)

### Q74: What is RAG (Retrieval-Augmented Generation)?

**A)** A model type  
**B)** Retrieving relevant docs before generating answers  
**C)** Hardware acceleration

**Answer: B**

RAG: retrieve relevant information from knowledge base, include in prompt, generate grounded answer. Reduces hallucinations, enables current/private data, no model retraining needed.

---

### Q75: What are embeddings in RAG?

**A)** Encrypted data  
**B)** Vector representations capturing semantic meaning  
**C)** Model parameters

**Answer: B**

Embeddings: convert text to high-dimensional vectors (e.g., 1536 dimensions). Similar meaning = similar vectors. Enable semantic search for RAG retrieval.

---

### Q76: Which AWS service offers managed RAG?

**A)** Amazon S3  
**B)** Amazon Bedrock Knowledge Bases  
**C)** Amazon EC2

**Answer: B**

Bedrock Knowledge Bases: fully managed RAG. Automatic chunking, embedding, vector storage (OpenSearch), retrieval, generation. No infrastructure management required.

---

### Q77: What is chunking in RAG?

**A)** Compressing models  
**B)** Splitting documents into smaller segments  
**C)** Batching requests

**Answer: B**

Chunking: divide documents into 500-1000 token segments. Enables focused retrieval, fits in context windows. Overlapping chunks (10-20%) maintain context across boundaries.

---

### Q78: What is a vector database used for?

**A)** Storing images  
**B)** Efficient similarity search on embeddings  
**C)** Caching responses

**Answer: B**

Vector databases store and search embeddings efficiently. Find nearest neighbors (most similar text). AWS options: OpenSearch (k-NN), Aurora (pgvector), MemoryDB (vector search).

---

### Q79: When should you use RAG vs fine-tuning?

**A)** Always use RAG  
**B)** RAG for facts, fine-tuning for style  
**C)** Always use fine-tuning

**Answer: B**

RAG: current/changing information, factual Q&A, private data, cheaper. Fine-tuning: specialized style/tone, static domain knowledge, format requirements. Combine both for style + facts.

---

### Q80: What is hybrid search in RAG?

**A)** Using multiple models  
**B)** Combining semantic (vector) and keyword search  
**C)** Cloud and on-premise

**Answer: B**

Hybrid search: blend vector similarity (semantic) + keyword matching (BM25). Better accuracy: semantic understands meaning, keywords catch exact terms/codes. Typical: 70% semantic, 30% keyword.

---

### Q81: How does RAG reduce hallucinations?

**A)** Trains model longer  
**B)** Grounds answers in retrieved source documents  
**C)** Uses smaller models

**Answer: B**

RAG provides factual context from knowledge base. Model answers from provided documents, not just memorized training data. Can cite sources, admit when info unavailable.

---

## Production & AWS Services (Q82-85)

### Q82: What are Amazon Bedrock Guardrails?

**A)** Physical security  
**B)** Content filters and safety controls for model outputs  
**C)** Network firewalls

**Answer: B**

Guardrails filter harmful content (hate, violence, sexual), block denied topics (medical advice), redact PII, enforce custom word filters. Multi-layer safety for production GenAI.

---

### Q83: What is Amazon Q?

**A)** Queue service  
**B)** AI assistant for AWS and business tasks  
**C)** Quantum computing

**Answer: B**

Amazon Q: AI assistant for AWS (troubleshooting, code generation), business intelligence queries, document analysis. Understands AWS context, provides actionable answers.

---

### Q84: How do you optimize GenAI costs?

**A)** Always use largest model  
**B)** Choose appropriate model size, cache responses, optimize tokens  
**C)** Use highest temperature

**Answer: B**

Cost optimization: use smallest model meeting requirements (Lite for simple tasks), cache common queries, compress prompts, set max_tokens, batch processing for non-urgent.

---

### Q85: What is provisioned throughput in Bedrock?

**A)** Free tier  
**B)** Reserved model capacity with guaranteed availability  
**C)** Open source access

**Answer: B**

Provisioned throughput: reserve model capacity for predictable latency/availability. Cost-effective at high volume. Alternative to on-demand (pay-per-token). Commit for 1-6 months.

---

## Fine-Tuning & Customization (Q86-93)

### Q86: What is fine-tuning a foundation model?

**A)** Adjusting temperature  
**B)** Training model on domain-specific data to specialize it  
**C)** Prompt engineering

**Answer: B**

Fine-tuning: continue training pre-trained model on your data. Teaches specialized vocabulary, style, domain knowledge. Requires labeled data, compute. Bedrock supports fine-tuning for some models.

---

### Q87: What is instruction tuning?

**A)** Writing better prompts  
**B)** Training models to follow natural language instructions  
**C)** User guides

**Answer: B**

Instruction tuning: train model on instruction-response pairs. Makes models better at following directions. Foundation models like Claude, GPT use instruction tuning. Improves task generalization.

---

### Q88: What is model context window?

**A)** Training dataset size  
**B)** Maximum input/output tokens model can process  
**C)** Memory usage

**Answer: B**

Context window: token limit for combined input + output. Claude: 200K, GPT-4: 8-32K, Titan: 8K. Exceeding limit requires chunking or summarization. Larger = more context but slower/costlier.

---

### Q89: What is top-p (nucleus sampling)?

**A)** Selecting top model  
**B)** Sampling from smallest set of tokens whose cumulative probability exceeds p  
**C)** Precision metric

**Answer: B**

Top-p (0-1): sample from tokens totaling probability p. Lower = more focused/deterministic. Higher = more diverse. Alternative to temperature. Combine for control: low temp + low top-p = consistent.

---

### Q90: What is model quantization?

**A)** Adding more parameters  
**B)** Reducing model size by using lower precision  
**C)** Training longer

**Answer: B**

Quantization: convert 32-bit weights to 8-bit or lower. Smaller model size, faster inference, lower memory. Some accuracy trade-off. Used for edge deployment.

---

### Q91: What is Amazon Bedrock Model Evaluation?

**A)** User reviews  
**B)** Automated testing of models with benchmarks and custom prompts  
**C)** Hardware testing

**Answer: B**

Model Evaluation: compare foundation models on standard benchmarks (reasoning, knowledge) or your custom prompts. Helps select best model for use case. Human or automated evaluation.

---

### Q92: What is prompt template?

**A)** Pre-built website  
**B)** Reusable prompt structure with variables  
**C)** Model architecture

**Answer: B**

Prompt template: standardized prompt with placeholders (e.g., "Summarize {document} for {audience}"). Ensures consistency, enables testing, supports A/B testing. Bedrock supports saved templates.

---

### Q93: What is semantic similarity in embeddings?

**A)** Text length matching  
**B)** How close vectors are in embedding space (cosine similarity)  
**C)** Character matching

**Answer: B**

Semantic similarity: measure vector distance (cosine similarity: -1 to 1). High score (> 0.8) = similar meaning. Used in RAG retrieval, duplicate detection, recommendation.

---

## Production & Enterprise (Q94-100)

### Q94: What is model latency?

**A)** Training time  
**B)** Time from request to response  
**C)** Data lag

**Answer: B**

Latency: request-to-response time (milliseconds). Affected by: model size, prompt length, output length, concurrent requests. Optimize: smaller models, provisioned throughput, streaming.

---

### Q95: What is token-based pricing?

**A)** Fixed monthly cost  
**B)** Pay per 1000 input/output tokens  
**C)** Free unlimited

**Answer: B**

Bedrock charges per 1000 tokens (input + output separately). Input cheaper than output. Track usage to control costs. Example: Claude Haiku $0.00025 in, $0.00125 out per 1K tokens.

---

### Q96: What is streaming response?

**A)** Video generation  
**B)** Sending generated tokens progressively as they're created  
**C)** Batch processing

**Answer: B**

Streaming: return partial responses as generated (word-by-word or sentence-by-sentence). Better UX (immediate feedback), can cancel early. Supported by Bedrock API. Use for chatbots.

---

### Q92: What is Amazon Bedrock Playground?

**A)** Game development  
**B)** Interactive UI for testing models and prompts  
**C)** Sandbox environment

**Answer: B**

Playground: web UI for experimenting with foundation models. Test prompts, compare models, adjust parameters (temperature, top-p). No code needed. Iterate before API integration.

---

### Q97: What is inference profile in Bedrock?

**A)** User profile  
**B)** Configuration for cross-region routing and failover  
**C)** Hardware specs

**Answer: B**

Inference profiles: route requests across regions for resilience, load balancing. Automatic failover if region unavailable. Improves availability and latency for global applications.

---

### Q98: What is model invocation logging?

**A)** Training logs  
**B)** Recording requests/responses for audit, debugging, analysis  
**C)** Error logs only

**Answer: B**

Invocation logging: capture prompts, responses, metadata to CloudWatch or S3. Use for: debugging, compliance audits, quality analysis, usage tracking. PII considerations for storage.

---

### Q99: What is Amazon Bedrock Studio?

**A)** Video editing  
**B)** Development environment for building GenAI applications  
**C)** Recording studio

**Answer: B**

Bedrock Studio: IDE for building, testing, deploying GenAI apps. Includes RAG setup, agent creation, collaboration tools. Simplified workflow for developers.

---

### Q100: What is Amazon Bedrock Playground?

**A)** Testing environment  
**B)** Interactive UI for experimenting with foundation models  
**C)** Gaming platform

**Answer: B**

Playground: web UI for experimenting with foundation models. Test prompts, compare models, adjust parameters (temperature, top-p). No code needed. Iterate before API integration.

---

## Advanced RAG & Agents (Q101-105)

### Q101: What is agent reasoning in Bedrock Agents?

**A)** Human thinking  
**B)** LLM decides which tools/APIs to call based on user request  
**C)** Rule engine

**Answer: B**

Agent reasoning: LLM analyzes user goal, breaks into steps, selects appropriate tools/APIs, executes plan. ReAct pattern (Reasoning + Acting). Handles multi-step complex tasks.

---

### Q102: What is metadata filtering in RAG?

**A)** Removing PII  
**B)** Narrowing search by document attributes (date, author, category)  
**C)** Content moderation

**Answer: B**

Metadata filtering: combine semantic search with filters (e.g., "documents from 2024 by Finance department"). Improves precision, reduces irrelevant results. Store metadata with embeddings.

---

### Q103: What is retrieval confidence scoring?

**A)** Model accuracy  
**B)** How relevant retrieved documents are to query  
**C)** User rating

**Answer: B**

Confidence scoring: measure document relevance (similarity score 0-1). Set threshold (e.g., > 0.7) to filter low-quality matches. Prevents poor context degrading LLM answers.

---

### Q104: What is Amazon Bedrock Agents action group?

**A)** User permissions  
**B)** Set of API operations agent can perform  
**C)** Model ensemble

**Answer: B**

Action group: define APIs/functions agent can call (e.g., database queries, order creation, email sending). Provide OpenAPI specs. Agent decides when/how to use.

---

### Q105: What is synthetic data generation with GenAI?

**A)** Real data collection  
**B)** Using LLMs to create training data  
**C)** Data deletion

**Answer: B**

Synthetic data: LLM generates training examples (questions, answers, variations). Useful when real data scarce, expensive, or privacy-sensitive. Augments real data, tests edge cases.

---

## Additional Bedrock Concepts (Q106-110)

### Q106: What is Amazon Bedrock Playground?

**A)** Testing environment  
**B)** Interactive UI for experimenting with foundation models  
**C)** Gaming platform

**Answer: B**

Playground: web UI for experimenting with foundation models. Test prompts, compare models, adjust parameters (temperature, top-p). No code needed. Iterate before API integration.

---

### Q107: What is the purpose of a system prompt?

**A)** Operating system commands  
**B)** Instructions that set model behavior and persona  
**C)** Hardware initialization

**Answer: B**

System prompt: pre-prompt defining model's role, constraints, and behavior (e.g., "You are a helpful AWS expert"). Persists across conversation. Guides tone, expertise level, guardrails.

---

### Q108: What is model inference optimization?

**A)** Training acceleration  
**B)** Techniques to reduce latency and cost during generation  
**C)** Data preprocessing

**Answer: B**

Inference optimization: reduce latency via smaller models, quantization, batching, caching, streaming responses. Monitor tokens per second (TPS), time to first token (TTFT). Trade accuracy vs speed.

---

### Q109: What is multi-modal prompting?

**A)** Using multiple models  
**B)** Combining text, images, audio in a single prompt  
**C)** Multiple languages

**Answer: B**

Multi-modal prompting: input different modalities together (e.g., image + text question). Models like Claude 3 Sonnet accept images. Enables visual Q&A, document analysis, image understanding.

---

### Q110: What is the purpose of Bedrock model access settings?

**A)** User login  
**B)** Requesting and managing access to specific foundation models  
**C)** Network configuration

**Answer: B**

Model access settings: request permission to use specific models in Bedrock (per region). Some models require approval. Manage via AWS Console → Bedrock → Model access. Security and compliance control.

---

## Exam Tips

**Key Concepts to Remember:**

1. **GenAI vs Traditional ML:** Creates content vs predicts/classifies
2. **Foundation Models:** Large, pre-trained, multi-task capable (billions of parameters)
3. **AWS Bedrock:** Managed access to foundation models (Claude, Titan, Llama, Stable Diffusion)
4. **Transformers:** Self-attention architecture powering LLMs
5. **Prompt Engineering:**
   - Zero-shot (no examples)
   - Few-shot (with examples)
   - Chain-of-thought (step-by-step)
   - Temperature: low (factual), high (creative)
6. **RAG:** Retrieve + Generate for grounded answers
   - Embeddings (vector representations)
   - Vector databases (similarity search)
   - Bedrock Knowledge Bases (managed RAG)
7. **AWS Services:**
   - **Bedrock:** Foundation models API
   - **CodeWhisperer:** Code generation
   - **Amazon Q:** AI assistant
   - **Titan:** AWS foundation models
8. **Safety:** Guardrails (content filters, PII redaction, topic blocking)

**Study Focus:**

- Match GenAI use case to appropriate model/service
- Understand when to use RAG vs fine-tuning
- Know prompt engineering techniques
- Identify cost optimization strategies
- Recognize safety/guardrail scenarios
