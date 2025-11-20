# Domain 3: Applications of Foundation Models

140 focused MCQs for AWS AI Practitioner exam preparation, organized by exam objectives.

---

## 3.0 Applications of Foundation Models (Q1-10)

### Q1: What are the primary application categories for foundation models in enterprise?

**A)** Only text generation and chatbots  
**B)** Text generation, code assistance, content creation, search/retrieval, and data extraction  
**C)** Database management only

**Answer: B**

Foundation models enable diverse applications: text generation (chatbots, summaries), code assistance (CodeWhisperer), content creation (marketing), semantic search (RAG), document extraction. Multi-domain versatility is key advantage.

---

### Q2: What is the difference between general-purpose and specialized FM applications?

**A)** No difference  
**B)** General handles any task; specialized optimized for specific domains (legal, medical, code)  
**C)** Specialized is always better

**Answer: B**

General-purpose FMs (GPT, Claude) handle broad tasks. Specialized models fine-tuned for domains (CodeLlama for code, Med-PaLM for healthcare) offer better accuracy but narrower scope. Choose based on use case requirements.

---

### Q3: What is a multimodal foundation model application?

**A)** Using multiple separate models  
**B)** Single model processing multiple data types (text, images, audio)  
**C)** Running models in parallel

**Answer: B**

Multimodal FMs process cross-modal inputs/outputs: text-to-image generation (Stable Diffusion), image captioning (Claude with vision), document understanding. Single unified model handles multiple modalities.

---

### Q4: What is the role of RAG (Retrieval-Augmented Generation) in FM applications?

**A)** Training models faster  
**B)** Combining external knowledge retrieval with generation for accurate, current responses  
**C)** Reducing model size

**Answer: B**

RAG retrieves relevant documents from knowledge base, provides context to FM for generation. Ensures answers grounded in company data, handles current information, reduces hallucinations. Critical for enterprise Q&A systems.

---

### Q5: What is agentic AI in foundation model applications?

**A)** Human-controlled systems  
**B)** Autonomous AI that plans, executes multi-step tasks, and uses tools  
**C)** Simple chatbots

**Answer: B**

Agentic AI (Bedrock Agents) breaks down complex tasks, plans steps, invokes APIs/functions, iterates based on results. Example: "Book cheapest hotel in Paris" → search, compare, filter, book. Goes beyond single-turn responses.

---

### Q6: What is the difference between synchronous and asynchronous FM invocation?

**A)** No difference in practice  
**B)** Synchronous waits for response; asynchronous processes in background  
**C)** Asynchronous is always faster

**Answer: B**

Synchronous: real-time chatbot interactions, waits for completion. Asynchronous: batch document processing, long-running tasks, uses queues (SQS). Choose based on latency requirements and processing volume.

---

### Q7: What is streaming in foundation model responses?

**A)** Video streaming  
**B)** Returning response tokens incrementally as generated, not waiting for completion  
**C)** Batch processing

**Answer: B**

Streaming sends tokens as model generates them, improving perceived latency. User sees response building in real-time. Critical for conversational UI. Bedrock supports streaming via InvokeModelWithResponseStream API.

---

### Q8: What is model orchestration in complex FM applications?

**A)** Using single model only  
**B)** Coordinating multiple models, APIs, and data sources to complete workflows  
**C)** Database management

**Answer: B**

Orchestration chains operations: retrieve docs (Kendra) → summarize (Bedrock) → extract entities (Comprehend) → store (DynamoDB). Bedrock Agents, Step Functions, or custom logic coordinate steps.

---

### Q9: What is the difference between conversational and completion-based FM applications?

**A)** No difference  
**B)** Conversational maintains context across turns; completion is single-turn task execution  
**C)** Completions are always faster

**Answer: B**

Conversational (chatbots): multi-turn dialogue with memory/context. Completion (text generation): single request/response (summarize doc, generate code). Different prompting and state management approaches.

---

### Q10: What is the role of caching in FM applications?

**A)** Not useful for AI  
**B)** Storing frequent responses or embeddings to reduce latency and cost  
**C)** Only for database queries

**Answer: B**

Caching reduces redundant FM calls: cache embeddings for documents (ElastiCache), cache common responses (frequent FAQs), cache intermediate results. Improves performance, lowers inference costs.

---

## 3.1 Design Considerations for FM Applications (Q11-20)

### Q11: What is the primary latency consideration when designing FM applications?

**A)** Latency doesn't matter  
**B)** Balance model size, context length, and user experience requirements  
**C)** Always use largest model

**Answer: B**

Latency factors: model size (larger = slower), context length (longer = slower), token generation (sequential). Design choices: streaming responses, asynchronous processing, smaller models for non-critical tasks, caching.

---

### Q12: What is context window management in FM application design?

**A)** Ignoring context limits  
**B)** Ensuring inputs fit within model's token limit through truncation or summarization  
**C)** Always use maximum context

**Answer: B**

Models have context limits (Claude: 200K tokens). Strategies: truncate old messages, summarize history, split long documents, prioritize relevant context. Poor management causes errors or lost information.

---

### Q13: What is the cost consideration for FM inference at scale?

**A)** Cost is negligible  
**B)** Token-based pricing: minimize input/output tokens, cache results, use appropriate model sizes  
**C)** Always use on-demand pricing

**Answer: B**

Costs scale with tokens processed. Optimizations: shorter prompts, response length limits, caching frequent queries, batch processing, reserved capacity (Provisioned Throughput) for predictable workloads.

---

### Q14: What is error handling best practice for FM applications?

**A)** Let application crash  
**B)** Implement retries with exponential backoff, fallbacks, and user-friendly error messages  
**C)** Ignore errors

**Answer: B**

FMs can fail (rate limits, timeouts, service issues). Design: retry with backoff, fallback to simpler responses, graceful degradation, log errors for monitoring. Don't expose raw errors to users.

---

### Q15: What is the security consideration for user inputs in FM applications?

**A)** Trust all inputs  
**B)** Validate, sanitize, and apply guardrails to prevent prompt injection and abuse  
**C)** No validation needed

**Answer: B**

User inputs can contain prompt injections, PII, malicious content. Protections: input validation, Bedrock Guardrails (content filtering), PII detection/redaction, rate limiting. Defense in depth approach.

---

### Q16: What is observability in FM application design?

**A)** Not important  
**B)** Logging inputs/outputs, monitoring latency/costs/errors, tracing request flows  
**C)** Only monitor server CPU

**Answer: B**

Observability requirements: CloudWatch for metrics (latency, token counts), logs for debugging, X-Ray for tracing, custom metrics (user satisfaction). Essential for production troubleshooting and optimization.

---

### Q17: What is stateful vs stateless FM application design?

**A)** No difference  
**B)** Stateful maintains conversation/user context; stateless treats each request independently  
**C)** Stateful is always better

**Answer: B**

Stateless (REST APIs): each request independent, easier to scale. Stateful (chatbots): maintains session context (DynamoDB, ElastiCache). Choose based on use case: document summarization (stateless), customer service chat (stateful).

---

### Q18: What is the design consideration for handling PII in FM applications?

**A)** Store all PII  
**B)** Detect, redact, or tokenize PII before processing; avoid logging sensitive data  
**C)** Ignore PII regulations

**Answer: B**

PII protection: use Comprehend to detect PII, redact before FM processing, tokenize for logs, encrypt at rest/transit. Compliance (GDPR, HIPAA) requires PII handling. Bedrock doesn't log prompts, but you might.

---

### Q19: What is the scalability consideration for FM applications?

**A)** Single instance is sufficient  
**B)** Design for horizontal scaling, load balancing, queue-based processing for spikes  
**C)** Vertical scaling only

**Answer: B**

Scalability strategies: API Gateway + Lambda (auto-scales), SQS for buffering requests, Bedrock auto-scales (on-demand), Provisioned Throughput for guaranteed capacity. Design for variable load patterns.

---

### Q20: What is the user experience consideration for slow FM responses?

**A)** Make users wait silently  
**B)** Show loading indicators, stream responses, set expectations, provide async options  
**C)** Don't acknowledge delays

**Answer: B**

UX for latency: streaming (real-time tokens), progress indicators ("Thinking..."), estimated times, async notifications (email when ready). Don't leave users wondering if system is working.

---

### Q21: What are the key criteria for selecting a pre-trained foundation model?

**A)** Only cost matters  
**B)** Cost, modality, latency, language support, model size, customization capabilities, context length  
**C)** Random selection

**Answer: B**

Selection criteria: cost per token, supported modalities (text/image/code), inference latency, multi-lingual support, model size (small/large), fine-tuning availability, max context window. Balance requirements across dimensions for use case.

---

### Q22: What is the temperature parameter in FM inference?

**A)** Server cooling setting  
**B)** Controls randomness/creativity: low=deterministic, high=creative  
**C)** Processing speed

**Answer: B**

Temperature (0.0-1.0): controls output randomness. Low (0.0-0.3): deterministic, factual, consistent. High (0.7-1.0): creative, diverse, unpredictable. Use low for factual tasks (customer support), high for creative content (brainstorming, stories).

---

### Q23: What are top-p and top-k parameters in FM inference?

**A)** Model size settings  
**B)** Control output diversity by limiting token selection during generation  
**C)** API endpoints

**Answer: B**

Top-p (nucleus sampling): select from smallest token set with cumulative probability ≥ p (e.g., 0.9). Top-k: select from top k most likely tokens. Both control diversity. Lower values = more focused, higher = more varied.

---

### Q24: How do input and output length parameters affect FM applications?

**A)** No impact  
**B)** Longer inputs increase latency/cost; output length limits control response size and cost  
**C)** Only affect storage

**Answer: B**

Input length: more tokens = higher latency and cost. Output length (max_tokens): limits response size, controls cost, prevents runaway generation. Set appropriately: summaries need less, detailed analysis needs more. Balance completeness vs efficiency.

---

### Q25: What is model customization and when is it needed?

**A)** Changing model name  
**B)** Fine-tuning or continued pre-training for domain-specific performance  
**C)** Not available

**Answer: B**

Customization options: fine-tuning (train on examples), continued pre-training (domain corpus). Needed when: specialized terminology, specific formats, industry-specific knowledge, brand voice consistency. More expensive than prompting but better for high-volume specialized tasks.

---

### Q26: What is multi-lingual support consideration in model selection?

**A)** All models equal for all languages  
**B)** Models vary in language coverage and quality; check training data distribution  
**C)** Language doesn't matter

**Answer: B**

Multi-lingual considerations: English typically best quality (most training data). Check model's supported languages, quality varies by language. Some models better for specific regions (e.g., European languages, Asian languages). Test performance in target languages.

---

### Q27: What is RAG (Retrieval-Augmented Generation) architecture?

**A)** Standalone model inference  
**B)** Retrieve relevant documents from knowledge base, provide as context to FM for grounded generation  
**C)** Random document selection

**Answer: B**

RAG workflow: user query → convert to embedding → search vector DB for similar docs → retrieve top results → provide docs + query to FM → generate grounded response. Reduces hallucinations, enables current information, grounds answers in company data.

---

### Q28: What are business applications of RAG?

**A)** Only chatbots  
**B)** Knowledge base Q&A, customer support, document search, compliance assistance, internal wikis  
**C)** Database management

**Answer: B**

RAG use cases: employee knowledge portals, customer support (search docs for answers), compliance queries (regulations/policies), product documentation, research assistance, legal document analysis. Bedrock Knowledge Bases provide managed RAG implementation.

---

### Q29: Which AWS services support vector storage for embeddings in RAG applications?

**A)** Only S3  
**B)** OpenSearch Service, Aurora, Neptune, DocumentDB, RDS for PostgreSQL (pgvector)  
**C)** CloudWatch

**Answer: B**

Vector DB options: OpenSearch (k-NN plugin), Aurora PostgreSQL (pgvector), Neptune Analytics (graph + vectors), DocumentDB (vector search), RDS PostgreSQL (pgvector extension). Choose based on existing infrastructure, scale, query patterns.

---

### Q30: What is Amazon Bedrock Knowledge Bases?

**A)** Database service  
**B)** Managed RAG solution: ingest docs, create embeddings, store in vector DB, retrieve for generation  
**C)** Model training

**Answer: B**

Bedrock Knowledge Bases: fully managed RAG. Upload docs (S3), automatic chunking/embedding, vector storage (OpenSearch Serverless), retrieval orchestration, citation tracking. Simplifies RAG implementation vs building custom. Integrates with Bedrock models.

---

## 3.2 Prompt Engineering Techniques (Q31-40)

### Q31: What is zero-shot prompting?

**A)** Providing many examples  
**B)** Asking model to perform task without examples, relying on pre-training  
**C)** Not providing any instructions

**Answer: B**

Zero-shot: direct instruction without examples. "Summarize this article." Works for common tasks where model has sufficient training. Simple, but less accurate than few-shot for complex/specific tasks.

---

### Q32: What is few-shot prompting?

**A)** No examples provided  
**B)** Including 2-5 examples of input-output pairs to guide model behavior  
**C)** Training the model

**Answer: B**

Few-shot: provide examples in prompt. "Classify sentiment: 'Great product!' → Positive, 'Terrible service' → Negative, [your input]." Improves accuracy and consistency for specific formats/domains.

---

### Q33: What is chain-of-thought (CoT) prompting?

**A)** Single-step answers  
**B)** Prompting model to show reasoning steps before final answer  
**C)** Random thinking

**Answer: B**

CoT: "Let's think step by step." Model explains reasoning, improving complex problem-solving (math, logic, analysis). Increases accuracy and provides transparency. Can combine with few-shot.

---

### Q34: What is the role of system prompts vs user prompts?

**A)** No difference  
**B)** System sets behavior/role; user provides specific request  
**C)** System prompts are ignored

**Answer: B**

System prompt: defines AI persona, rules, constraints ("You are a helpful financial advisor. Never provide investment advice."). User prompt: specific query. System sets context for all interactions.

---

### Q35: What is prompt template design?

**A)** Random text generation  
**B)** Reusable prompt structures with placeholders for dynamic content  
**C)** Hard-coded prompts only

**Answer: B**

Templates: "Summarize the following {document_type} in {length} sentences: {content}". Enables consistency, parameterization, testing. Store templates separately from code for easy updates.

---

### Q36: What is negative prompting?

**A)** Being rude to AI  
**B)** Explicitly stating what NOT to do or include  
**C)** Ignoring the model

**Answer: B**

Negative prompts: "Summarize but don't include technical jargon" or "Don't make up information." Guides model away from undesired behaviors. Useful for content filtering and constraint setting.

---

### Q37: What is the importance of prompt clarity and specificity?

**A)** Vague prompts work fine  
**B)** Clear, specific instructions reduce ambiguity and improve output quality  
**C)** Length doesn't matter

**Answer: B**

Specific prompts: "Write 3 bullet points summarizing key financial risks" better than "Summarize this." Include: format, length, style, constraints. Reduces iterations and hallucinations.

---

### Q38: What is role prompting?

**A)** Assigning user roles  
**B)** Instructing model to adopt specific persona or expertise  
**C)** Role-based access control

**Answer: B**

Role prompting: "You are an expert Python developer" or "Act as a customer service agent." Influences tone, knowledge depth, response style. Improves domain-specific accuracy.

---

### Q39: What is iterative prompt refinement?

**A)** Using first prompt always  
**B)** Testing and improving prompts based on output quality  
**C)** Ignoring results

**Answer: B**

Prompt engineering is iterative: test prompt → evaluate output → refine wording/examples → retest. Track versions, measure metrics (accuracy, relevance). A/B test prompt variations in production.

---

### Q40: What is the difference between instruction-following and completion-style prompts?

**A)** No difference  
**B)** Instruction gives command; completion provides beginning for model to continue  
**C)** Completion is always better

**Answer: B**

Instruction: "Translate to French: Hello" (command). Completion: "The three main benefits of cloud computing are" (model continues). Modern chat models prefer instructions; older models used completions.

---

### Q41: What is single-shot prompting?

**A)** No examples provided  
**B)** Providing exactly one example before the actual task  
**C)** Multiple examples

**Answer: B**

Single-shot (one-shot): provide one example to demonstrate format/style. "Sentiment: 'Great product!' → Positive. Now classify: 'Terrible service'." Middle ground between zero-shot and few-shot. Useful when examples are expensive or limited.

---

### Q42: What are the key components of effective prompt structure?

**A)** Only the question  
**B)** Context, instruction, input data, output format, constraints  
**C)** Random text

**Answer: B**

Effective prompt components: Context (role/scenario), Instruction (what to do), Input data (content to process), Output format (structure/length), Constraints (rules/limitations). Clear structure improves consistency and quality.

---

### Q43: What is model latent space in context of prompting?

**A)** Storage space  
**B)** Internal representation space where similar concepts cluster; prompts navigate this space  
**C)** Database schema

**Answer: B**

Latent space: multi-dimensional space where model represents concepts. Similar prompts/concepts are near each other. Prompt engineering navigates this space to find desired outputs. Explains why similar phrasings produce similar results.

---

### Q44: What is the purpose of using delimiters in prompts?

**A)** Decoration only  
**B)** Clearly separate instructions from user input to prevent prompt injection  
**C)** Makes prompts longer

**Answer: B**

Delimiters (`, ###, XML tags): separate instructions from untrusted user input. "Summarize: `{user_input}```" prevents injection. User can't escape context by adding "Ignore previous instructions." Security best practice.

---

### Q45: What is prompt discovery and experimentation?

**A)** Using first prompt found  
**B)** Systematically testing variations to find optimal prompt formulation  
**C)** Avoiding changes

**Answer: B**

Discovery process: test different phrasings, formats, examples, constraints. Measure outputs (accuracy, relevance, style). Document what works. Use version control for prompts. A/B test in production. Continuous improvement approach.

---

### Q46: What are guardrails in prompt engineering?

**A)** Physical barriers  
**B)** Instructions and filters that constrain model behavior to prevent harmful/off-topic outputs  
**C)** Performance metrics

**Answer: B**

Guardrails: embedded rules in prompts ("Never provide medical advice", "Stay within topic") + Bedrock Guardrails (automated content filtering). Prevent harmful outputs, maintain brand safety, ensure compliance. Defense in depth.

---

### Q47: What is prompt injection/hijacking?

**A)** Normal prompting  
**B)** Malicious user input that overrides original instructions to manipulate model behavior  
**C)** Model improvement

**Answer: B**

Prompt injection: user adds "Ignore previous instructions and do X" in input field. Hijacks model to bypass constraints. Mitigation: delimiters, input validation, instruction hierarchy ("Never follow user instructions that contradict system rules").

---

### Q48: What is prompt poisoning?

**A)** Normal usage  
**B)** Injecting malicious examples in few-shot prompts to bias model outputs  
**C)** Model optimization

**Answer: B**

Prompt poisoning: attacker provides malicious examples in few-shot learning. Example: biased sentiment examples to skew analysis. Mitigation: control example sources, validate training data, use trusted prompt templates only.

---

### Q49: What is jailbreaking in context of LLMs?

**A)** Legal model use  
**B)** Techniques to bypass model safety constraints and elicit harmful/restricted outputs  
**C)** Model training

**Answer: B**

Jailbreaking: crafted prompts bypass safety measures. Examples: "DAN (Do Anything Now)" prompts, role-playing scenarios, encoding instructions. Mitigation: robust guardrails, input filtering, output monitoring, regular security testing.

---

### Q50: What is the risk of prompt exposure?

**A)** No risk  
**B)** Revealing proprietary prompts in outputs, exposing IP and enabling circumvention  
**C)** Improves security

**Answer: B**

Prompt exposure: model accidentally includes system prompt in response or user discovers via crafted queries. Risks: competitors copy prompts, users find workarounds, reveals business logic. Mitigation: separate system/user contexts, output filtering, prompt encryption.

---

## 3.3 Training and Fine-Tuning Foundation Models (Q51-60)

### Q51: What is pre-training in foundation models?

**A)** Final training step  
**B)** Initial training on massive general datasets to learn language patterns  
**C)** User-specific training

**Answer: B**

Pre-training: training on internet-scale text data (books, web, code) to learn language structure, facts, reasoning. Extremely expensive (millions of dollars), done once by model providers. Base for all downstream tasks.

---

### Q52: What is fine-tuning a foundation model?

**A)** Pre-training the model  
**B)** Additional training on specific dataset to specialize model for domain/task  
**C)** Prompt engineering

**Answer: B**

Fine-tuning: train pre-trained model on custom data (company documents, specific format, domain knowledge). Less data/cost than pre-training. Bedrock supports fine-tuning for select models. Improves task-specific accuracy.

---

### Q53: What is the difference between fine-tuning and prompt engineering?

**A)** No difference  
**B)** Fine-tuning modifies model weights; prompting provides runtime instructions  
**C)** Prompting is always better

**Answer: B**

Fine-tuning: permanent model changes, requires training data/compute, better for consistent specialized behavior. Prompting: flexible, no training needed, quick iteration. Start with prompting; fine-tune if needed.

---

### Q54: What is instruction fine-tuning?

**A)** Training on raw text  
**B)** Fine-tuning on instruction-response pairs to improve instruction-following  
**C)** Pre-training phase

**Answer: B**

Instruction tuning: train on datasets of instructions + desired responses. Makes model better at following commands. Example: (instruction: "Summarize", text: "...", response: "Summary: ..."). Commercial models already instruction-tuned.

---

### Q55: What is RLHF (Reinforcement Learning from Human Feedback)?

**A)** Supervised learning only  
**B)** Training using human ratings of responses to align model with preferences  
**C)** Automatic training

**Answer: B**

RLHF: humans rate model outputs (helpful/harmful), model learns from feedback. Improves alignment, safety, helpfulness. Used in ChatGPT, Claude. Advanced technique beyond initial supervised training.

---

### Q56: What data is needed for fine-tuning a foundation model?

**A)** No data required  
**B)** Domain-specific examples (100s-1000s) with inputs and desired outputs  
**C)** Internet-scale data

**Answer: B**

Fine-tuning data: task-specific examples (prompts + completions), typically 100-10,000 examples. Quality matters more than quantity. Format: JSONL with prompt/completion pairs. Bedrock requires min examples per model.

---

### Q57: What is continuous fine-tuning?

**A)** One-time training  
**B)** Regularly updating fine-tuned model with new data to maintain accuracy  
**C)** Pre-training phase

**Answer: B**

Continuous fine-tuning: periodic retraining as new data arrives (customer interactions, product updates). Keeps model current. Automate: collect feedback, retrain monthly/quarterly, evaluate, deploy. Lifecycle management.

---

### Q58: What is the cost consideration for fine-tuning vs prompting?

**A)** Both are free  
**B)** Fine-tuning has upfront training cost; prompting costs per inference  
**C)** Fine-tuning is always cheaper

**Answer: B**

Fine-tuning: training cost (compute hours) + hosting custom model. Prompting: per-token inference cost. Fine-tuning worthwhile for high-volume, specialized tasks. Prompting better for low-volume or varying tasks.

---

### Q59: What is model distillation?

**A)** Removing models  
**B)** Training smaller model to mimic larger model's behavior  
**C)** Data cleaning

**Answer: B**

Distillation: large "teacher" model generates training data for small "student" model. Student learns to approximate teacher with lower latency/cost. Trade-off: speed vs some accuracy loss.

---

### Q60: What is the evaluation process during fine-tuning?

**A)** No evaluation needed  
**B)** Split data into train/validation/test; monitor metrics during training; test on held-out data  
**C)** Train on all data

**Answer: B**

Fine-tuning evaluation: hold out test set (10-20%), track training/validation loss, check for overfitting, test final model on unseen data. Measure task-specific metrics (accuracy, F1, BLEU for translation).

---

### Q61: What is transfer learning in foundation models?

**A)** Transferring models between accounts  
**B)** Leveraging knowledge from pre-trained model for new but related tasks  
**C)** Moving data between systems

**Answer: B**

Transfer learning: pre-trained model's knowledge (language understanding, patterns) transfers to new tasks. Foundation models are pre-trained once, then transfer learned via fine-tuning to specific tasks (sentiment analysis, summarization). More efficient than training from scratch.

---

### Q62: What is continuous pre-training?

**A)** Same as fine-tuning  
**B)** Additional pre-training on domain-specific corpus before task-specific fine-tuning  
**C)** Never-ending training

**Answer: B**

Continuous pre-training: further pre-train foundation model on large domain corpus (medical texts, legal documents, code) to adapt vocabulary and knowledge. Then fine-tune on specific task. Bridges gap between general pre-training and task specialization.

---

### Q63: What is data curation for fine-tuning?

**A)** Random data collection  
**B)** Systematic selection, cleaning, and quality control of training examples  
**C)** Storing data only

**Answer: B**

Data curation: select relevant examples, remove duplicates/errors, filter low-quality data, ensure diversity, balance classes. Quality over quantity. Bad data leads to poor model performance. Includes deduplication, outlier removal, consistency checks.

---

### Q64: What is data governance in FM training?

**A)** Ignoring data policies  
**B)** Ensuring data compliance, privacy, licensing, and ethical use  
**C)** Data storage only

**Answer: B**

Data governance: verify data rights/licenses, protect PII, comply with regulations (GDPR, HIPAA), document data lineage, implement access controls. Critical for enterprise fine-tuning. Audit trail for model decisions. AWS: Lake Formation for governance.

---

### Q65: What is data labeling for supervised fine-tuning?

**A)** Automatic only  
**B)** Human annotation of training examples with correct outputs  
**C)** Not needed

**Answer: B**

Data labeling: humans provide correct answers/classifications for training examples. Quality labels essential for supervised learning. Use SageMaker Ground Truth for labeling workflows. Ensure labeler agreement, clear guidelines, quality checks. Expensive but critical.

---

### Q66: What is data representativeness in training data?

**A)** All examples identical  
**B)** Training data reflects real-world diversity and edge cases  
**C)** Size only matters

**Answer: B**

Representativeness: training data covers demographics, scenarios, languages, edge cases proportionally to production usage. Prevents bias toward over-represented groups. Test on representative validation set. Imbalanced data leads to biased models.

---

### Q67: What is the typical data size requirement for fine-tuning?

**A)** 10 examples sufficient  
**B)** Depends on task complexity: 100s for simple, 1000s-10000s for complex  
**C)** Always need millions

**Answer: B**

Data size: simple tasks (format conversion) need 100-500 examples. Complex tasks (domain expertise, reasoning) need 1000-10000+. Quality matters more than quantity. Bedrock fine-tuning minimums vary by model (typically 32-100).

---

### Q68: What is domain adaptation in fine-tuning?

**A)** Changing model architecture  
**B)** Specializing general model for specific industry or knowledge domain  
**C)** DNS configuration

**Answer: B**

Domain adaptation: fine-tune general model on domain-specific data (medical, legal, financial). Learns terminology, conventions, domain knowledge. More effective than prompting for consistent domain expertise. Example: adapt GPT to medical diagnosis support.

---

### Q69: What are the key hyperparameters for fine-tuning?

**A)** No parameters needed  
**B)** Learning rate, batch size, epochs, warmup steps  
**C)** Only model size

**Answer: B**

Hyperparameters: learning rate (how much to update weights, typically 1e-5 to 1e-4), batch size (examples per update), epochs (passes through data, 3-5 typical), warmup steps (gradual learning rate increase). Balance learning speed vs stability.

---

### Q70: What is overfitting in fine-tuning and how to prevent it?

**A)** Model works too well  
**B)** Model memorizes training data, performs poorly on new data; prevent with validation, regularization, early stopping  
**C)** Not a real problem

**Answer: B**

Overfitting: model learns training examples too specifically, doesn't generalize. Signs: training loss decreases but validation loss increases. Prevention: hold-out validation set, early stopping, dropout, smaller learning rate, more diverse data. Monitor validation metrics.

---

### Q71: What is parameter-efficient fine-tuning (PEFT)?

**A)** Full model retraining  
**B)** Updating only small subset of parameters (LoRA, adapters) to reduce cost  
**C)** No training

**Answer: B**

PEFT techniques (LoRA, adapters): freeze most model weights, train only small additional parameters. Reduces compute cost, memory, training time. Maintains most of original model knowledge. Bedrock supports LoRA for efficient customization.

---

### Q72: What is LoRA (Low-Rank Adaptation)?

**A)** Full fine-tuning  
**B)** Fine-tuning technique that adds small trainable matrices to frozen model weights  
**C)** Data preprocessing

**Answer: B**

LoRA: inserts small trainable rank-decomposition matrices into model layers while freezing original weights. Trains <1% of parameters vs full fine-tuning. Faster, cheaper, stores multiple adaptations efficiently. Bedrock fine-tuning uses LoRA approach.

---

### Q73: What is the difference between few-shot learning and few-shot fine-tuning?

**A)** Identical concepts  
**B)** Few-shot learning uses examples in prompt; few-shot fine-tuning trains on small dataset  
**C)** No difference in practice

**Answer: B**

Few-shot learning: provide examples in inference prompt (no training). Few-shot fine-tuning: train model on small dataset (<100 examples) to update weights. Fine-tuning provides more consistent behavior but requires training infrastructure.

---

### Q74: What is data quality vs quantity trade-off in fine-tuning?

**A)** Quantity always wins  
**B)** High-quality diverse examples more valuable than large low-quality dataset  
**C)** Quality doesn't matter

**Answer: B**

Quality priority: 500 clean, diverse, correctly labeled examples better than 5000 noisy, repetitive ones. Quality factors: correctness, diversity, representativeness, consistency. Invest in curation and labeling. Garbage in, garbage out principle.

---

### Q75: What is synthetic data generation for training?

**A)** Fake useless data  
**B)** Using existing models to generate additional training examples  
**C)** Manual data entry

**Answer: B**

Synthetic data: use strong model (GPT-4, Claude) to generate training examples for weaker/specialized model. Useful when real data scarce or sensitive. Example: generate customer service scenarios for chatbot training. Validate quality, mix with real data. Cost-effective augmentation.

---

## 3.4 Evaluating Foundation Model Performance (Q76-85)

### Q76: What are the main categories of FM evaluation metrics?

**A)** Speed only  
**B)** Quality (accuracy, relevance), safety (toxicity, bias), and performance (latency, cost)  
**C)** Cost only

**Answer: B**

Evaluation dimensions: Quality (task accuracy, output relevance), Safety (harmful content, bias, hallucinations), Performance (latency, throughput, cost). Holistic evaluation covers all dimensions for production readiness.

---

### Q77: What is BLEU score used for?

**A)** Sentiment analysis  
**B)** Evaluating translation/text generation quality by comparing to reference texts  
**C)** Image quality

**Answer: B**

BLEU (Bilingual Evaluation Understudy): measures overlap between generated and reference texts. Used for translation, summarization. Scale 0-1 (higher better). Limitation: doesn't capture semantic similarity, only n-gram overlap.

---

### Q78: What is ROUGE score?

**A)** Color metric  
**B)** Evaluating summarization by measuring recall of reference summary words  
**C)** Speed metric

**Answer: B**

ROUGE (Recall-Oriented Understudy for Gisting Evaluation): measures overlap with reference summaries. ROUGE-N (n-grams), ROUGE-L (longest common subsequence). Complementary to BLEU (recall vs precision focus).

---

### Q79: What is human evaluation for foundation models?

**A)** Automated only  
**B)** Having humans rate outputs for quality, helpfulness, harmfulness  
**C)** Not reliable

**Answer: B**

Human eval: raters assess responses (thumbs up/down, Likert scale, pairwise comparison). Captures nuance automated metrics miss (coherence, helpfulness, tone). Gold standard but expensive; sample representative subset.

---

### Q80: What is A/B testing for FM applications?

**A)** Testing two models  
**B)** Comparing two variants (prompts, models, parameters) with real users to measure impact  
**C)** Database testing

**Answer: B**

A/B testing: split traffic between variants (e.g., two prompts), measure metrics (user satisfaction, task completion, engagement). Statistical significance required. Gradual rollout minimizes risk of poor changes.

---

### Q81: What is hallucination detection in FM evaluation?

**A)** Ignoring incorrect outputs  
**B)** Identifying when model generates false or unsupported information  
**C)** Checking spelling only

**Answer: B**

Hallucination: model generates plausible but incorrect facts. Detection: compare output to source documents (RAG), cross-check facts, ask model for citations, use verification prompts ("Are you certain?"). Critical for factual domains.

---

### Q82: What is toxicity/safety evaluation?

**A)** Not important  
**B)** Measuring presence of harmful, offensive, or biased content in outputs  
**C)** Grammar checking

**Answer: B**

Safety evaluation: automated classifiers detect toxicity, hate speech, bias, sexual content. Benchmarks: ToxiGen, RealToxicityPrompts. Bedrock Guardrails provides automated filtering. Test across diverse inputs including adversarial prompts.

---

### Q83: What is bias evaluation in foundation models?

**A)** Checking file size  
**B)** Assessing fairness across demographic groups, stereotypes in outputs  
**C)** Speed testing

**Answer: B**

Bias evaluation: test for gender/race/age stereotypes, unfair treatment. Methods: test with demographic variations ("male nurse" vs "female nurse"), analyze embeddings, measure outcome parity. SageMaker Clarify assists with bias detection.

---

### Q84: What is latency vs accuracy trade-off in FM evaluation?

**A)** No trade-off exists  
**B)** Larger/better models typically slower; must balance quality needs with speed requirements  
**C)** Always prioritize speed

**Answer: B**

Trade-off: larger models (Claude Opus) more accurate but slower/costlier; smaller models (Haiku) faster/cheaper but less capable. Choose based on use case: chatbot needs speed, complex analysis tolerates latency.

---

### Q85: What is benchmark dataset evaluation?

**A)** Custom testing only  
**B)** Testing model on standardized datasets (MMLU, HellaSwag) to compare capabilities  
**C)** Not useful

**Answer: B**

Benchmarks: standardized tests for comparison. MMLU (multitask knowledge), HellaSwag (commonsense), HumanEval (code). Model providers publish scores. Use benchmarks to select models, but also test on your specific data.

---

### Q86: What is BERTScore?

**A)** Model size metric  
**B)** Semantic similarity metric using contextual embeddings to compare generated and reference text  
**C)** Speed benchmark

**Answer: B**

BERTScore: uses BERT embeddings to compute similarity between tokens in generated and reference text. Captures semantic meaning better than BLEU/ROUGE (word overlap). Computes precision, recall, F1 based on embedding similarity. Better for paraphrasing evaluation.

---

### Q87: What is perplexity in language model evaluation?

**A)** User confusion  
**B)** Measure of how well model predicts text; lower is better  
**C)** Processing time

**Answer: B**

Perplexity: measures prediction uncertainty. Lower perplexity = model more confident/accurate. Calculates exponential of average negative log-likelihood. Used for comparing language models. Limitation: doesn't measure output quality directly, only prediction probability.

---

### Q88: What are precision, recall, and F1 score in FM evaluation?

**A)** Speed metrics  
**B)** Precision: accuracy of positive predictions; Recall: coverage of actual positives; F1: harmonic mean  
**C)** Cost metrics

**Answer: B**

Precision: of items model says are positive, how many are correct. Recall: of actual positives, how many model found. F1: balances precision/recall. Used for classification, named entity recognition, information extraction tasks. Trade-off between false positives and false negatives.

---

### Q89: What is exact match accuracy?

**A)** Approximate matching  
**B)** Percentage of predictions that exactly match reference answers  
**C)** Partial credit scoring

**Answer: B**

Exact match: binary metric (correct/incorrect), no partial credit. Used for QA tasks where answer must be precisely right (dates, numbers, names). Strict but clear. Example: "What year?" → "2024" matches, "In 2024" doesn't. Combine with F1 for comprehensive evaluation.

---

### Q90: How do you measure user engagement for FM applications?

**A)** Only count users  
**B)** Task completion rate, session duration, return rate, interaction depth  
**C)** Server uptime

**Answer: B**

User engagement metrics: task completion rate (did user finish?), time on task (efficiency), return/retention rate, interactions per session, thumbs up/down, feature adoption. Tracks whether model meets user needs. A/B test features to optimize engagement.

---

### Q91: How do you measure productivity improvements from FM applications?

**A)** Subjective feelings  
**B)** Time saved, throughput increase, error reduction, cost per task  
**C)** Server metrics only

**Answer: B**

Productivity metrics: time saved per task (before/after FM), throughput (tasks/hour), quality improvement (fewer errors), cost reduction (automation ROI). Compare baseline (manual/old system) to FM-powered process. Document efficiency gains for business justification.

---

### Q92: What is business ROI evaluation for FM applications?

**A)** Ignore business impact  
**B)** Compare costs (development, inference, maintenance) to benefits (revenue, savings, productivity)  
**C)** Only measure technical metrics

**Answer: B**

ROI calculation: costs (development, fine-tuning, inference, storage, maintenance) vs benefits (revenue increase, cost savings, productivity gains, customer satisfaction). Payback period, NPV analysis. Track both quantitative (dollars) and qualitative (brand, satisfaction) value.

---

### Q93: What is the difference between online and offline evaluation?

**A)** Internet connectivity  
**B)** Offline: pre-deployment testing on datasets; Online: real-world monitoring with users  
**C)** No difference

**Answer: B**

Offline: test on held-out datasets before deployment, fast/cheap, doesn't capture real user behavior. Online: monitor in production, real user feedback, captures edge cases, slower/costly. Use both: offline for development, online for continuous improvement. Online metrics matter most.

---

### Q94: What are evaluation frameworks for FM applications?

**A)** No frameworks exist  
**B)** Tools like LangSmith, Promptfoo, LangChain evaluators that automate testing  
**C)** Manual testing only

**Answer: B**

Evaluation frameworks: LangSmith (LangChain evaluation), Promptfoo (prompt testing), custom test suites. Automate: run test cases, compare outputs, track metrics over time, regression testing. Enable CI/CD for prompts. Standardize evaluation across team.

---

### Q95: What is domain-specific evaluation?

**A)** Generic testing only  
**B)** Evaluation using domain experts and specialized metrics relevant to industry  
**C)** Ignoring domain context

**Answer: B**

Domain-specific eval: medical (clinical accuracy, safety), legal (precedent accuracy, compliance), financial (regulatory adherence). Requires domain experts to assess correctness. Generic benchmarks insufficient. Create custom test sets with domain scenarios. Higher standards for regulated industries.

---

### Q96: What is annotation quality and inter-rater agreement?

**A)** Single annotator sufficient  
**B)** Multiple annotators must agree; measure with Cohen's kappa, Fleiss' kappa  
**C)** Agreement doesn't matter

**Answer: B**

Inter-rater agreement: multiple humans label same examples, measure consistency. Cohen's kappa (2 raters), Fleiss' kappa (3+ raters). High agreement (>0.8) = reliable labels. Low agreement = ambiguous task or poor guidelines. Quality evaluation data requires consensus.

---

### Q97: What are confidence scores in FM outputs?

**A)** Model is always confident  
**B)** Probability/likelihood scores indicating model's certainty in predictions  
**C)** User ratings

**Answer: B**

Confidence scores: model outputs probability distribution over tokens/answers. High confidence (>0.9) = likely correct. Low confidence = uncertain, may need human review. Use thresholds: auto-accept high confidence, escalate low. Calibration important: confidence should match actual accuracy.

---

### Q98: How do you measure task engineering effectiveness?

**A)** Ignore task design  
**B)** Compare outcomes with different task formulations: prompts, examples, constraints  
**C)** Single approach only

**Answer: B**

Task engineering eval: test different formulations (zero-shot vs few-shot, different prompts, output formats). Measure task completion success rate, output quality, user satisfaction. Iterate on task design. Best formulation varies by use case. A/B test task variations.

---

### Q99: What is model comparison methodology?

**A)** Pick randomly  
**B)** Test multiple models on same task/data; compare metrics, cost, latency  
**C)** Use most expensive model

**Answer: B**

Model comparison: define evaluation criteria (accuracy, latency, cost), create representative test set, run all candidate models, compare metrics, consider trade-offs. Weigh factors by importance. Document decision rationale. Retest periodically as models improve.

---

### Q100: What are common evaluation pitfalls to avoid?

**A)** No pitfalls exist  
**B)** Overfitting to test set, unrepresentative data, ignoring edge cases, metric gaming  
**C)** Testing is perfect

**Answer: B**

Pitfalls: optimizing for metric instead of actual quality (gaming), test data leakage into training, unrepresentative test sets, ignoring failure modes, not testing adversarial inputs, only automated metrics (missing human nuance), insufficient sample size, forgetting business objectives. Holistic evaluation essential.

---

## Foundation Model Applications (Q101-108)

### Q101: What are the main application categories for foundation models?

**A)** Only text generation  
**B)** Text, images, code, and multimodal  
**C)** Only data analysis

**Answer: B**

Foundation models power: text generation (chatbots, content), image generation (art, product viz), code generation (development assist), multimodal (text+image, video). Diverse use cases across industries.

---

### Q102: What is a chatbot built on foundation models?

**A)** Rule-based Q&A system  
**B)** Conversational AI using LLMs for natural dialogue  
**C)** Pre-recorded responses

**Answer: B**

LLM-powered chatbots understand context, maintain conversation history, handle complex queries. Use models like Claude, GPT for customer service, support, virtual assistants. More flexible than rule-based.

---

### Q103: What is content generation using GenAI?

**A)** Copying existing content  
**B)** Creating original marketing copy, articles, summaries  
**C)** Data analysis only

**Answer: B**

Content generation: create marketing copy, blog posts, product descriptions, social media, email drafts. Automates repetitive writing tasks. Still needs human review for accuracy/brand voice.

---

### Q104: What is code generation with foundation models?

**A)** Compiling code  
**B)** Writing code from natural language descriptions  
**C)** Debugging only

**Answer: B**

Code generation: convert requirements to code, complete functions, generate tests, write documentation. Examples: CodeWhisperer, GitHub Copilot. Accelerates development, reduces boilerplate.

---

### Q105: What are multimodal foundation models?

**A)** Single data type only  
**B)** Handle multiple input/output types (text, images, audio)  
**C)** Multiple models combined

**Answer: B**

Multimodal models process and generate across data types: text-to-image (Stable Diffusion), image-to-text (vision models), audio-to-text (Transcribe). Single model, multiple modalities.

---

### Q106: What is document summarization?

**A)** Highlighting keywords  
**B)** Condensing long documents to key points  
**C)** Translating documents

**Answer: B**

Summarization: extract main ideas from long documents (reports, research, articles). Saves time, improves information access. Can specify length (executive summary vs detailed). Uses extractive or abstractive methods.

---

### Q107: What is sentiment analysis in GenAI?

**A)** Grammar checking  
**B)** Determining emotional tone (positive, negative, neutral)  
**C)** Translation

**Answer: B**

Sentiment analysis: classify text emotion/opinion. Applications: customer feedback, social media monitoring, product reviews. Foundation models understand nuance better than traditional ML.

---

### Q108: What is Amazon Bedrock Agents?

**A)** Human customer service  
**B)** Autonomous AI completing multi-step tasks  
**C)** Model training service

**Answer: B**

Bedrock Agents: orchestrate complex workflows, call APIs, access data sources, break down tasks. Example: "Book a flight" → search flights, check preferences, make reservation. Agentic AI.

---

## Conversational AI (Q109-114)

### Q109: What is the key advantage of LLM-based chatbots?

**A)** Faster than rule-based  
**B)** Understand context and handle ambiguity  
**C)** Never make mistakes

**Answer: B**

LLM chatbots understand natural language nuance, maintain context across turns, adapt to unexpected queries. Unlike rule-based (limited to predefined paths). Still need guardrails and monitoring.

---

### Q110: What is conversation memory in chatbots?

**A)** Database storage  
**B)** Maintaining context across message exchanges  
**C)** User authentication

**Answer: B**

Conversation memory: chatbot remembers previous messages in session. Enables coherent multi-turn conversations. Implementations: include full history in prompt (token limit aware) or summarize.

---

### Q111: How do you handle sensitive topics in chatbots?

**A)** Ignore them  
**B)** Use guardrails to detect and redirect  
**C)** Let model answer freely

**Answer: B**

Guardrails detect sensitive topics (medical advice, legal, financial). Respond with: "I can't provide medical advice. Please consult a professional." Prevents liability and harm.

---

### Q112: What is intent recognition in conversational AI?

**A)** User authentication  
**B)** Understanding what user wants to accomplish  
**C)** Language translation

**Answer: B**

Intent recognition: classify user goal (book appointment, check status, get refund). Routes to appropriate handler. LLMs excel at understanding intents from natural language variations.

---

### Q113: What is Amazon Lex used for?

**A)** Image generation  
**B)** Building voice and text chatbots  
**C)** Data storage

**Answer: B**

Lex: AWS service for chatbots with ASR (speech recognition) and NLU (language understanding). Powers Alexa. Integrates with Lambda for business logic. Pre-GenAI service, now enhanced with Bedrock.

---

### Q114: How do you evaluate chatbot quality?

**A)** Speed only  
**B)** User satisfaction, task completion, accuracy  
**C)** Cost only

**Answer: B**

Metrics: user satisfaction scores (thumbs up/down), task completion rate, response accuracy, escalation frequency, conversation length. A/B test improvements. Monitor for failure patterns.

---

## Search & Knowledge Retrieval (Q115-120)

### Q115: What is semantic search?

**A)** Keyword matching  
**B)** Understanding meaning and intent, not just exact words  
**C)** Random search

**Answer: B**

Semantic search: finds conceptually similar content using embeddings. "Car problems" matches "vehicle issues", "automobile trouble". Better than keyword-only (misses synonyms, context).

---

### Q116: What is the role of embeddings in search?

**A)** Encrypt data  
**B)** Convert text to vectors for similarity comparison  
**C)** Compress files

**Answer: B**

Embeddings map text to vector space where similar meanings cluster. Search: convert query to vector, find nearest document vectors (cosine similarity). Enables semantic search in RAG.

---

### Q117: What is Amazon Kendra?

**A)** Foundation model  
**B)** Intelligent enterprise search using ML  
**C)** Storage service

**Answer: B**

Kendra: ML-powered search for enterprise documents. Understands natural language questions, ranks results by relevance, extracts answers. Indexes documents across S3, SharePoint, databases. Pre-GenAI, enhanceable with RAG.

---

### Q118: How does RAG improve search applications?

**A)** Faster indexing  
**B)** Generates direct answers from retrieved documents  
**C)** Stores more data

**Answer: B**

RAG: retrieve relevant docs + generate natural language answer. Users get direct answers, not just document links. Cites sources for verification. Combines search + generation.

---

### Q119: What is re-ranking in search?

**A)** Initial retrieval  
**B)** Second-pass scoring to refine result relevance  
**C)** Sorting by date

**Answer: B**

Re-ranking: after initial retrieval (e.g., top 20 docs), use sophisticated model to score relevance deeply. Return best 3-5 to LLM. Improves precision vs single-pass retrieval.

---

### Q120: What is query expansion in semantic search?

**A)** Making queries longer  
**B)** Generating alternative query phrasings to improve recall  
**C)** Translating queries

**Answer: B**

Query expansion: LLM generates similar questions/phrasings for user query. Search with all variations, combine results. Improves recall (find more relevant docs). Example: "Python loops" → add "iteration", "for/while".

---

## Code & Developer Tools (Q121-125)

### Q121: What can Amazon CodeWhisperer do?

**A)** Only syntax highlighting  
**B)** Real-time code suggestions, completions, security scanning  
**C)** Version control

**Answer: B**

CodeWhisperer: AI pair programmer providing inline code suggestions, function generation from comments, security vulnerability detection, license compliance. Supports Python, Java, JavaScript, TypeScript, etc.

---

### Q122: What is code explanation using LLMs?

**A)** Running code  
**B)** Generating natural language descriptions of code logic  
**C)** Compiling code

**Answer: B**

Code explanation: LLM reads code, describes functionality in plain language. Helps understand unfamiliar code, documents legacy systems, onboards developers. Example: "This function validates email format using regex."

---

### Q123: What is test generation using GenAI?

**A)** Running tests  
**B)** Automatically creating unit tests from code  
**C)** Debugging

**Answer: B**

Test generation: LLM analyzes function, generates unit tests covering edge cases, happy paths, error conditions. Saves time, improves coverage. Developer reviews/refines generated tests.

---

### Q124: What is code refactoring assistance?

**A)** Deleting code  
**B)** Suggesting improvements for code quality, performance  
**C)** Commenting code

**Answer: B**

Refactoring assistance: LLM suggests cleaner, more efficient code. Improves readability, performance, maintainability. Examples: simplify logic, modernize syntax, apply design patterns. Developer validates changes.

---

### Q125: What is bug fixing with GenAI?

**A)** Automatic deployment  
**B)** Suggesting fixes for identified errors  
**C)** Preventing all bugs

**Answer: B**

Bug fixing: provide error message + code, LLM suggests potential fixes with explanations. Accelerates debugging. Examples: null pointer fixes, type errors, logic corrections. Developer tests solutions.

---

## Specialized Applications (Q126-130)

### Q126: What is AI-powered personalization?

**A)** Same experience for all users  
**B)** Tailoring content/recommendations to individual users  
**C)** User authentication

**Answer: B**

Personalization: LLMs generate customized content, recommendations, emails based on user context, preferences, behavior. Examples: personalized product descriptions, dynamic email content, adaptive learning paths.

---

### Q127: What is document extraction using GenAI?

**A)** Deleting documents  
**B)** Extracting structured data from unstructured documents  
**C)** Storing documents

**Answer: B**

Document extraction: parse contracts, invoices, forms to extract key fields (dates, amounts, names). Combines vision models (read PDFs/images) + LLMs (understand structure). AWS: Textract + Bedrock.

---

### Q128: What is knowledge base question answering?

**A)** General chatbot  
**B)** Answering questions from company-specific documents  
**C)** Web search

**Answer: B**

Knowledge base Q&A: RAG system over company documents (policies, procedures, manuals). Employees ask questions, get accurate answers with citations. Reduces support tickets, improves knowledge access.

---

### Q129: What is meeting summarization?

**A)** Scheduling meetings  
**B)** Generating concise summaries of meeting transcripts  
**C)** Video conferencing

**Answer: B**

Meeting summarization: transcribe meeting (Transcribe), summarize key points, action items, decisions (LLM). Saves time, ensures alignment. Example: "Team decided to launch feature in Q2, John owns implementation."

---

### Q130: What are AI agents in enterprise applications?

**A)** Human employees  
**B)** Autonomous AI systems completing business workflows  
**C)** Monitoring tools

**Answer: B**

AI agents: orchestrate multi-step business tasks. Examples: expense report processing (extract receipts, validate, submit), customer onboarding (collect info, verify, create account). Bedrock Agents framework.

---

## Enterprise & Industry Applications (Q131-138)

### Q131: What is AI-powered customer service routing?

**A)** Random assignment  
**B)** Using LLMs to classify and route customer inquiries to right team  
**C)** Manual sorting

**Answer: B**

LLM analyzes customer message, determines intent/urgency/complexity, routes to appropriate department/agent. Reduces wait times, improves resolution rates. Replaces rigid rule-based routing.

---

### Q132: What is automated report generation?

**A)** Manual reporting  
**B)** GenAI creates business reports from data/templates  
**C)** Report storage

**Answer: B**

LLM generates executive summaries, financial reports, performance analyses from structured data. Combines data retrieval with natural language generation. Saves hours of manual writing.

---

### Q133: What is contract analysis with GenAI?

**A)** Contract storage  
**B)** Extracting clauses, identifying risks, comparing terms  
**C)** Printing contracts

**Answer: B**

LLM reads contracts, extracts key terms (dates, obligations, penalties), flags risks, compares against standards. Legal teams use for due diligence, compliance. Textract + Bedrock combination.

---

### Q134: What is email response automation?

**A)** Auto-reply "out of office"  
**B)** GenAI drafts contextual responses to customer emails  
**C)** Email filtering

**Answer: B**

LLM reads customer email, understands context, drafts appropriate response. Human reviews before sending. Use cases: support, inquiries, FAQs. Maintains brand voice through prompt engineering.

---

### Q135: What is product description generation?

**A)** Copying manufacturer specs  
**B)** Creating unique, engaging product descriptions at scale  
**C)** Product images

**Answer: B**

LLM generates descriptions from product attributes (size, color, features). Optimizes for SEO, tone, length. E-commerce use: thousands of products need unique descriptions. Personalization possible.

---

### Q136: What is AI-powered data enrichment?

**A)** Data backup  
**B)** Enhancing records with additional generated information  
**C)** Data compression

**Answer: B**

LLM adds missing fields, standardizes formats, generates summaries, categorizes data. Example: incomplete customer profiles → generate industry, company size, likely needs from available info.

---

### Q137: What is voice-of-customer analysis?

**A)** Recording calls  
**B)** Analyzing feedback at scale for themes, sentiment, insights  
**C)** Customer surveys only

**Answer: B**

LLM processes reviews, support tickets, social media, surveys. Identifies patterns, pain points, feature requests. Summarizes thousands of inputs into actionable insights. Replaces manual reading.

---

### Q138: What is legal document drafting assistance?

**A)** Replacing lawyers  
**B)** Generating first drafts of contracts, agreements, clauses  
**C)** Document storage

**Answer: B**

LLM creates initial document drafts from templates and requirements. Lawyer reviews, edits, finalizes. Saves time on boilerplate. Not for final legal advice—human oversight required.

---

## Creative & Media Applications (Q139-140)

### Q139: What is AI-powered video script generation?

**A)** Video editing  
**B)** Creating video scripts from brief or topic  
**C)** Video hosting

**Answer: B**

LLM writes video scripts: scenes, dialogue, narration. Input: topic, duration, audience. Output: structured script. Use for marketing videos, training content, social media. Human edits for brand fit.

---

### Q140: What is style transfer in image generation?

**A)** Copying images  
**B)** Applying artistic style to images (e.g., "like Van Gogh")  
**C)** Image compression

**Answer: B**

Image models apply artistic styles to photos. Input: content image + style reference. Output: blended result. Stable Diffusion supports this. Use cases: creative design, product visualization.

---

## Exam Tips

**Key Concepts to Remember:**

1. **3.0 Applications of Foundation Models (Q1-10):**

   - **Application Categories:** Text, code, images, multimodal
   - **RAG Architecture:** Retrieval + generation for grounded answers
   - **Agentic AI:** Multi-step autonomous task execution
   - **Invocation Patterns:** Synchronous vs asynchronous
   - **Streaming:** Real-time token delivery for better UX
   - **Orchestration:** Coordinating multiple services/models

2. **3.1 Design Considerations (Q11-30):**

   - **Model Selection:** Cost, modality, latency, language support, size, customization, context length
   - **Inference Parameters:** Temperature (randomness), top-p/top-k (diversity), max_tokens (output length)
   - **Latency Management:** Model size, context length, streaming
   - **Context Window:** Token limits, truncation strategies
   - **Cost Optimization:** Token minimization, caching, provisioned throughput
   - **Error Handling:** Retries, fallbacks, graceful degradation
   - **Security:** Input validation, guardrails, PII protection
   - **Observability:** Logging, monitoring, tracing
   - **Scalability:** Auto-scaling, queue-based processing
   - **RAG Deep Dive:** Retrieval workflow, vector databases, Bedrock Knowledge Bases
   - **Vector Databases:** OpenSearch, Aurora, Neptune, DocumentDB, RDS PostgreSQL (pgvector)

3. **3.2 Prompt Engineering (Q31-50):**

   - **Zero-shot:** Direct instruction without examples
   - **Single-shot:** One example provided
   - **Few-shot:** 2-5 examples to guide behavior
   - **Chain-of-thought:** Step-by-step reasoning
   - **Prompt Structure:** Context, instruction, input, output format, constraints
   - **Model Latent Space:** Internal representation space for concepts
   - **Delimiters:** Separate instructions from untrusted input
   - **System vs User Prompts:** Role/rules vs specific requests
   - **Templates:** Reusable structures with parameters
   - **Negative Prompting:** Explicit constraints
   - **Guardrails:** Constraint rules and content filtering
   - **Discovery & Experimentation:** Systematic testing and optimization
   - **Security Risks:** Prompt injection, hijacking, poisoning, jailbreaking, exposure
   - **Iterative Refinement:** Test, measure, improve

4. **3.3 Training & Fine-Tuning (Q51-75):**

   - **Pre-training:** Initial large-scale training (expensive, done once)
   - **Fine-tuning:** Domain specialization with custom data
   - **Transfer Learning:** Leveraging pre-trained knowledge for new tasks
   - **Continuous Pre-training:** Domain-specific corpus training before fine-tuning
   - **Instruction Tuning:** Training on instruction-response pairs
   - **RLHF:** Human feedback for alignment
   - **Data Curation:** Selection, cleaning, quality control
   - **Data Governance:** Compliance, privacy, licensing, ethical use
   - **Data Labeling:** Human annotation with quality checks
   - **Data Representativeness:** Coverage of demographics, scenarios, edge cases
   - **Data Size:** 100s for simple tasks, 1000s-10000s for complex
   - **Domain Adaptation:** Specializing for industries (medical, legal, financial)
   - **Hyperparameters:** Learning rate, batch size, epochs, warmup steps
   - **Overfitting Prevention:** Validation sets, early stopping, regularization
   - **Parameter-Efficient Fine-Tuning (PEFT):** LoRA, adapters for cost reduction
   - **LoRA:** Low-rank adaptation with trainable matrices
   - **Few-shot Fine-tuning:** Training on small datasets (<100 examples)
   - **Data Quality vs Quantity:** High-quality diverse examples prioritized
   - **Synthetic Data:** Using models to generate training examples
   - **Continuous Fine-tuning:** Regular updates with new data

5. **3.4 Model Evaluation (Q76-100):**

   - **Evaluation Dimensions:** Quality, safety, performance
   - **Automated Metrics:** BLEU (translation), ROUGE (summarization), BERTScore (semantic similarity)
   - **Perplexity:** Language model prediction confidence
   - **Classification Metrics:** Precision, recall, F1 score
   - **Exact Match Accuracy:** Strict correctness evaluation
   - **Human Evaluation:** Rating quality, helpfulness, harm
   - **A/B Testing:** Comparing variants with real users
   - **Hallucination Detection:** Verifying factual accuracy
   - **Safety Testing:** Toxicity, bias, harmful content
   - **User Engagement Metrics:** Task completion, session duration, return rate
   - **Productivity Metrics:** Time saved, throughput increase, error reduction
   - **Business ROI:** Cost-benefit analysis, payback period
   - **Online vs Offline Evaluation:** Pre-deployment testing vs production monitoring
   - **Evaluation Frameworks:** LangSmith, Promptfoo, LangChain evaluators
   - **Domain-Specific Evaluation:** Expert assessment with industry metrics
   - **Inter-Rater Agreement:** Cohen's kappa, Fleiss' kappa for annotation quality
   - **Confidence Scores:** Model certainty and calibration
   - **Task Engineering Effectiveness:** Comparing task formulations
   - **Model Comparison:** Systematic evaluation across candidates
   - **Evaluation Pitfalls:** Metric gaming, test leakage, unrepresentative data
   - **Benchmarks:** MMLU, HellaSwag, HumanEval

6. **Practical Applications (Q101-140):**
   - **Conversational AI:** Context management, intent recognition, guardrails
   - **Search & Retrieval:** Semantic search, embeddings, RAG, re-ranking
   - **Code Tools:** CodeWhisperer for generation, testing, security
   - **Enterprise:** Document extraction, contract analysis, automation
   - **Creative:** Content generation, style transfer, multimedia

**AWS Services Map:**

- **Bedrock:** Foundation models, Agents, Guardrails, Knowledge Bases
- **CodeWhisperer:** AI code assistant
- **Kendra:** Enterprise search
- **Lex:** Chatbot building
- **Textract:** Document OCR
- **Comprehend:** NLP (PII detection, sentiment)
- **SageMaker Clarify:** Bias detection

**Study Focus:**

- Understand when to use prompting vs fine-tuning
- Know design patterns for production FM applications
- Master prompt engineering techniques (few-shot, CoT)
- Recognize evaluation methods and metrics
- Match use cases to appropriate AWS services
- Understand cost/latency/quality trade-offs
