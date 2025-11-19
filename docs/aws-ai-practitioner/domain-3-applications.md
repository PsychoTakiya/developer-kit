# Domain 3: Applications of Foundation Models

90 focused MCQs for AWS AI Practitioner exam preparation, organized by exam objectives.

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

## 3.2 Prompt Engineering Techniques (Q21-30)

### Q21: What is zero-shot prompting?

**A)** Providing many examples  
**B)** Asking model to perform task without examples, relying on pre-training  
**C)** Not providing any instructions

**Answer: B**

Zero-shot: direct instruction without examples. "Summarize this article." Works for common tasks where model has sufficient training. Simple, but less accurate than few-shot for complex/specific tasks.

---

### Q22: What is few-shot prompting?

**A)** No examples provided  
**B)** Including 2-5 examples of input-output pairs to guide model behavior  
**C)** Training the model

**Answer: B**

Few-shot: provide examples in prompt. "Classify sentiment: 'Great product!' → Positive, 'Terrible service' → Negative, [your input]." Improves accuracy and consistency for specific formats/domains.

---

### Q23: What is chain-of-thought (CoT) prompting?

**A)** Single-step answers  
**B)** Prompting model to show reasoning steps before final answer  
**C)** Random thinking

**Answer: B**

CoT: "Let's think step by step." Model explains reasoning, improving complex problem-solving (math, logic, analysis). Increases accuracy and provides transparency. Can combine with few-shot.

---

### Q24: What is the role of system prompts vs user prompts?

**A)** No difference  
**B)** System sets behavior/role; user provides specific request  
**C)** System prompts are ignored

**Answer: B**

System prompt: defines AI persona, rules, constraints ("You are a helpful financial advisor. Never provide investment advice."). User prompt: specific query. System sets context for all interactions.

---

### Q25: What is prompt template design?

**A)** Random text generation  
**B)** Reusable prompt structures with placeholders for dynamic content  
**C)** Hard-coded prompts only

**Answer: B**

Templates: "Summarize the following {document_type} in {length} sentences: {content}". Enables consistency, parameterization, testing. Store templates separately from code for easy updates.

---

### Q26: What is negative prompting?

**A)** Being rude to AI  
**B)** Explicitly stating what NOT to do or include  
**C)** Ignoring the model

**Answer: B**

Negative prompts: "Summarize but don't include technical jargon" or "Don't make up information." Guides model away from undesired behaviors. Useful for content filtering and constraint setting.

---

### Q27: What is the importance of prompt clarity and specificity?

**A)** Vague prompts work fine  
**B)** Clear, specific instructions reduce ambiguity and improve output quality  
**C)** Length doesn't matter

**Answer: B**

Specific prompts: "Write 3 bullet points summarizing key financial risks" better than "Summarize this." Include: format, length, style, constraints. Reduces iterations and hallucinations.

---

### Q28: What is role prompting?

**A)** Assigning user roles  
**B)** Instructing model to adopt specific persona or expertise  
**C)** Role-based access control

**Answer: B**

Role prompting: "You are an expert Python developer" or "Act as a customer service agent." Influences tone, knowledge depth, response style. Improves domain-specific accuracy.

---

### Q29: What is iterative prompt refinement?

**A)** Using first prompt always  
**B)** Testing and improving prompts based on output quality  
**C)** Ignoring results

**Answer: B**

Prompt engineering is iterative: test prompt → evaluate output → refine wording/examples → retest. Track versions, measure metrics (accuracy, relevance). A/B test prompt variations in production.

---

### Q30: What is the difference between instruction-following and completion-style prompts?

**A)** No difference  
**B)** Instruction gives command; completion provides beginning for model to continue  
**C)** Completion is always better

**Answer: B**

Instruction: "Translate to French: Hello" (command). Completion: "The three main benefits of cloud computing are" (model continues). Modern chat models prefer instructions; older models used completions.

---

## 3.3 Training and Fine-Tuning Foundation Models (Q31-40)

### Q31: What is pre-training in foundation models?

**A)** Final training step  
**B)** Initial training on massive general datasets to learn language patterns  
**C)** User-specific training

**Answer: B**

Pre-training: training on internet-scale text data (books, web, code) to learn language structure, facts, reasoning. Extremely expensive (millions of dollars), done once by model providers. Base for all downstream tasks.

---

### Q32: What is fine-tuning a foundation model?

**A)** Pre-training the model  
**B)** Additional training on specific dataset to specialize model for domain/task  
**C)** Prompt engineering

**Answer: B**

Fine-tuning: train pre-trained model on custom data (company documents, specific format, domain knowledge). Less data/cost than pre-training. Bedrock supports fine-tuning for select models. Improves task-specific accuracy.

---

### Q33: What is the difference between fine-tuning and prompt engineering?

**A)** No difference  
**B)** Fine-tuning modifies model weights; prompting provides runtime instructions  
**C)** Prompting is always better

**Answer: B**

Fine-tuning: permanent model changes, requires training data/compute, better for consistent specialized behavior. Prompting: flexible, no training needed, quick iteration. Start with prompting; fine-tune if needed.

---

### Q34: What is instruction fine-tuning?

**A)** Training on raw text  
**B)** Fine-tuning on instruction-response pairs to improve instruction-following  
**C)** Pre-training phase

**Answer: B**

Instruction tuning: train on datasets of instructions + desired responses. Makes model better at following commands. Example: (instruction: "Summarize", text: "...", response: "Summary: ..."). Commercial models already instruction-tuned.

---

### Q35: What is RLHF (Reinforcement Learning from Human Feedback)?

**A)** Supervised learning only  
**B)** Training using human ratings of responses to align model with preferences  
**C)** Automatic training

**Answer: B**

RLHF: humans rate model outputs (helpful/harmful), model learns from feedback. Improves alignment, safety, helpfulness. Used in ChatGPT, Claude. Advanced technique beyond initial supervised training.

---

### Q36: What data is needed for fine-tuning a foundation model?

**A)** No data required  
**B)** Domain-specific examples (100s-1000s) with inputs and desired outputs  
**C)** Internet-scale data

**Answer: B**

Fine-tuning data: task-specific examples (prompts + completions), typically 100-10,000 examples. Quality matters more than quantity. Format: JSONL with prompt/completion pairs. Bedrock requires min examples per model.

---

### Q37: What is continuous fine-tuning?

**A)** One-time training  
**B)** Regularly updating fine-tuned model with new data to maintain accuracy  
**C)** Pre-training phase

**Answer: B**

Continuous fine-tuning: periodic retraining as new data arrives (customer interactions, product updates). Keeps model current. Automate: collect feedback, retrain monthly/quarterly, evaluate, deploy. Lifecycle management.

---

### Q38: What is the cost consideration for fine-tuning vs prompting?

**A)** Both are free  
**B)** Fine-tuning has upfront training cost; prompting costs per inference  
**C)** Fine-tuning is always cheaper

**Answer: B**

Fine-tuning: training cost (compute hours) + hosting custom model. Prompting: per-token inference cost. Fine-tuning worthwhile for high-volume, specialized tasks. Prompting better for low-volume or varying tasks.

---

### Q39: What is model distillation?

**A)** Removing models  
**B)** Training smaller model to mimic larger model's behavior  
**C)** Data cleaning

**Answer: B**

Distillation: large "teacher" model generates training data for small "student" model. Student learns to approximate teacher with lower latency/cost. Trade-off: speed vs some accuracy loss.

---

### Q40: What is the evaluation process during fine-tuning?

**A)** No evaluation needed  
**B)** Split data into train/validation/test; monitor metrics during training; test on held-out data  
**C)** Train on all data

**Answer: B**

Fine-tuning evaluation: hold out test set (10-20%), track training/validation loss, check for overfitting, test final model on unseen data. Measure task-specific metrics (accuracy, F1, BLEU for translation).

---

## 3.4 Evaluating Foundation Model Performance (Q41-50)

## 3.4 Evaluating Foundation Model Performance (Q41-50)

### Q41: What are the main categories of FM evaluation metrics?

**A)** Speed only  
**B)** Quality (accuracy, relevance), safety (toxicity, bias), and performance (latency, cost)  
**C)** Cost only

**Answer: B**

Evaluation dimensions: Quality (task accuracy, output relevance), Safety (harmful content, bias, hallucinations), Performance (latency, throughput, cost). Holistic evaluation covers all dimensions for production readiness.

---

### Q42: What is BLEU score used for?

**A)** Sentiment analysis  
**B)** Evaluating translation/text generation quality by comparing to reference texts  
**C)** Image quality

**Answer: B**

BLEU (Bilingual Evaluation Understudy): measures overlap between generated and reference texts. Used for translation, summarization. Scale 0-1 (higher better). Limitation: doesn't capture semantic similarity, only n-gram overlap.

---

### Q43: What is ROUGE score?

**A)** Color metric  
**B)** Evaluating summarization by measuring recall of reference summary words  
**C)** Speed metric

**Answer: B**

ROUGE (Recall-Oriented Understudy for Gisting Evaluation): measures overlap with reference summaries. ROUGE-N (n-grams), ROUGE-L (longest common subsequence). Complementary to BLEU (recall vs precision focus).

---

### Q44: What is human evaluation for foundation models?

**A)** Automated only  
**B)** Having humans rate outputs for quality, helpfulness, harmfulness  
**C)** Not reliable

**Answer: B**

Human eval: raters assess responses (thumbs up/down, Likert scale, pairwise comparison). Captures nuance automated metrics miss (coherence, helpfulness, tone). Gold standard but expensive; sample representative subset.

---

### Q45: What is A/B testing for FM applications?

**A)** Testing two models  
**B)** Comparing two variants (prompts, models, parameters) with real users to measure impact  
**C)** Database testing

**Answer: B**

A/B testing: split traffic between variants (e.g., two prompts), measure metrics (user satisfaction, task completion, engagement). Statistical significance required. Gradual rollout minimizes risk of poor changes.

---

### Q46: What is hallucination detection in FM evaluation?

**A)** Ignoring incorrect outputs  
**B)** Identifying when model generates false or unsupported information  
**C)** Checking spelling only

**Answer: B**

Hallucination: model generates plausible but incorrect facts. Detection: compare output to source documents (RAG), cross-check facts, ask model for citations, use verification prompts ("Are you certain?"). Critical for factual domains.

---

### Q47: What is toxicity/safety evaluation?

**A)** Not important  
**B)** Measuring presence of harmful, offensive, or biased content in outputs  
**C)** Grammar checking

**Answer: B**

Safety evaluation: automated classifiers detect toxicity, hate speech, bias, sexual content. Benchmarks: ToxiGen, RealToxicityPrompts. Bedrock Guardrails provides automated filtering. Test across diverse inputs including adversarial prompts.

---

### Q48: What is bias evaluation in foundation models?

**A)** Checking file size  
**B)** Assessing fairness across demographic groups, stereotypes in outputs  
**C)** Speed testing

**Answer: B**

Bias evaluation: test for gender/race/age stereotypes, unfair treatment. Methods: test with demographic variations ("male nurse" vs "female nurse"), analyze embeddings, measure outcome parity. SageMaker Clarify assists with bias detection.

---

### Q49: What is latency vs accuracy trade-off in FM evaluation?

**A)** No trade-off exists  
**B)** Larger/better models typically slower; must balance quality needs with speed requirements  
**C)** Always prioritize speed

**Answer: B**

Trade-off: larger models (Claude Opus) more accurate but slower/costlier; smaller models (Haiku) faster/cheaper but less capable. Choose based on use case: chatbot needs speed, complex analysis tolerates latency.

---

### Q50: What is benchmark dataset evaluation?

**A)** Custom testing only  
**B)** Testing model on standardized datasets (MMLU, HellaSwag) to compare capabilities  
**C)** Not useful

**Answer: B**

Benchmarks: standardized tests for comparison. MMLU (multitask knowledge), HellaSwag (commonsense), HumanEval (code). Model providers publish scores. Use benchmarks to select models, but also test on your specific data.

---

## Foundation Model Applications (Q51-58)

### Q51: What are the main application categories for foundation models?

**A)** Only text generation  
**B)** Text, images, code, and multimodal  
**C)** Only data analysis

**Answer: B**

Foundation models power: text generation (chatbots, content), image generation (art, product viz), code generation (development assist), multimodal (text+image, video). Diverse use cases across industries.

---

### Q52: What is a chatbot built on foundation models?

**A)** Rule-based Q&A system  
**B)** Conversational AI using LLMs for natural dialogue  
**C)** Pre-recorded responses

**Answer: B**

LLM-powered chatbots understand context, maintain conversation history, handle complex queries. Use models like Claude, GPT for customer service, support, virtual assistants. More flexible than rule-based.

---

### Q53: What is content generation using GenAI?

**A)** Copying existing content  
**B)** Creating original marketing copy, articles, summaries  
**C)** Data analysis only

**Answer: B**

Content generation: create marketing copy, blog posts, product descriptions, social media, email drafts. Automates repetitive writing tasks. Still needs human review for accuracy/brand voice.

---

### Q54: What is code generation with foundation models?

**A)** Compiling code  
**B)** Writing code from natural language descriptions  
**C)** Debugging only

**Answer: B**

Code generation: convert requirements to code, complete functions, generate tests, write documentation. Examples: CodeWhisperer, GitHub Copilot. Accelerates development, reduces boilerplate.

---

### Q55: What are multimodal foundation models?

**A)** Single data type only  
**B)** Handle multiple input/output types (text, images, audio)  
**C)** Multiple models combined

**Answer: B**

Multimodal models process and generate across data types: text-to-image (Stable Diffusion), image-to-text (vision models), audio-to-text (Transcribe). Single model, multiple modalities.

---

### Q56: What is document summarization?

**A)** Highlighting keywords  
**B)** Condensing long documents to key points  
**C)** Translating documents

**Answer: B**

Summarization: extract main ideas from long documents (reports, research, articles). Saves time, improves information access. Can specify length (executive summary vs detailed). Uses extractive or abstractive methods.

---

### Q57: What is sentiment analysis in GenAI?

**A)** Grammar checking  
**B)** Determining emotional tone (positive, negative, neutral)  
**C)** Translation

**Answer: B**

Sentiment analysis: classify text emotion/opinion. Applications: customer feedback, social media monitoring, product reviews. Foundation models understand nuance better than traditional ML.

---

### Q58: What is Amazon Bedrock Agents?

**A)** Human customer service  
**B)** Autonomous AI completing multi-step tasks  
**C)** Model training service

**Answer: B**

Bedrock Agents: orchestrate complex workflows, call APIs, access data sources, break down tasks. Example: "Book a flight" → search flights, check preferences, make reservation. Agentic AI.

---

## Conversational AI (Q59-64)

### Q59: What is the key advantage of LLM-based chatbots?

**A)** Faster than rule-based  
**B)** Understand context and handle ambiguity  
**C)** Never make mistakes

**Answer: B**

LLM chatbots understand natural language nuance, maintain context across turns, adapt to unexpected queries. Unlike rule-based (limited to predefined paths). Still need guardrails and monitoring.

---

### Q60: What is conversation memory in chatbots?

**A)** Database storage  
**B)** Maintaining context across message exchanges  
**C)** User authentication

**Answer: B**

Conversation memory: chatbot remembers previous messages in session. Enables coherent multi-turn conversations. Implementations: include full history in prompt (token limit aware) or summarize.

---

### Q61: How do you handle sensitive topics in chatbots?

**A)** Ignore them  
**B)** Use guardrails to detect and redirect  
**C)** Let model answer freely

**Answer: B**

Guardrails detect sensitive topics (medical advice, legal, financial). Respond with: "I can't provide medical advice. Please consult a professional." Prevents liability and harm.

---

### Q62: What is intent recognition in conversational AI?

**A)** User authentication  
**B)** Understanding what user wants to accomplish  
**C)** Language translation

**Answer: B**

Intent recognition: classify user goal (book appointment, check status, get refund). Routes to appropriate handler. LLMs excel at understanding intents from natural language variations.

---

### Q63: What is Amazon Lex used for?

**A)** Image generation  
**B)** Building voice and text chatbots  
**C)** Data storage

**Answer: B**

Lex: AWS service for chatbots with ASR (speech recognition) and NLU (language understanding). Powers Alexa. Integrates with Lambda for business logic. Pre-GenAI service, now enhanced with Bedrock.

---

### Q64: How do you evaluate chatbot quality?

**A)** Speed only  
**B)** User satisfaction, task completion, accuracy  
**C)** Cost only

**Answer: B**

Metrics: user satisfaction scores (thumbs up/down), task completion rate, response accuracy, escalation frequency, conversation length. A/B test improvements. Monitor for failure patterns.

---

## Search & Knowledge Retrieval (Q65-70)

### Q65: What is semantic search?

**A)** Keyword matching  
**B)** Understanding meaning and intent, not just exact words  
**C)** Random search

**Answer: B**

Semantic search: finds conceptually similar content using embeddings. "Car problems" matches "vehicle issues", "automobile trouble". Better than keyword-only (misses synonyms, context).

---

### Q66: What is the role of embeddings in search?

**A)** Encrypt data  
**B)** Convert text to vectors for similarity comparison  
**C)** Compress files

**Answer: B**

Embeddings map text to vector space where similar meanings cluster. Search: convert query to vector, find nearest document vectors (cosine similarity). Enables semantic search in RAG.

---

### Q67: What is Amazon Kendra?

**A)** Foundation model  
**B)** Intelligent enterprise search using ML  
**C)** Storage service

**Answer: B**

Kendra: ML-powered search for enterprise documents. Understands natural language questions, ranks results by relevance, extracts answers. Indexes documents across S3, SharePoint, databases. Pre-GenAI, enhanceable with RAG.

---

### Q68: How does RAG improve search applications?

**A)** Faster indexing  
**B)** Generates direct answers from retrieved documents  
**C)** Stores more data

**Answer: B**

RAG: retrieve relevant docs + generate natural language answer. Users get direct answers, not just document links. Cites sources for verification. Combines search + generation.

---

### Q69: What is re-ranking in search?

**A)** Initial retrieval  
**B)** Second-pass scoring to refine result relevance  
**C)** Sorting by date

**Answer: B**

Re-ranking: after initial retrieval (e.g., top 20 docs), use sophisticated model to score relevance deeply. Return best 3-5 to LLM. Improves precision vs single-pass retrieval.

---

### Q70: What is query expansion in semantic search?

**A)** Making queries longer  
**B)** Generating alternative query phrasings to improve recall  
**C)** Translating queries

**Answer: B**

Query expansion: LLM generates similar questions/phrasings for user query. Search with all variations, combine results. Improves recall (find more relevant docs). Example: "Python loops" → add "iteration", "for/while".

---

## Code & Developer Tools (Q71-75)

### Q71: What can Amazon CodeWhisperer do?

**A)** Only syntax highlighting  
**B)** Real-time code suggestions, completions, security scanning  
**C)** Version control

**Answer: B**

CodeWhisperer: AI pair programmer providing inline code suggestions, function generation from comments, security vulnerability detection, license compliance. Supports Python, Java, JavaScript, TypeScript, etc.

---

### Q72: What is code explanation using LLMs?

**A)** Running code  
**B)** Generating natural language descriptions of code logic  
**C)** Compiling code

**Answer: B**

Code explanation: LLM reads code, describes functionality in plain language. Helps understand unfamiliar code, documents legacy systems, onboards developers. Example: "This function validates email format using regex."

---

### Q73: What is test generation using GenAI?

**A)** Running tests  
**B)** Automatically creating unit tests from code  
**C)** Debugging

**Answer: B**

Test generation: LLM analyzes function, generates unit tests covering edge cases, happy paths, error conditions. Saves time, improves coverage. Developer reviews/refines generated tests.

---

### Q74: What is code refactoring assistance?

**A)** Deleting code  
**B)** Suggesting improvements for code quality, performance  
**C)** Commenting code

**Answer: B**

Refactoring assistance: LLM suggests cleaner, more efficient code. Improves readability, performance, maintainability. Examples: simplify logic, modernize syntax, apply design patterns. Developer validates changes.

---

### Q75: What is bug fixing with GenAI?

**A)** Automatic deployment  
**B)** Suggesting fixes for identified errors  
**C)** Preventing all bugs

**Answer: B**

Bug fixing: provide error message + code, LLM suggests potential fixes with explanations. Accelerates debugging. Examples: null pointer fixes, type errors, logic corrections. Developer tests solutions.

---

## Specialized Applications (Q76-80)

### Q76: What is AI-powered personalization?

**A)** Same experience for all users  
**B)** Tailoring content/recommendations to individual users  
**C)** User authentication

**Answer: B**

Personalization: LLMs generate customized content, recommendations, emails based on user context, preferences, behavior. Examples: personalized product descriptions, dynamic email content, adaptive learning paths.

---

### Q77: What is document extraction using GenAI?

**A)** Deleting documents  
**B)** Extracting structured data from unstructured documents  
**C)** Storing documents

**Answer: B**

Document extraction: parse contracts, invoices, forms to extract key fields (dates, amounts, names). Combines vision models (read PDFs/images) + LLMs (understand structure). AWS: Textract + Bedrock.

---

### Q78: What is knowledge base question answering?

**A)** General chatbot  
**B)** Answering questions from company-specific documents  
**C)** Web search

**Answer: B**

Knowledge base Q&A: RAG system over company documents (policies, procedures, manuals). Employees ask questions, get accurate answers with citations. Reduces support tickets, improves knowledge access.

---

### Q79: What is meeting summarization?

**A)** Scheduling meetings  
**B)** Generating concise summaries of meeting transcripts  
**C)** Video conferencing

**Answer: B**

Meeting summarization: transcribe meeting (Transcribe), summarize key points, action items, decisions (LLM). Saves time, ensures alignment. Example: "Team decided to launch feature in Q2, John owns implementation."

---

### Q80: What are AI agents in enterprise applications?

**A)** Human employees  
**B)** Autonomous AI systems completing business workflows  
**C)** Monitoring tools

**Answer: B**

AI agents: orchestrate multi-step business tasks. Examples: expense report processing (extract receipts, validate, submit), customer onboarding (collect info, verify, create account). Bedrock Agents framework.

---

## Enterprise & Industry Applications (Q81-88)

### Q81: What is AI-powered customer service routing?

**A)** Random assignment  
**B)** Using LLMs to classify and route customer inquiries to right team  
**C)** Manual sorting

**Answer: B**

LLM analyzes customer message, determines intent/urgency/complexity, routes to appropriate department/agent. Reduces wait times, improves resolution rates. Replaces rigid rule-based routing.

---

### Q82: What is automated report generation?

**A)** Manual reporting  
**B)** GenAI creates business reports from data/templates  
**C)** Report storage

**Answer: B**

LLM generates executive summaries, financial reports, performance analyses from structured data. Combines data retrieval with natural language generation. Saves hours of manual writing.

---

### Q83: What is contract analysis with GenAI?

**A)** Contract storage  
**B)** Extracting clauses, identifying risks, comparing terms  
**C)** Printing contracts

**Answer: B**

LLM reads contracts, extracts key terms (dates, obligations, penalties), flags risks, compares against standards. Legal teams use for due diligence, compliance. Textract + Bedrock combination.

---

### Q84: What is email response automation?

**A)** Auto-reply "out of office"  
**B)** GenAI drafts contextual responses to customer emails  
**C)** Email filtering

**Answer: B**

LLM reads customer email, understands context, drafts appropriate response. Human reviews before sending. Use cases: support, inquiries, FAQs. Maintains brand voice through prompt engineering.

---

### Q85: What is product description generation?

**A)** Copying manufacturer specs  
**B)** Creating unique, engaging product descriptions at scale  
**C)** Product images

**Answer: B**

LLM generates descriptions from product attributes (size, color, features). Optimizes for SEO, tone, length. E-commerce use: thousands of products need unique descriptions. Personalization possible.

---

### Q86: What is AI-powered data enrichment?

**A)** Data backup  
**B)** Enhancing records with additional generated information  
**C)** Data compression

**Answer: B**

LLM adds missing fields, standardizes formats, generates summaries, categorizes data. Example: incomplete customer profiles → generate industry, company size, likely needs from available info.

---

### Q87: What is voice-of-customer analysis?

**A)** Recording calls  
**B)** Analyzing feedback at scale for themes, sentiment, insights  
**C)** Customer surveys only

**Answer: B**

LLM processes reviews, support tickets, social media, surveys. Identifies patterns, pain points, feature requests. Summarizes thousands of inputs into actionable insights. Replaces manual reading.

---

### Q88: What is legal document drafting assistance?

**A)** Replacing lawyers  
**B)** Generating first drafts of contracts, agreements, clauses  
**C)** Document storage

**Answer: B**

LLM creates initial document drafts from templates and requirements. Lawyer reviews, edits, finalizes. Saves time on boilerplate. Not for final legal advice—human oversight required.

---

## Creative & Media Applications (Q89-90)

### Q89: What is AI-powered video script generation?

**A)** Video editing  
**B)** Creating video scripts from brief or topic  
**C)** Video hosting

**Answer: B**

LLM writes video scripts: scenes, dialogue, narration. Input: topic, duration, audience. Output: structured script. Use for marketing videos, training content, social media. Human edits for brand fit.

---

### Q90: What is style transfer in image generation?

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

2. **3.1 Design Considerations (Q11-20):**
   - **Latency Management:** Model size, context length, streaming
   - **Context Window:** Token limits, truncation strategies
   - **Cost Optimization:** Token minimization, caching, provisioned throughput
   - **Error Handling:** Retries, fallbacks, graceful degradation
   - **Security:** Input validation, guardrails, PII protection
   - **Observability:** Logging, monitoring, tracing
   - **Scalability:** Auto-scaling, queue-based processing

3. **3.2 Prompt Engineering (Q21-30):**
   - **Zero-shot:** Direct instruction without examples
   - **Few-shot:** 2-5 examples to guide behavior
   - **Chain-of-thought:** Step-by-step reasoning
   - **System vs User Prompts:** Role/rules vs specific requests
   - **Templates:** Reusable structures with parameters
   - **Negative Prompting:** Explicit constraints
   - **Iterative Refinement:** Test, measure, improve

4. **3.3 Training & Fine-Tuning (Q31-40):**
   - **Pre-training:** Initial large-scale training (expensive, done once)
   - **Fine-tuning:** Domain specialization with custom data
   - **Instruction Tuning:** Training on instruction-response pairs
   - **RLHF:** Human feedback for alignment
   - **Data Requirements:** 100s-1000s quality examples
   - **Continuous Fine-tuning:** Regular updates with new data
   - **Distillation:** Training smaller models from larger ones

5. **3.4 Model Evaluation (Q41-50):**
   - **Evaluation Dimensions:** Quality, safety, performance
   - **Automated Metrics:** BLEU (translation), ROUGE (summarization)
   - **Human Evaluation:** Rating quality, helpfulness, harm
   - **A/B Testing:** Comparing variants with real users
   - **Hallucination Detection:** Verifying factual accuracy
   - **Safety Testing:** Toxicity, bias, harmful content
   - **Benchmarks:** MMLU, HellaSwag, HumanEval

6. **Practical Applications (Q51-90):**
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
