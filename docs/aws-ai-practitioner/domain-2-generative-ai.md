# Domain 2: Fundamentals of Generative AI

30 focused MCQs for AWS AI Practitioner exam preparation.

---

## Generative AI Basics (Q1-10)

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

## Prompt Engineering (Q11-18)

### Q11: What is prompt engineering?

**A)** Training new models  
**B)** Crafting inputs to guide model outputs  
**C)** Hardware optimization

**Answer: B**

Prompt engineering: designing effective instructions for GenAI models. Same model, different prompts = different quality results. No code changes needed, just better questions/instructions.

---

### Q12: What is zero-shot prompting?

**A)** Asking without examples  
**B)** Always providing examples  
**C)** Using empty prompts

**Answer: A**

Zero-shot: direct task request without examples ("Summarize this article"). Works for common tasks models saw during training. Simplest approach, use first.

---

### Q13: What is few-shot prompting?

**A)** Short prompts only  
**B)** Providing examples before the task  
**C)** Using multiple models

**Answer: B**

Few-shot: include 2-5 examples showing desired format/pattern before actual task. Model learns from examples. Better consistency and handles unusual formats. More tokens used.

---

### Q14: What is chain-of-thought prompting?

**A)** Multiple separate prompts  
**B)** Asking model to show reasoning steps  
**C)** Connecting models together

**Answer: B**

Chain-of-thought: request step-by-step reasoning ("Let's think step by step"). Improves accuracy on complex reasoning, math, analysis. Makes model's logic transparent.

---

### Q15: What is the purpose of temperature in generation?

**A)** Hardware cooling  
**B)** Controls randomness/creativity of outputs  
**C)** Training speed

**Answer: B**

Temperature (0-1): Low (0.1-0.3) = deterministic, consistent, factual. High (0.7-0.9) = creative, diverse, varied. Use low for facts, high for creative writing.

---

### Q16: What should a well-structured prompt include?

**A)** Only the question  
**B)** Role, task, context, format, constraints  
**C)** Just keywords

**Answer: B**

Effective prompts specify: role (persona), clear task, relevant context, desired format, constraints (length, tone, what to avoid). Explicit instructions = better results.

---

### Q17: What is prompt chaining?

**A)** Long single prompt  
**B)** Breaking complex tasks into sequential prompts  
**C)** Repeating same prompt

**Answer: B**

Prompt chaining: multi-step process where each prompt builds on previous output. Example: extract facts → summarize → generate questions. Higher quality for complex tasks.

---

### Q18: How do you reduce hallucinations in responses?

**A)** Use higher temperature  
**B)** Provide context, use low temperature, request citations  
**C)** Use longer prompts

**Answer: B**

Reduce hallucinations: provide relevant context (RAG), lower temperature (0.1-0.3), explicit instructions ("use only provided information"), request citations, validate facts.

---

## RAG & Advanced Techniques (Q19-26)

### Q19: What is RAG (Retrieval-Augmented Generation)?

**A)** A model type  
**B)** Retrieving relevant docs before generating answers  
**C)** Hardware acceleration

**Answer: B**

RAG: retrieve relevant information from knowledge base, include in prompt, generate grounded answer. Reduces hallucinations, enables current/private data, no model retraining needed.

---

### Q20: What are embeddings in RAG?

**A)** Encrypted data  
**B)** Vector representations capturing semantic meaning  
**C)** Model parameters

**Answer: B**

Embeddings: convert text to high-dimensional vectors (e.g., 1536 dimensions). Similar meaning = similar vectors. Enable semantic search for RAG retrieval.

---

### Q21: Which AWS service offers managed RAG?

**A)** Amazon S3  
**B)** Amazon Bedrock Knowledge Bases  
**C)** Amazon EC2

**Answer: B**

Bedrock Knowledge Bases: fully managed RAG. Automatic chunking, embedding, vector storage (OpenSearch), retrieval, generation. No infrastructure management required.

---

### Q22: What is chunking in RAG?

**A)** Compressing models  
**B)** Splitting documents into smaller segments  
**C)** Batching requests

**Answer: B**

Chunking: divide documents into 500-1000 token segments. Enables focused retrieval, fits in context windows. Overlapping chunks (10-20%) maintain context across boundaries.

---

### Q23: What is a vector database used for?

**A)** Storing images  
**B)** Efficient similarity search on embeddings  
**C)** Caching responses

**Answer: B**

Vector databases store and search embeddings efficiently. Find nearest neighbors (most similar text). AWS options: OpenSearch (k-NN), Aurora (pgvector), MemoryDB (vector search).

---

### Q24: When should you use RAG vs fine-tuning?

**A)** Always use RAG  
**B)** RAG for facts, fine-tuning for style  
**C)** Always use fine-tuning

**Answer: B**

RAG: current/changing information, factual Q&A, private data, cheaper. Fine-tuning: specialized style/tone, static domain knowledge, format requirements. Combine both for style + facts.

---

### Q25: What is hybrid search in RAG?

**A)** Using multiple models  
**B)** Combining semantic (vector) and keyword search  
**C)** Cloud and on-premise

**Answer: B**

Hybrid search: blend vector similarity (semantic) + keyword matching (BM25). Better accuracy: semantic understands meaning, keywords catch exact terms/codes. Typical: 70% semantic, 30% keyword.

---

### Q26: How does RAG reduce hallucinations?

**A)** Trains model longer  
**B)** Grounds answers in retrieved source documents  
**C)** Uses smaller models

**Answer: B**

RAG provides factual context from knowledge base. Model answers from provided documents, not just memorized training data. Can cite sources, admit when info unavailable.

---

## Production & AWS Services (Q27-30)

### Q27: What are Amazon Bedrock Guardrails?

**A)** Physical security  
**B)** Content filters and safety controls for model outputs  
**C)** Network firewalls

**Answer: B**

Guardrails filter harmful content (hate, violence, sexual), block denied topics (medical advice), redact PII, enforce custom word filters. Multi-layer safety for production GenAI.

---

### Q28: What is Amazon Q?

**A)** Queue service  
**B)** AI assistant for AWS and business tasks  
**C)** Quantum computing

**Answer: B**

Amazon Q: AI assistant for AWS (troubleshooting, code generation), business intelligence queries, document analysis. Understands AWS context, provides actionable answers.

---

### Q29: How do you optimize GenAI costs?

**A)** Always use largest model  
**B)** Choose appropriate model size, cache responses, optimize tokens  
**C)** Use highest temperature

**Answer: B**

Cost optimization: use smallest model meeting requirements (Lite for simple tasks), cache common queries, compress prompts, set max_tokens, batch processing for non-urgent.

---

### Q30: What is provisioned throughput in Bedrock?

**A)** Free tier  
**B)** Reserved model capacity with guaranteed availability  
**C)** Open source access

**Answer: B**

Provisioned throughput: reserve model capacity for predictable latency/availability. Cost-effective at high volume. Alternative to on-demand (pay-per-token). Commit for 1-6 months.

---

## Fine-Tuning & Customization (Q31-38)

### Q31: What is fine-tuning a foundation model?

**A)** Adjusting temperature  
**B)** Training model on domain-specific data to specialize it  
**C)** Prompt engineering

**Answer: B**

Fine-tuning: continue training pre-trained model on your data. Teaches specialized vocabulary, style, domain knowledge. Requires labeled data, compute. Bedrock supports fine-tuning for some models.

---

### Q32: What is instruction tuning?

**A)** Writing better prompts  
**B)** Training models to follow natural language instructions  
**C)** User guides

**Answer: B**

Instruction tuning: train model on instruction-response pairs. Makes models better at following directions. Foundation models like Claude, GPT use instruction tuning. Improves task generalization.

---

### Q33: What is model context window?

**A)** Training dataset size  
**B)** Maximum input/output tokens model can process  
**C)** Memory usage

**Answer: B**

Context window: token limit for combined input + output. Claude: 200K, GPT-4: 8-32K, Titan: 8K. Exceeding limit requires chunking or summarization. Larger = more context but slower/costlier.

---

### Q34: What is top-p (nucleus sampling)?

**A)** Selecting top model  
**B)** Sampling from smallest set of tokens whose cumulative probability exceeds p  
**C)** Precision metric

**Answer: B**

Top-p (0-1): sample from tokens totaling probability p. Lower = more focused/deterministic. Higher = more diverse. Alternative to temperature. Combine for control: low temp + low top-p = consistent.

---

### Q35: What is model quantization?

**A)** Adding more parameters  
**B)** Reducing model size by using lower precision  
**C)** Training longer

**Answer: B**

Quantization: convert 32-bit weights to 8-bit or lower. Smaller model size, faster inference, lower memory. Some accuracy trade-off. Used for edge deployment.

---

### Q36: What is Amazon Bedrock Model Evaluation?

**A)** User reviews  
**B)** Automated testing of models with benchmarks and custom prompts  
**C)** Hardware testing

**Answer: B**

Model Evaluation: compare foundation models on standard benchmarks (reasoning, knowledge) or your custom prompts. Helps select best model for use case. Human or automated evaluation.

---

### Q37: What is prompt template?

**A)** Pre-built website  
**B)** Reusable prompt structure with variables  
**C)** Model architecture

**Answer: B**

Prompt template: standardized prompt with placeholders (e.g., "Summarize {document} for {audience}"). Ensures consistency, enables testing, supports A/B testing. Bedrock supports saved templates.

---

### Q38: What is semantic similarity in embeddings?

**A)** Text length matching  
**B)** How close vectors are in embedding space (cosine similarity)  
**C)** Character matching

**Answer: B**

Semantic similarity: measure vector distance (cosine similarity: -1 to 1). High score (> 0.8) = similar meaning. Used in RAG retrieval, duplicate detection, recommendation.

---

## Production & Enterprise (Q39-45)

### Q39: What is model latency?

**A)** Training time  
**B)** Time from request to response  
**C)** Data lag

**Answer: B**

Latency: request-to-response time (milliseconds). Affected by: model size, prompt length, output length, concurrent requests. Optimize: smaller models, provisioned throughput, streaming.

---

### Q40: What is token-based pricing?

**A)** Fixed monthly cost  
**B)** Pay per 1000 input/output tokens  
**C)** Free unlimited

**Answer: B**

Bedrock charges per 1000 tokens (input + output separately). Input cheaper than output. Track usage to control costs. Example: Claude Haiku $0.00025 in, $0.00125 out per 1K tokens.

---

### Q41: What is streaming response?

**A)** Video generation  
**B)** Sending generated tokens progressively as they're created  
**C)** Batch processing

**Answer: B**

Streaming: return partial responses as generated (word-by-word or sentence-by-sentence). Better UX (immediate feedback), can cancel early. Supported by Bedrock API. Use for chatbots.

---

### Q42: What is Amazon Bedrock Playground?

**A)** Game development  
**B)** Interactive UI for testing models and prompts  
**C)** Sandbox environment

**Answer: B**

Playground: web UI for experimenting with foundation models. Test prompts, compare models, adjust parameters (temperature, top-p). No code needed. Iterate before API integration.

---

### Q43: What is inference profile in Bedrock?

**A)** User profile  
**B)** Configuration for cross-region routing and failover  
**C)** Hardware specs

**Answer: B**

Inference profiles: route requests across regions for resilience, load balancing. Automatic failover if region unavailable. Improves availability and latency for global applications.

---

### Q44: What is model invocation logging?

**A)** Training logs  
**B)** Recording requests/responses for audit, debugging, analysis  
**C)** Error logs only

**Answer: B**

Invocation logging: capture prompts, responses, metadata to CloudWatch or S3. Use for: debugging, compliance audits, quality analysis, usage tracking. PII considerations for storage.

---

### Q45: What is Amazon Bedrock Studio?

**A)** Video editing  
**B)** Development environment for building GenAI applications  
**C)** Recording studio

**Answer: B**

Bedrock Studio: IDE for building, testing, deploying GenAI apps. Includes RAG setup, agent creation, collaboration tools. Simplified workflow for developers.

---

## Advanced RAG & Agents (Q46-50)

### Q46: What is agent reasoning in Bedrock Agents?

**A)** Human thinking  
**B)** LLM decides which tools/APIs to call based on user request  
**C)** Rule engine

**Answer: B**

Agent reasoning: LLM analyzes user goal, breaks into steps, selects appropriate tools/APIs, executes plan. ReAct pattern (Reasoning + Acting). Handles multi-step complex tasks.

---

### Q47: What is metadata filtering in RAG?

**A)** Removing PII  
**B)** Narrowing search by document attributes (date, author, category)  
**C)** Content moderation

**Answer: B**

Metadata filtering: combine semantic search with filters (e.g., "documents from 2024 by Finance department"). Improves precision, reduces irrelevant results. Store metadata with embeddings.

---

### Q48: What is retrieval confidence scoring?

**A)** Model accuracy  
**B)** How relevant retrieved documents are to query  
**C)** User rating

**Answer: B**

Confidence scoring: measure document relevance (similarity score 0-1). Set threshold (e.g., > 0.7) to filter low-quality matches. Prevents poor context degrading LLM answers.

---

### Q49: What is Amazon Bedrock Agents action group?

**A)** User permissions  
**B)** Set of API operations agent can perform  
**C)** Model ensemble

**Answer: B**

Action group: define APIs/functions agent can call (e.g., database queries, order creation, email sending). Provide OpenAPI specs. Agent decides when/how to use.

---

### Q50: What is synthetic data generation with GenAI?

**A)** Real data collection  
**B)** Using LLMs to create training data  
**C)** Data deletion

**Answer: B**

Synthetic data: LLM generates training examples (questions, answers, variations). Useful when real data scarce, expensive, or privacy-sensitive. Augments real data, tests edge cases.

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
