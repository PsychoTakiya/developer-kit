# Domain 3: Applications of Foundation Models

30 focused MCQs for AWS AI Practitioner exam preparation.

---

## Foundation Model Applications (Q1-8)

### Q1: What are the main application categories for foundation models?

**A)** Only text generation  
**B)** Text, images, code, and multimodal  
**C)** Only data analysis

**Answer: B**

Foundation models power: text generation (chatbots, content), image generation (art, product viz), code generation (development assist), multimodal (text+image, video). Diverse use cases across industries.

---

### Q2: What is a chatbot built on foundation models?

**A)** Rule-based Q&A system  
**B)** Conversational AI using LLMs for natural dialogue  
**C)** Pre-recorded responses

**Answer: B**

LLM-powered chatbots understand context, maintain conversation history, handle complex queries. Use models like Claude, GPT for customer service, support, virtual assistants. More flexible than rule-based.

---

### Q3: What is content generation using GenAI?

**A)** Copying existing content  
**B)** Creating original marketing copy, articles, summaries  
**C)** Data analysis only

**Answer: B**

Content generation: create marketing copy, blog posts, product descriptions, social media, email drafts. Automates repetitive writing tasks. Still needs human review for accuracy/brand voice.

---

### Q4: What is code generation with foundation models?

**A)** Compiling code  
**B)** Writing code from natural language descriptions  
**C)** Debugging only

**Answer: B**

Code generation: convert requirements to code, complete functions, generate tests, write documentation. Examples: CodeWhisperer, GitHub Copilot. Accelerates development, reduces boilerplate.

---

### Q5: What are multimodal foundation models?

**A)** Single data type only  
**B)** Handle multiple input/output types (text, images, audio)  
**C)** Multiple models combined

**Answer: B**

Multimodal models process and generate across data types: text-to-image (Stable Diffusion), image-to-text (vision models), audio-to-text (Transcribe). Single model, multiple modalities.

---

### Q6: What is document summarization?

**A)** Highlighting keywords  
**B)** Condensing long documents to key points  
**C)** Translating documents

**Answer: B**

Summarization: extract main ideas from long documents (reports, research, articles). Saves time, improves information access. Can specify length (executive summary vs detailed). Uses extractive or abstractive methods.

---

### Q7: What is sentiment analysis in GenAI?

**A)** Grammar checking  
**B)** Determining emotional tone (positive, negative, neutral)  
**C)** Translation

**Answer: B**

Sentiment analysis: classify text emotion/opinion. Applications: customer feedback, social media monitoring, product reviews. Foundation models understand nuance better than traditional ML.

---

### Q8: What is Amazon Bedrock Agents?

**A)** Human customer service  
**B)** Autonomous AI completing multi-step tasks  
**C)** Model training service

**Answer: B**

Bedrock Agents: orchestrate complex workflows, call APIs, access data sources, break down tasks. Example: "Book a flight" → search flights, check preferences, make reservation. Agentic AI.

---

## Conversational AI (Q9-14)

### Q9: What is the key advantage of LLM-based chatbots?

**A)** Faster than rule-based  
**B)** Understand context and handle ambiguity  
**C)** Never make mistakes

**Answer: B**

LLM chatbots understand natural language nuance, maintain context across turns, adapt to unexpected queries. Unlike rule-based (limited to predefined paths). Still need guardrails and monitoring.

---

### Q10: What is conversation memory in chatbots?

**A)** Database storage  
**B)** Maintaining context across message exchanges  
**C)** User authentication

**Answer: B**

Conversation memory: chatbot remembers previous messages in session. Enables coherent multi-turn conversations. Implementations: include full history in prompt (token limit aware) or summarize.

---

### Q11: How do you handle sensitive topics in chatbots?

**A)** Ignore them  
**B)** Use guardrails to detect and redirect  
**C)** Let model answer freely

**Answer: B**

Guardrails detect sensitive topics (medical advice, legal, financial). Respond with: "I can't provide medical advice. Please consult a professional." Prevents liability and harm.

---

### Q12: What is intent recognition in conversational AI?

**A)** User authentication  
**B)** Understanding what user wants to accomplish  
**C)** Language translation

**Answer: B**

Intent recognition: classify user goal (book appointment, check status, get refund). Routes to appropriate handler. LLMs excel at understanding intents from natural language variations.

---

### Q13: What is Amazon Lex used for?

**A)** Image generation  
**B)** Building voice and text chatbots  
**C)** Data storage

**Answer: B**

Lex: AWS service for chatbots with ASR (speech recognition) and NLU (language understanding). Powers Alexa. Integrates with Lambda for business logic. Pre-GenAI service, now enhanced with Bedrock.

---

### Q14: How do you evaluate chatbot quality?

**A)** Speed only  
**B)** User satisfaction, task completion, accuracy  
**C)** Cost only

**Answer: B**

Metrics: user satisfaction scores (thumbs up/down), task completion rate, response accuracy, escalation frequency, conversation length. A/B test improvements. Monitor for failure patterns.

---

## Search & Knowledge Retrieval (Q15-20)

### Q15: What is semantic search?

**A)** Keyword matching  
**B)** Understanding meaning and intent, not just exact words  
**C)** Random search

**Answer: B**

Semantic search: finds conceptually similar content using embeddings. "Car problems" matches "vehicle issues", "automobile trouble". Better than keyword-only (misses synonyms, context).

---

### Q16: What is the role of embeddings in search?

**A)** Encrypt data  
**B)** Convert text to vectors for similarity comparison  
**C)** Compress files

**Answer: B**

Embeddings map text to vector space where similar meanings cluster. Search: convert query to vector, find nearest document vectors (cosine similarity). Enables semantic search in RAG.

---

### Q17: What is Amazon Kendra?

**A)** Foundation model  
**B)** Intelligent enterprise search using ML  
**C)** Storage service

**Answer: B**

Kendra: ML-powered search for enterprise documents. Understands natural language questions, ranks results by relevance, extracts answers. Indexes documents across S3, SharePoint, databases. Pre-GenAI, enhanceable with RAG.

---

### Q18: How does RAG improve search applications?

**A)** Faster indexing  
**B)** Generates direct answers from retrieved documents  
**C)** Stores more data

**Answer: B**

RAG: retrieve relevant docs + generate natural language answer. Users get direct answers, not just document links. Cites sources for verification. Combines search + generation.

---

### Q19: What is re-ranking in search?

**A)** Initial retrieval  
**B)** Second-pass scoring to refine result relevance  
**C)** Sorting by date

**Answer: B**

Re-ranking: after initial retrieval (e.g., top 20 docs), use sophisticated model to score relevance deeply. Return best 3-5 to LLM. Improves precision vs single-pass retrieval.

---

### Q20: What is query expansion in semantic search?

**A)** Making queries longer  
**B)** Generating alternative query phrasings to improve recall  
**C)** Translating queries

**Answer: B**

Query expansion: LLM generates similar questions/phrasings for user query. Search with all variations, combine results. Improves recall (find more relevant docs). Example: "Python loops" → add "iteration", "for/while".

---

## Code & Developer Tools (Q21-25)

### Q21: What can Amazon CodeWhisperer do?

**A)** Only syntax highlighting  
**B)** Real-time code suggestions, completions, security scanning  
**C)** Version control

**Answer: B**

CodeWhisperer: AI pair programmer providing inline code suggestions, function generation from comments, security vulnerability detection, license compliance. Supports Python, Java, JavaScript, TypeScript, etc.

---

### Q22: What is code explanation using LLMs?

**A)** Running code  
**B)** Generating natural language descriptions of code logic  
**C)** Compiling code

**Answer: B**

Code explanation: LLM reads code, describes functionality in plain language. Helps understand unfamiliar code, documents legacy systems, onboards developers. Example: "This function validates email format using regex."

---

### Q23: What is test generation using GenAI?

**A)** Running tests  
**B)** Automatically creating unit tests from code  
**C)** Debugging

**Answer: B**

Test generation: LLM analyzes function, generates unit tests covering edge cases, happy paths, error conditions. Saves time, improves coverage. Developer reviews/refines generated tests.

---

### Q24: What is code refactoring assistance?

**A)** Deleting code  
**B)** Suggesting improvements for code quality, performance  
**C)** Commenting code

**Answer: B**

Refactoring assistance: LLM suggests cleaner, more efficient code. Improves readability, performance, maintainability. Examples: simplify logic, modernize syntax, apply design patterns. Developer validates changes.

---

### Q25: What is bug fixing with GenAI?

**A)** Automatic deployment  
**B)** Suggesting fixes for identified errors  
**C)** Preventing all bugs

**Answer: B**

Bug fixing: provide error message + code, LLM suggests potential fixes with explanations. Accelerates debugging. Examples: null pointer fixes, type errors, logic corrections. Developer tests solutions.

---

## Specialized Applications (Q26-30)

### Q26: What is AI-powered personalization?

**A)** Same experience for all users  
**B)** Tailoring content/recommendations to individual users  
**C)** User authentication

**Answer: B**

Personalization: LLMs generate customized content, recommendations, emails based on user context, preferences, behavior. Examples: personalized product descriptions, dynamic email content, adaptive learning paths.

---

### Q27: What is document extraction using GenAI?

**A)** Deleting documents  
**B)** Extracting structured data from unstructured documents  
**C)** Storing documents

**Answer: B**

Document extraction: parse contracts, invoices, forms to extract key fields (dates, amounts, names). Combines vision models (read PDFs/images) + LLMs (understand structure). AWS: Textract + Bedrock.

---

### Q28: What is knowledge base question answering?

**A)** General chatbot  
**B)** Answering questions from company-specific documents  
**C)** Web search

**Answer: B**

Knowledge base Q&A: RAG system over company documents (policies, procedures, manuals). Employees ask questions, get accurate answers with citations. Reduces support tickets, improves knowledge access.

---

### Q29: What is meeting summarization?

**A)** Scheduling meetings  
**B)** Generating concise summaries of meeting transcripts  
**C)** Video conferencing

**Answer: B**

Meeting summarization: transcribe meeting (Transcribe), summarize key points, action items, decisions (LLM). Saves time, ensures alignment. Example: "Team decided to launch feature in Q2, John owns implementation."

---

### Q30: What are AI agents in enterprise applications?

**A)** Human employees  
**B)** Autonomous AI systems completing business workflows  
**C)** Monitoring tools

**Answer: B**

AI agents: orchestrate multi-step business tasks. Examples: expense report processing (extract receipts, validate, submit), customer onboarding (collect info, verify, create account). Bedrock Agents framework.

---

## Enterprise & Industry Applications (Q31-38)

### Q31: What is AI-powered customer service routing?

**A)** Random assignment  
**B)** Using LLMs to classify and route customer inquiries to right team  
**C)** Manual sorting

**Answer: B**

LLM analyzes customer message, determines intent/urgency/complexity, routes to appropriate department/agent. Reduces wait times, improves resolution rates. Replaces rigid rule-based routing.

---

### Q32: What is automated report generation?

**A)** Manual reporting  
**B)** GenAI creates business reports from data/templates  
**C)** Report storage

**Answer: B**

LLM generates executive summaries, financial reports, performance analyses from structured data. Combines data retrieval with natural language generation. Saves hours of manual writing.

---

### Q33: What is contract analysis with GenAI?

**A)** Contract storage  
**B)** Extracting clauses, identifying risks, comparing terms  
**C)** Printing contracts

**Answer: B**

LLM reads contracts, extracts key terms (dates, obligations, penalties), flags risks, compares against standards. Legal teams use for due diligence, compliance. Textract + Bedrock combination.

---

### Q34: What is email response automation?

**A)** Auto-reply "out of office"  
**B)** GenAI drafts contextual responses to customer emails  
**C)** Email filtering

**Answer: B**

LLM reads customer email, understands context, drafts appropriate response. Human reviews before sending. Use cases: support, inquiries, FAQs. Maintains brand voice through prompt engineering.

---

### Q35: What is product description generation?

**A)** Copying manufacturer specs  
**B)** Creating unique, engaging product descriptions at scale  
**C)** Product images

**Answer: B**

LLM generates descriptions from product attributes (size, color, features). Optimizes for SEO, tone, length. E-commerce use: thousands of products need unique descriptions. Personalization possible.

---

### Q36: What is AI-powered data enrichment?

**A)** Data backup  
**B)** Enhancing records with additional generated information  
**C)** Data compression

**Answer: B**

LLM adds missing fields, standardizes formats, generates summaries, categorizes data. Example: incomplete customer profiles → generate industry, company size, likely needs from available info.

---

### Q37: What is voice-of-customer analysis?

**A)** Recording calls  
**B)** Analyzing feedback at scale for themes, sentiment, insights  
**C)** Customer surveys only

**Answer: B**

LLM processes reviews, support tickets, social media, surveys. Identifies patterns, pain points, feature requests. Summarizes thousands of inputs into actionable insights. Replaces manual reading.

---

### Q38: What is legal document drafting assistance?

**A)** Replacing lawyers  
**B)** Generating first drafts of contracts, agreements, clauses  
**C)** Document storage

**Answer: B**

LLM creates initial document drafts from templates and requirements. Lawyer reviews, edits, finalizes. Saves time on boilerplate. Not for final legal advice—human oversight required.

---

## Creative & Media Applications (Q39-44)

### Q39: What is AI-powered video script generation?

**A)** Video editing  
**B)** Creating video scripts from brief or topic  
**C)** Video hosting

**Answer: B**

LLM writes video scripts: scenes, dialogue, narration. Input: topic, duration, audience. Output: structured script. Use for marketing videos, training content, social media. Human edits for brand fit.

---

### Q40: What is style transfer in image generation?

**A)** Copying images  
**B)** Applying artistic style to images (e.g., "like Van Gogh")  
**C)** Image compression

**Answer: B**

Image models apply artistic styles to photos. Input: content image + style reference. Output: blended result. Stable Diffusion supports this. Use cases: creative design, product visualization.

---

### Q41: What is brand voice customization?

**A)** Logo design  
**B)** Training/prompting model to match company's writing style  
**C)** Voice recording

**Answer: B**

Fine-tune or prompt engineer model to match brand tone (professional, casual, technical). Provide examples of brand writing. Ensures consistent voice across generated content.

---

### Q42: What is caption/alt-text generation for images?

**A)** Image editing  
**B)** Automatically creating image descriptions  
**C)** Image storage

**Answer: B**

Multimodal models (vision + language) generate image descriptions. Accessibility use (screen readers), SEO (alt text), content management. Bedrock with vision models or Rekognition + LLM.

---

### Q43: What is podcast/video transcription + summarization?

**A)** Recording podcast  
**B)** Converting audio to text + generating summary/chapters  
**C)** Podcast hosting

**Answer: B**

Transcribe audio → text. LLM summarizes, extracts key points, generates chapter markers, creates show notes. Complete workflow: Transcribe + Bedrock.

---

### Q44: What is multilingual content generation?

**A)** Translation only  
**B)** Creating original content in multiple languages  
**C)** Language detection

**Answer: B**

LLMs generate content directly in target languages (not just translate). Maintains cultural nuance. Use for global marketing, documentation. Some models better at certain languages (check training data).

---

## Technical & Advanced (Q45-50)

### Q45: What is SQL query generation from natural language?

**A)** Manual query writing  
**B)** LLM converts English questions to SQL  
**C)** Database backup

**Answer: B**

Text-to-SQL: "Show sales by region last quarter" → `SELECT region, SUM(sales) FROM orders WHERE date...`. Democratizes data access. Amazon Q supports this for BI.

---

### Q46: What is log analysis with GenAI?

**A)** Log storage  
**B)** Analyzing system logs to identify issues, patterns, anomalies  
**C)** Log deletion

**Answer: B**

LLM processes application/system logs, identifies errors, patterns, root causes. Natural language queries: "Why did service crash at 3pm?". Accelerates troubleshooting.

---

### Q47: What is API documentation generation?

**A)** API testing  
**B)** Automatically creating documentation from code  
**C)** API deployment

**Answer: B**

LLM reads code, generates API docs: endpoint descriptions, parameters, examples, response formats. Keeps docs in sync with code. Improves developer experience.

---

### Q48: What is regulatory compliance checking?

**A)** Regulation writing  
**B)** Scanning documents for compliance with regulations  
**C)** Regulation storage

**Answer: B**

LLM analyzes policies, procedures, contracts against regulatory requirements (GDPR, HIPAA, SOC 2). Identifies gaps, suggests fixes. Speeds compliance audits.

---

### Q49: What is knowledge graph construction with GenAI?

**A)** Graph databases only  
**B)** Extracting entities and relationships from text to build graphs  
**C)** Data visualization

**Answer: B**

LLM extracts entities (people, companies, concepts) and relationships from documents. Builds knowledge graph for complex queries. Bedrock + Neptune (graph database).

---

### Q50: What is intelligent form filling?

**A)** Manual data entry  
**B)** Auto-populating forms from documents or conversations  
**C)** Form design

**Answer: B**

LLM extracts information from sources (emails, PDFs, conversations), maps to form fields. Reduces manual entry errors. Use cases: applications, onboarding, data migration.

---

## Exam Tips

**Key Concepts to Remember:**

1. **Application Categories:**

   - **Text:** Chatbots, content, summarization, sentiment
   - **Images:** Generation, editing, analysis
   - **Code:** Generation, explanation, testing, refactoring
   - **Multimodal:** Text+image, audio+text
   - **Search:** Semantic search, RAG-based Q&A

2. **Conversational AI:**

   - LLM chatbots vs rule-based
   - Context/memory management
   - Intent recognition
   - Guardrails for safety
   - Amazon Lex for voice/text bots

3. **Search & Retrieval:**

   - Semantic search with embeddings
   - RAG (retrieve + generate)
   - Amazon Kendra (enterprise search)
   - Re-ranking for precision
   - Query expansion for recall

4. **Code Applications:**

   - CodeWhisperer (suggestions, security)
   - Code explanation and documentation
   - Test generation
   - Bug fixing assistance
   - Refactoring recommendations

5. **Specialized Use Cases:**

   - Personalization (tailored content)
   - Document extraction (structured from unstructured)
   - Knowledge base Q&A (company-specific)
   - Meeting/document summarization
   - AI agents (autonomous workflows)

6. **AWS Services Map:**
   - **Bedrock:** Foundation models for all applications
   - **Bedrock Agents:** Multi-step task automation
   - **CodeWhisperer:** Code generation
   - **Kendra:** Enterprise search
   - **Lex:** Chatbot building
   - **Textract:** Document OCR

**Study Focus:**

- Match application to appropriate foundation model capability
- Understand RAG for knowledge-based applications
- Know when to use specialized services (Kendra, Lex) vs general Bedrock
- Recognize chatbot requirements (context, guardrails, evaluation)
- Identify code generation use cases and tools
