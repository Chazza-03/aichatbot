const path = require('path');
const fs = require('fs');
const express = require('express');
const OpenAI = require('openai');

// Import security middleware
const { limiter, sanitizeInput, injectionGuard } = require('./securityMiddleware');

// Configuration with validation
const CONFIG = {
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
  CHAT_MODEL: process.env.CHAT_MODEL || 'gpt-4o-mini',
  SIMILARITY_THRESHOLD: Math.max(0.1, Math.min(1.0, parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.4)),
  MAX_CONTEXT_ITEMS: Math.max(1, Math.min(20, parseInt(process.env.MAX_CONTEXT_ITEMS) || 8)),
  CACHE_TTL: 5 * 60 * 1000,
  MAX_TOKENS: 800,
  TEMPERATURE: 0.1,
  KEYWORD_BOOST: 0.1,
  INTENT_BOOST: 0.15,
  CATEGORY_BOOST: 0.08,
  PROCEDURAL_BOOST: 0.2
};

// Initialize OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Enhanced Knowledge Base with procedural awareness
class KnowledgeBase {
  constructor() {
    this.items = [];
    this.magnitudes = [];
    this.keywordIndex = new Map();
    this.intentIndex = new Map();
    this.categoryIndex = new Map();
    this.subCategoryIndex = new Map();
    this.isLoaded = false;
  }

  load(filePath = path.join(__dirname, 'knowledge_embeddings.json')) {
    try {
      // Check if file exists first
      if (!fs.existsSync(filePath)) {
        console.warn(`⚠ Knowledge base file not found: ${filePath}`);
        this.items = [];
        this.magnitudes = [];
        this.isLoaded = false;
        return;
      }
      
      const raw = fs.readFileSync(filePath, 'utf8');
      this.items = JSON.parse(raw);
      this.magnitudes = this.items.map(item =>
        item.embedding ? this.calculateMagnitude(item.embedding) : 0
      );
      this.buildMetadataIndexes();
      this.isLoaded = true;
      console.log(`✓ Loaded ${this.items.length} KB items from ${filePath}`);
      console.log(`✓ Built indexes: ${this.keywordIndex.size} keywords, ${this.intentIndex.size} intents, ${this.categoryIndex.size} categories`);
    } catch (err) {
      console.warn(`⚠ Could not load knowledge base: ${err.message}`);
      this.items = [];
      this.magnitudes = [];
      this.isLoaded = false;
    }
  }

  buildMetadataIndexes() {
    this.keywordIndex.clear();
    this.intentIndex.clear();
    this.categoryIndex.clear();
    this.subCategoryIndex.clear();

    this.items.forEach((item, index) => {
      const metadata = item.metadata || {};
      const category = item.category || 'uncategorized';
      const subCategory = item.sub_category || 'general';

      // Build keyword index
      if (metadata.keywords && Array.isArray(metadata.keywords)) {
        metadata.keywords.forEach(keyword => {
          const normalizedKeyword = keyword.toLowerCase().trim();
          if (!this.keywordIndex.has(normalizedKeyword)) {
            this.keywordIndex.set(normalizedKeyword, []);
          }
          this.keywordIndex.get(normalizedKeyword).push(index);
        });
      }

      // Build intent index
      if (metadata.intent) {
        const intent = metadata.intent.toLowerCase().trim();
        if (!this.intentIndex.has(intent)) {
          this.intentIndex.set(intent, []);
        }
        this.intentIndex.get(intent).push(index);
      }

      // Build category indexes
      const normalizedCategory = category.toLowerCase().trim();
      if (!this.categoryIndex.has(normalizedCategory)) {
        this.categoryIndex.set(normalizedCategory, []);
      }
      this.categoryIndex.get(normalizedCategory).push(index);

      const normalizedSubCategory = subCategory.toLowerCase().trim();
      if (!this.subCategoryIndex.has(normalizedSubCategory)) {
        this.subCategoryIndex.set(normalizedSubCategory, []);
      }
      this.subCategoryIndex.get(normalizedSubCategory).push(index);
    });
  }

  calculateMagnitude(vector) {
    return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  }

  dotProduct(a, b) {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  cosineSimilarity(queryVector, itemVector, itemMagnitude) {
    if (!queryVector || !itemVector || queryVector.length !== itemVector.length) {
      return -1;
    }

    const queryMagnitude = this.calculateMagnitude(queryVector);
    const denominator = (queryMagnitude * itemMagnitude) || 1e-10;

    return this.dotProduct(queryVector, itemVector) / denominator;
  }

  // Enhanced matching with metadata and procedural awareness
  findTopMatches(queryEmbedding, query, k = CONFIG.MAX_CONTEXT_ITEMS) {
    if (!this.isLoaded) return [];

    const queryWords = this.extractQueryWords(query);
    const detectedIntent = this.detectIntent(query);
    const isProcedural = this.isProceduralQuery(query);

    const scores = this.items.map((item, index) => {
      let baseScore = item.embedding
        ? this.cosineSimilarity(queryEmbedding, item.embedding, this.magnitudes[index])
        : -1;

      if (baseScore <= 0) return { item, score: baseScore, boosts: {} };

      const boosts = this.calculateBoosts(item, queryWords, detectedIntent, index, isProcedural);
      const boostedScore = baseScore + boosts.total;

      return {
        item,
        score: Math.min(1.0, boostedScore),
        baseScore,
        boosts
      };
    });

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .filter(match => match.score > CONFIG.SIMILARITY_THRESHOLD);
  }

  extractQueryWords(query) {
    return query.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2);
  }

  detectIntent(query) {
    const lowerQuery = query.toLowerCase();

    const intentPatterns = {
      'procedural': /\b(how to|steps|process|procedure|guide|instructions|set up|arrange|organize)\b/i,
      'billing_info': /\b(bill|billing|invoice|cost|price|fee|charge|payment)\b/i,
      'contact_info': /\b(contact|phone|email|speak|call|reach|touch)\b/i,
      'service_info': /\b(service|what|how|do you|can you|offer)\b/i,
      'location_info': /\b(where|location|address|based|office)\b/i,
      'hours_info': /\b(hours|open|closed|time|when)\b/i,
      'shipping_info': /\b(ship|shipping|freight|delivery|transport)\b/i,
      'storage_info': /\b(storage|warehouse|store|keep)\b/i,
      'customs_info': /\b(customs|clearance|import|export|brokerage)\b/i
    };

    // Intent mapping to align with KB metadata intents
    const intentMapping = {
      'procedural': 'explain_process',
      'billing_info': 'payment_info',
      'contact_info': 'request_contact',
      'service_info': 'service_inquiry',
      'location_info': 'location_info',
      'hours_info': 'operating_hours',
      'shipping_info': 'shipping_inquiry',
      'storage_info': 'storage_inquiry',
      'customs_info': 'customs_inquiry'
    };

    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(lowerQuery)) {
        return intentMapping[intent] || intent;
      }
    }

    return null;
  }

  isProceduralQuery(query) {
    const lowerQuery = query.toLowerCase();
    return /\b(how to|steps|process|procedure|guide|instructions|set up|arrange|organize)\b/i.test(lowerQuery);
  }

  calculateBoosts(item, queryWords, detectedIntent, itemIndex, isProcedural) {
    const boosts = {
      keyword: 0,
      intent: 0,
      category: 0,
      procedural: 0,
      priority: 0,
      total: 0
    };

    const metadata = item.metadata || {};
    const category = item.category || '';
    const subCategory = item.sub_category || '';

    // Keyword matching boost
    if (metadata.keywords && Array.isArray(metadata.keywords)) {
      const matchingKeywords = metadata.keywords.filter(keyword =>
        queryWords.some(word =>
          word.includes(keyword.toLowerCase()) ||
          keyword.toLowerCase().includes(word)
        )
      );
      boosts.keyword = Math.min(0.3, matchingKeywords.length * CONFIG.KEYWORD_BOOST);
    }

    // Intent matching boost
    if (detectedIntent && metadata.intent === detectedIntent) {
      boosts.intent = CONFIG.INTENT_BOOST;
    }

    // Category matching boost (for procedural queries)
    if (isProcedural && (category.toLowerCase().includes('process') || 
                         category.toLowerCase().includes('procedure') ||
                         subCategory.toLowerCase().includes('process') ||
                         subCategory.toLowerCase().includes('procedure'))) {
      boosts.category = CONFIG.CATEGORY_BOOST;
    }

    // Procedural content boost
    if (isProcedural && this.isProceduralContent(item)) {
      boosts.procedural = CONFIG.PROCEDURAL_BOOST;
    }

    // Priority boost
    const priorityWeights = { 'high': 0.1, 'medium': 0.05, 'low': 0.02 };
    if (metadata.priority && priorityWeights[metadata.priority]) {
      boosts.priority = priorityWeights[metadata.priority];
    }

    boosts.total = boosts.keyword + boosts.intent + boosts.category + boosts.procedural + boosts.priority;
    return boosts;
  }

  isProceduralContent(item) {
    const answer = item.A || item.answer || '';
    return /\b(step|first|next|then|finally|process|procedure|guide|instructions)\b/i.test(answer);
  }

  // Enhanced related content finding
  getRelatedContent(mainMatchIndex, maxRelated = 4) {
    const mainItem = this.items[mainMatchIndex];
    if (!mainItem) return [];

    const relatedItems = new Set();
    const metadata = mainItem.metadata || {};

    // Get related questions from metadata
    if (metadata.related_questions && Array.isArray(metadata.related_questions)) {
      metadata.related_questions.slice(0, maxRelated).forEach(index => {
        if (index < this.items.length) {
          relatedItems.add(this.items[index]);
        }
      });
    }

    // Get items from same category
    const category = mainItem.category;
    if (category && this.categoryIndex.has(category.toLowerCase())) {
      const categoryItems = this.categoryIndex.get(category.toLowerCase());
      categoryItems.slice(0, 2).forEach(index => {
        if (index !== mainMatchIndex) {
          relatedItems.add(this.items[index]);
        }
      });
    }

    // Get items from same sub-category
    const subCategory = mainItem.sub_category;
    if (subCategory && this.subCategoryIndex.has(subCategory.toLowerCase())) {
      const subCategoryItems = this.subCategoryIndex.get(subCategory.toLowerCase());
      subCategoryItems.slice(0, 2).forEach(index => {
        if (index !== mainMatchIndex) {
          relatedItems.add(this.items[index]);
        }
      });
    }

    return Array.from(relatedItems).slice(0, maxRelated);
  }
}

// Enhanced PatternMatcher with procedural detection
class PatternMatcher {
  static GREETING_PATTERNS = [
    /^(hi|hello|hey|howdy|greetings|good\s(morning|afternoon|evening)|yo|sup|what's up|hi there|hello there)\b/i,
    /^\b(hi|hello|hey)\s+(there|chatbot|bot|assistant)\b/i,
    /^thanks?(\s+(you|a lot|so much))?$/i,
    /^thank\s+(you|u)\b/i,
    /^(goodbye|bye|see ya|see you|farewell|cheers)\b/i,
    /^how\s+are\s+you(\s+doing)?\??$/i,
    /^what's\s+up\??$/i,
    /^how's\it\s+going\??$/i
  ];

  static PROCEDURAL_PATTERNS = [
    /\b(how to|steps?|process|procedure|guide|instructions?|set up|arrange|organize|get started)\b/i,
    /\b(what are the steps|what is the process|how do I)\b/i,
    /\b(first|next|then|finally|after that)\b.*\?/i
  ];

  static isGreeting(message) {
    const trimmed = message.trim().toLowerCase();
    return this.GREETING_PATTERNS.some(pattern => pattern.test(trimmed));
  }

  static isProcedural(message) {
    return this.PROCEDURAL_PATTERNS.some(pattern => pattern.test(message));
  }

  static isContactQuery(message) {
    const contactPatterns = [
      /\b(contact|phone|email|call|reach|speak|talk)\b/i,
      /\b(how.*reach|how.*contact|get in touch)\b/i
    ];
    return contactPatterns.some(pattern => pattern.test(message));
  }
}

// Enhanced ContextBuilder for procedural questions
class ContextBuilder {
  static PRIORITY_WEIGHTS = { high: 3, medium: 2, low: 1 };

  static build(matches, userQuestion, knowledgeBase) {
    if (!matches.length) return { contextText: '', relatedContent: [] };

    const isProcedural = PatternMatcher.isProcedural(userQuestion);
    const prioritized = this.prioritizeMatches(matches, isProcedural);
    let contextText = '';
    let relatedContent = [];

    // For procedural questions, include more context and related content
    const maxItems = isProcedural ? Math.min(6, CONFIG.MAX_CONTEXT_ITEMS) : CONFIG.MAX_CONTEXT_ITEMS;

    for (let i = 0; i < Math.min(prioritized.length, maxItems); i++) {
      const match = prioritized[i];
      const { item, score } = match;
      const metadata = item.metadata || {};

      contextText += this.formatMatch(item, metadata, match.boosts);

      // Add related content for the first high-scoring match in procedural queries
      if (i === 0 && isProcedural && relatedContent.length === 0) {
        const itemIndex = knowledgeBase.items.findIndex(kbItem => kbItem === item);
        if (itemIndex !== -1) {
          relatedContent = knowledgeBase.getRelatedContent(itemIndex);
        }
      }

      if (score > 0.8 && contextText.length > 600) break;
    }

    // Add related content to context for procedural questions
    if (isProcedural && relatedContent.length > 0) {
      contextText += '\n\n## Related Information:\n';
      relatedContent.forEach((item, index) => {
        if (index < 3) { // Limit to 3 related items
          const metadata = item.metadata || {};
          contextText += this.formatMatch(item, metadata, {});
        }
      });
    }

    return { contextText, relatedContent };
  }

  static prioritizeMatches(matches, isProcedural) {
    return matches.sort((a, b) => {
      // For procedural queries, prioritize items with procedural content
      if (isProcedural) {
        const aIsProcedural = /\b(step|first|next|then|finally)\b/i.test(a.item.A || a.item.answer || '');
        const bIsProcedural = /\b(step|first|next|then|finally)\b/i.test(b.item.A || b.item.answer || '');
        
        if (aIsProcedural && !bIsProcedural) return -1;
        if (!aIsProcedural && bIsProcedural) return 1;
      }

      // Primary: by score
      if (b.score !== a.score) return b.score - a.score;

      // Secondary: by priority
      const aPriority = this.PRIORITY_WEIGHTS[a.item.metadata?.priority] || 1;
      const bPriority = this.PRIORITY_WEIGHTS[b.item.metadata?.priority] || 1;
      return bPriority - aPriority;
    });
  }

  static formatMatch(item, metadata, boosts = {}) {
    const intentTag = metadata.intent ? `[${metadata.intent.replace('_', ' ').toUpperCase()}]` : '';
    const priorityTag = metadata.priority ? `(${metadata.priority})` : '';
    const categoryTag = item.category ? `Category: ${item.category}` : '';
    const subCategoryTag = item.sub_category ? `Sub-category: ${item.sub_category}` : '';

    const question = item.Q || item.question || '';
    const answer = item.A || item.answer || item.text || '';

    let formatted = `\n${intentTag}${priorityTag}\nQ: ${question}\nA: ${answer}\n`;

    if (categoryTag) {
      formatted += `${categoryTag}`;
    }
    if (subCategoryTag) {
      formatted += ` | ${subCategoryTag}`;
    }

    if (metadata.context) {
      formatted += `\nContext: ${metadata.context}`;
    }

    formatted += '\n---\n';
    return formatted;
  }
}

// Enhanced system prompt for procedural questions
class SystemPromptBuilder {
  static buildSystemPrompt(context, isProcedural = false) {
    const basePrompt = `
# ROLE & PERSONA
You are an official AI assistant for Jeavons Eurotir Ltd., a family-owned logistics company. You speak on behalf of the company. You are helpful, professional, and proud of the company's 46 years of experience.

# CORE DIRECTIVES
1.  **FIRST PERSON:** Always refer to the company as "we", "us", or "our". NEVER use third-person like "Jeavons Eurotir offers..." or "They offer...". Example: "We offer global shipping services" NOT "Jeavons Eurotir offers global shipping."
2.  **STRICT CONTEXT USE:** Your knowledge is STRICTLY LIMITED to the context provided below. If the answer is not found in the context, you MUST say so. DO NOT HALLUCINATE or make up information.
3.  **NO KNOWLEDGE RESPONSE:** If you lack information, say: "I don't have that specific information on hand," or "I'm not sure about that detail," and guide them to contact the team.
4.  **FORMATTING:** Respond in clear, plain text. Use natural paragraphs. Do NOT use markdown, bullet points (*, -), or numbered lists.

# CONTEXT TO USE:
${context}

# FINAL INSTRUCTION
Answer the user's question based SOLELY on the context above. Speak as a representative of Jeavons Eurotir.
    `.trim();

    if (isProcedural) {
      return basePrompt + `

# SPECIAL INSTRUCTIONS FOR PROCEDURAL QUESTIONS:
- If the context contains multiple related procedures or steps, synthesize them into a coherent process
- Use transitional words like "First", "Next", "Then", "Finally" to create a clear flow
- If steps are mentioned across different context items, combine them logically
- Maintain the first-person perspective throughout the process description
      `.trim();
    }

    return basePrompt;
  }
}

// Enhanced main chatbot class
class ChatBot {
  constructor() {
    this.knowledgeBase = new KnowledgeBase();
    this.cache = new QueryCache();
    this.conversationHistory = new Map();
    this.knowledgeBase.load();
  }

  // NEW: Check if answer already has a closing phrase
  answerHasClosingPhrase(answerText) {
    const closingPhrasePatterns = [
      /feel free to (contact|reach out|ask)/i,
      /let me know if you need/i,
      /is there anything else.*help with/i,
      /please don't hesitate to contact/i,
      /we're here to help/i,
      /for more information.*contact/i,
      /reach out to.*team.*assist/i
    ];
    return closingPhrasePatterns.some(pattern => pattern.test(answerText));
  }

  // NEW: Add closing phrase to answers
  addClosingPhrase(answerText, hasContext) {
    const closingPhrases = [
      "Let me know if you need assistance with anything else."
    ];
    
    const selectedPhrase = closingPhrases[Math.floor(Math.random() * closingPhrases.length)];
    return `${answerText}\n\n${selectedPhrase}`;
  }

  async processQuery(question, sessionId = 'default') {
    if (PatternMatcher.isGreeting(question)) {
      return {
        answer: ResponseGenerator.generateGreeting(question),
        matches: [],
        context_used: false,
        detected_intent: null,
        is_greeting: true,
        related_content: []
      };
    }

    const cacheKey = question.toLowerCase().trim();
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    try {
      const embeddingResponse = await openai.embeddings.create({
        model: CONFIG.EMBEDDING_MODEL,
        input: question
      });
      const queryEmbedding = embeddingResponse.data[0].embedding;

      const matches = this.knowledgeBase.findTopMatches(queryEmbedding, question);
      const isProcedural = PatternMatcher.isProcedural(question);
      
      const { contextText, relatedContent } = ContextBuilder.build(matches, question, this.knowledgeBase);

      if (contextText.length === 0) {
        const noContextResponse = {
          answer: "I don't have specific information about that on hand. For detailed or specialized inquiries, please contact our team directly at +44 (0)121 765 4166. They'll be happy to assist you.",
          matches: [],
          context_used: false,
          detected_intent: this.knowledgeBase.detectIntent(question),
          is_greeting: false,
          related_content: []
        };
        return noContextResponse;
      }

      const systemPrompt = SystemPromptBuilder.buildSystemPrompt(contextText, isProcedural);
      
      const chatResponse = await openai.chat.completions.create({
        model: CONFIG.CHAT_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: question }
        ],
        temperature: CONFIG.TEMPERATURE,
        max_tokens: CONFIG.MAX_TOKENS
      });

      let answer = chatResponse.choices?.[0]?.message?.content ||
        "I apologize, but I couldn't generate a response at this time.";

      answer = this.cleanupFormattingArtifacts(answer);

      // NEW: Add closing phrase if appropriate
      if (!PatternMatcher.isGreeting(question) && !this.answerHasClosingPhrase(answer)) {
        answer = this.addClosingPhrase(answer, matches.length > 0);
      }

      const detectedIntent = this.knowledgeBase.detectIntent(question);

      const response = {
        answer,
        matches: matches.map(m => ({
          score: m.score,
          base_score: m.baseScore,
          boosts: m.boosts,
          Q: m.item.Q || m.item.question,
          A: m.item.A || m.item.answer,
          category: m.item.category,
          sub_category: m.item.sub_category,
          metadata: m.item.metadata
        })),
        context_used: contextText.length > 0,
        detected_intent: detectedIntent,
        is_greeting: false,
        is_procedural: isProcedural,
        related_content: relatedContent.map(item => ({
          Q: item.Q || item.question,
          A: item.A || item.answer,
          category: item.category,
          sub_category: item.sub_category
        }))
      };

      if (matches.length > 0) {
        this.cache.set(cacheKey, response);
      }

      return response;

    } catch (error) {
      console.error('Error processing query:', error);
      
      let errorMessage = "I'm experiencing technical difficulties. Please try again later.";
      
      // More specific error messages
      if (error.code === 'insufficient_quota') {
        errorMessage = "I'm temporarily unavailable due to service limits. Please contact us directly at +44 (0)121 765 4166.";
      } else if (error.code === 'rate_limit_exceeded') {
        errorMessage = "I'm receiving too many requests right now. Please try again in a moment.";
      }
      
      throw new Error(errorMessage);
    }
  }

  cleanupFormattingArtifacts(text) {
    return text
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/__([^_]+)__/g, '$1')
      .replace(/_([^_]+)_/g, '$1')
      .replace(/^#{1,6}\s+(.+)$/gm, '$1')
      .replace(/^\s*[-*•]\s+/gm, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  getStats() {
    return {
      knowledgeBaseSize: this.knowledgeBase.items.length,
      keywordIndexSize: this.knowledgeBase.keywordIndex.size,
      intentIndexSize: this.knowledgeBase.intentIndex.size,
      categoryIndexSize: this.knowledgeBase.categoryIndex.size,
      cacheSize: this.cache.size(),
      isKBLoaded: this.knowledgeBase.isLoaded
    };
  }
}

// Cache implementation
class QueryCache {
  constructor(ttl = CONFIG.CACHE_TTL) {
    this.cache = new Map();
    this.ttl = ttl;
    setInterval(() => this.cleanup(), 10 * 60 * 1000);
  }

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }
    return entry.data;
  }

  set(key, data) {
    this.cache.set(key, {
      timestamp: Date.now(),
      data
    });
  }

  cleanup() {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.ttl) {
        this.cache.delete(key);
      }
    }
  }

  size() {
    return this.cache.size;
  }
}

// ResponseGenerator
class ResponseGenerator {
  static GREETING_RESPONSES = {
    morning: (match) => `${match}! How can I help you with Jeavons Eurotir services today?`,
    thanks: () => "You're welcome! Is there anything else I can help you with?",
    goodbye: () => "Goodbye! Feel free to reach out if you have any more questions about our services.",
    howAreYou: () => "I'm doing well, thank you! I'm here to help you with any questions about Jeavons Eurotir's services. How can I assist you today?",
    default: [
      "Hello! How can I assist you with Jeavons Eurotir services today?",
      "Hi there! What can I help you with regarding our logistics services?",
      "Hello! I'm here to help with questions about warehousing, shipping, customs, and freight services. What do you need assistance with?",
      "Hi! How can I help you with Jeavons Eurotir today?"
    ]
  };

  static generateGreeting(message) {
    const lower = message.toLowerCase().trim();
    const timeMatch = lower.match(/good\s(morning|afternoon|evening)/i);
    if (timeMatch) return this.GREETING_RESPONSES.morning(timeMatch[0]);
    if (/^(thanks?|thank\s+(you|u))/i.test(lower)) return this.GREETING_RESPONSES.thanks();
    if (/^(goodbye|bye|see ya|see you|farewell)/i.test(lower)) return this.GREETING_RESPONSES.goodbye();
    if (/how\s+(are\s+you|are\s+things|is\s+it\s+going)/i.test(lower)) return this.GREETING_RESPONSES.howAreYou();
    
    const defaults = this.GREETING_RESPONSES.default;
    return defaults[Math.floor(Math.random() * defaults.length)];
  }
}

// ContactHandler
class ContactHandler {
  static shouldIncludeContact(question, matches, answer) {
    if (PatternMatcher.isGreeting(question)) return false;
    return PatternMatcher.isContactQuery(question) ||
      matches.length === 0 ||
      matches[0]?.score < 0.3 ||
      /don'?t know|unsure|not sure|no information|sorry|apologize/i.test(answer);
  }

  static formatContactInfo(question, matches, answer) {
    const contactMatch = matches.find(m =>
      m.item.metadata?.intent === 'contact_info' ||
      m.item.metadata?.keywords?.includes('contact') ||
      /contact|phone|email/i.test(m.item.Q || m.item.question || '')
    );

    if (contactMatch) {
      const contactAnswer = contactMatch.item.A || contactMatch.item.answer || '';
      if (!answer.includes(contactAnswer)) {
        return `${answer}\n\n${contactAnswer}`;
      }
    }

    if (!answer.includes('+44') && !answer.includes('sales@jeavonseurotir.co.uk')) {
      return `${answer} For more information, please contact us at +44 (0)121 765 4166.`;
    }

    return answer;
  }
}

// Express router setup
const router = express.Router();
const chatBot = new ChatBot();

router.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    ...chatBot.getStats(),
    timestamp: new Date().toISOString()
  });
});

router.post(
  '/',
  limiter,
  sanitizeInput,
  injectionGuard,
  async (req, res) => {
    const { question, sessionId } = req.body;

    try {
      const response = await chatBot.processQuery(question.trim(), sessionId || 'default');
      res.json(response);
    } catch (error) {
      console.error('Chat endpoint error:', error);
      res.status(500).json({
        error: 'Internal server error',
        answer: "I'm experiencing technical difficulties. Please try again later or contact us directly at +44 (0)121 765 4166."
      });
    }
  }
);

module.exports = router;

