const fs = require('fs');
const path = require('path');
const express = require('express');
const OpenAI = require('openai');

// Configuration with validation
const CONFIG = {
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
  CHAT_MODEL: process.env.CHAT_MODEL || 'gpt-4o-mini',
  SIMILARITY_THRESHOLD: Math.max(0.1, Math.min(1.0, parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.4)),
  MAX_CONTEXT_ITEMS: Math.max(1, Math.min(20, parseInt(process.env.MAX_CONTEXT_ITEMS) || 6)),
  CACHE_TTL: 5 * 60 * 1000, // 5 minutes
  MAX_TOKENS: 600,
  TEMPERATURE: 0.1, // Low temperature for factual accuracy
  KEYWORD_BOOST: 0.1,
  INTENT_BOOST: 0.15
};

// Initialize OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Knowledge base storage
class KnowledgeBase {
  constructor() {
    this.items = [];
    this.magnitudes = [];
    this.keywordIndex = new Map();
    this.intentIndex = new Map();
    this.isLoaded = false;
  }

  load(filePath = path.join(__dirname, 'knowledge_embeddings.json')) {
    try {
      const raw = fs.readFileSync(filePath, 'utf8');
      this.items = JSON.parse(raw);
      this.magnitudes = this.items.map(item =>
        item.embedding ? this.calculateMagnitude(item.embedding) : 0
      );
      this.buildMetadataIndexes();
      this.isLoaded = true;
      console.log(`✓ Loaded ${this.items.length} KB items from ${filePath}`);
      console.log(`✓ Built indexes: ${this.keywordIndex.size} keywords, ${this.intentIndex.size} intents`);
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

    this.items.forEach((item, index) => {
      const metadata = item.metadata;
      if (!metadata) return;

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

  // Enhanced matching with metadata boosting
  findTopMatches(queryEmbedding, query, k = CONFIG.MAX_CONTEXT_ITEMS) {
    if (!this.isLoaded) return [];

    const queryWords = this.extractQueryWords(query);
    const detectedIntent = this.detectIntent(query);

    const scores = this.items.map((item, index) => {
      let baseScore = item.embedding
        ? this.cosineSimilarity(queryEmbedding, item.embedding, this.magnitudes[index])
        : -1;

      if (baseScore <= 0) return { item, score: baseScore, boosts: {} };

      const boosts = this.calculateBoosts(item, queryWords, detectedIntent, index);
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

    // Intent detection patterns
    const intentPatterns = {
      'billing_info': /\b(bill|billing|invoice|cost|price|fee|charge|payment)\b/i,
      'contact_info': /\b(contact|phone|email|speak|call|reach|touch)\b/i,
      'service_info': /\b(service|what|how|do you|can you|offer)\b/i,
      'location_info': /\b(where|location|address|based|office)\b/i,
      'hours_info': /\b(hours|open|closed|time|when)\b/i,
      'shipping_info': /\b(ship|shipping|freight|delivery|transport)\b/i,
      'storage_info': /\b(storage|warehouse|store|keep)\b/i,
      'customs_info': /\b(customs|clearance|import|export|brokerage)\b/i
    };

    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(lowerQuery)) {
        return intent;
      }
    }

    return null;
  }

  calculateBoosts(item, queryWords, detectedIntent, itemIndex) {
    const boosts = {
      keyword: 0,
      intent: 0,
      priority: 0,
      total: 0
    };

    const metadata = item.metadata;
    if (!metadata) return boosts;

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

    // Priority boost
    const priorityWeights = { 'high': 0.1, 'medium': 0.05, 'low': 0.02 };
    if (metadata.priority && priorityWeights[metadata.priority]) {
      boosts.priority = priorityWeights[metadata.priority];
    }

    boosts.total = boosts.keyword + boosts.intent + boosts.priority;
    return boosts;
  }

  // Get related questions based on metadata
  getRelatedQuestions(itemIndex, maxRelated = 3) {
    const item = this.items[itemIndex];
    if (!item?.metadata?.related_questions) return [];

    return item.metadata.related_questions
      .slice(0, maxRelated)
      .map(relatedIndex => {
        if (relatedIndex < this.items.length) {
          const relatedItem = this.items[relatedIndex];
          return {
            question: relatedItem.Q || relatedItem.question,
            answer: relatedItem.A || relatedItem.answer
          };
        }
        return null;
      })
      .filter(Boolean);
  }
}

// Enhanced pattern matching utilities
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

  static CONTACT_PATTERNS = [
    /(who (to )?(contact|speak to|call|email|reach).*(about|for))/i,
    /(contact.*(details|info|information))/i,
    /(phone.*number)/i,
    /(email.*address)/i,
    /(who.*contact)/i,
    /(speak.*to)/i,
    /(get.*in.*touch)/i
  ];

  static isGreeting(message) {
    const trimmed = message.trim().toLowerCase();
    return this.GREETING_PATTERNS.some(pattern => pattern.test(trimmed));
  }

  static isContactQuery(message) {
    return this.CONTACT_PATTERNS.some(pattern => pattern.test(message));
  }
}

// Enhanced response generators
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

    // Time-specific greetings
    const timeMatch = lower.match(/good\s(morning|afternoon|evening)/i);
    if (timeMatch) return this.GREETING_RESPONSES.morning(timeMatch[0]);

    // Thank you responses
    if (/^(thanks?|thank\s+(you|u))/i.test(lower)) {
      return this.GREETING_RESPONSES.thanks();
    }

    // Goodbye responses
    if (/^(goodbye|bye|see ya|see you|farewell)/i.test(lower)) {
      return this.GREETING_RESPONSES.goodbye();
    }

    // How are you responses
    if (/how\s+(are\s+you|are\s+things|is\s+it\s+going)/i.test(lower)) {
      return this.GREETING_RESPONSES.howAreYou();
    }

    // Random default greeting
    const defaults = this.GREETING_RESPONSES.default;
    return defaults[Math.floor(Math.random() * defaults.length)];
  }
}

// Enhanced context builder with metadata awareness
class ContextBuilder {
  static PRIORITY_WEIGHTS = { high: 3, medium: 2, low: 1 };

  static build(matches, userQuestion, knowledgeBase) {
    if (!matches.length) return { contextText: '', relatedQuestions: [] };

    const seenIntents = new Set();
    const prioritized = this.prioritizeMatches(matches);
    let contextText = '';
    let relatedQuestions = [];

    for (let i = 0; i < prioritized.length; i++) {
      const match = prioritized[i];
      const { item, score } = match;
      const metadata = item.metadata || {};

      // Avoid duplicate intents unless high confidence
      if (seenIntents.has(metadata.intent) && score < 0.7) continue;
      seenIntents.add(metadata.intent);

      contextText += this.formatMatch(item, metadata, match.boosts);

      // Add related questions from the first high-scoring match
      if (i === 0 && relatedQuestions.length === 0) {
        const itemIndex = knowledgeBase.items.findIndex(kbItem => kbItem === item);
        if (itemIndex !== -1) {
          relatedQuestions = knowledgeBase.getRelatedQuestions(itemIndex);
        }
      }

      // Early exit for high-confidence matches
      if (score > 0.8 && contextText.length > 500) break;
    }

    return { contextText, relatedQuestions };
  }

  static prioritizeMatches(matches) {
    return matches.sort((a, b) => {
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
    const keywordTags = metadata.keywords ? `Keywords: ${metadata.keywords.join(', ')}` : '';

    const question = item.Q || item.question || '';
    const answer = item.A || item.answer || item.text || '';

    let formatted = `\n${intentTag}${priorityTag} Q: ${question}\nA: ${answer}`;

    if (keywordTags) {
      formatted += `\n${keywordTags}`;
    }

    if (metadata.context) {
      formatted += `\nContext: ${metadata.context}`;
    }

    formatted += '\n';
    return formatted;
  }
}

// Enhanced contact handler
class ContactHandler {
  static shouldIncludeContact(question, matches, answer) {
    if (PatternMatcher.isGreeting(question)) return false;

    return PatternMatcher.isContactQuery(question) ||
      matches.length === 0 ||
      matches[0]?.score < 0.3 ||
      /don'?t know|unsure|not sure|no information|sorry|apologize/i.test(answer);
  }

  static formatContactInfo(question, matches, answer) {
    // Look for contact info in matches
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

    // Fallback to default contact
    if (!answer.includes('+44') && !answer.includes('sales@jeavonseurotir.co.uk')) {
      return `${answer} For more information, please contact us at +44 (0)121 765 4166.`;
    }

    return answer;
  }
}

// Cache with TTL
class QueryCache {
  constructor(ttl = CONFIG.CACHE_TTL) {
    this.cache = new Map();
    this.ttl = ttl;

    // Cleanup expired entries every 10 minutes
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

// Enhanced main chatbot class
class ChatBot {
  constructor() {
    this.knowledgeBase = new KnowledgeBase();
    this.cache = new QueryCache();
    this.systemPrompt = this.buildSystemPrompt(); // *** UPDATED PROMPT
    this.conversationHistory = new Map();

    // Load KB at startup
    this.knowledgeBase.load();
  }

  buildSystemPrompt() {
    return `
# ROLE & PERSONA
You are an official AI assistant for Jeavons Eurotir Ltd., a family-owned logistics company. You speak on behalf of the company. You are helpful, professional, and proud of the company's 46 years of experience.

# CORE DIRECTIVES
1.  **FIRST PERSON:** Always refer to the company as "we", "us", or "our". NEVER use third-person like "Jeavons Eurotir offers..." or "They offer...". Example: "We offer global shipping services" NOT "Jeavons Eurotir offers global shipping."
2.  **STRICT CONTEXT USE:** Your knowledge is STRICTLY LIMITED to the context provided below. If the answer is not found in the context, you MUST say so. DO NOT HALLUCINATE or make up information.
3.  **NO KNOWLEDGE RESPONSE:** If you lack information, say: "I don't have that specific information on hand," or "I'm not sure about that detail," and guide them to contact the team.
4.  **FORMATTING:** Respond in clear, plain text. Use natural paragraphs. Do NOT use markdown, bullet points (*, -), or numbered lists.

# CONTEXT TO USE:
{context}

# FINAL INSTRUCTION
Answer the user's question based SOLELY on the context above. Speak as a representative of Jeavons Eurotir.
    `.trim();
  }

  async processQuery(question, sessionId = 'default') {
    // Handle greetings
    if (PatternMatcher.isGreeting(question)) {
      return {
        answer: ResponseGenerator.generateGreeting(question),
        matches: [],
        context_used: false,
        detected_intent: null,
        is_greeting: true,
        related_questions: []
      };
    }

    // Check cache
    const cacheKey = question.toLowerCase().trim();
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    try {
      // Generate embedding
      const embeddingResponse = await openai.embeddings.create({
        model: CONFIG.EMBEDDING_MODEL,
        input: question
      });
      const queryEmbedding = embeddingResponse.data[0].embedding;

      // Find matches with enhanced metadata-aware scoring
      const matches = this.knowledgeBase.findTopMatches(queryEmbedding, question);

      // Build enhanced context
      const { contextText, relatedQuestions } = ContextBuilder.build(matches, question, this.knowledgeBase);

      // *** CRITICAL: If no context is found, build a specific response to avoid hallucination.
      if (contextText.length === 0) {
        const noContextResponse = {
          answer: "I don't have specific information about that on hand. For detailed or specialized inquiries, please contact our team directly at +44 (0)121 765 4166. They'll be happy to assist you.",
          matches: [],
          context_used: false,
          detected_intent: this.knowledgeBase.detectIntent(question),
          is_greeting: false,
          related_questions: []
        };
        return noContextResponse;
      }

      // Generate response
      const chatResponse = await openai.chat.completions.create({
        model: CONFIG.CHAT_MODEL,
        messages: [
          { role: 'system', content: this.systemPrompt.replace('{context}', contextText) }, // *** Inject context into prompt
          { role: 'user', content: question } // *** Simplified user prompt
        ],
        temperature: CONFIG.TEMPERATURE,
        max_tokens: CONFIG.MAX_TOKENS
      });

      let answer = chatResponse.choices?.[0]?.message?.content ||
        "I apologize, but I couldn't generate a response at this time.";

      // Clean the answer
      answer = this.cleanupFormattingArtifacts(answer);

      // Enhanced contact handling
      if (ContactHandler.shouldIncludeContact(question, matches, answer)) {
        answer = ContactHandler.formatContactInfo(question, matches, answer);
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
          metadata: m.item.metadata
        })),
        context_used: contextText.length > 0,
        detected_intent: detectedIntent,
        is_greeting: false,
        related_questions: relatedQuestions
      };

      // Cache successful responses
      if (matches.length > 0) {
        this.cache.set(cacheKey, response);
      }

      return response;

    } catch (error) {
      console.error('Error processing query:', error);
      throw new Error('Internal server error');
    }
  }

  // *** REMOVED the old buildUserPrompt function as the prompt is now handled in the system message.

  // Clean up any markdown formatting artifacts
  cleanupFormattingArtifacts(text) {
    return text
      // Remove markdown bold/italic
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/__([^_]+)__/g, '$1')
      .replace(/_([^_]+)_/g, '$1')
      // Remove markdown headers
      .replace(/^#{1,6}\s+(.+)$/gm, '$1')
      // Replace bullet points with natural language
      .replace(/^\s*[-*•]\s+/gm, '') // Remove bullet characters
      // Clean up extra whitespace
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  getStats() {
    return {
      knowledgeBaseSize: this.knowledgeBase.items.length,
      keywordIndexSize: this.knowledgeBase.keywordIndex.size,
      intentIndexSize: this.knowledgeBase.intentIndex.size,
      cacheSize: this.cache.size(),
      isKBLoaded: this.knowledgeBase.isLoaded
    };
  }
}

// Express router setup
const router = express.Router();
const chatBot = new ChatBot();

// Health check endpoint
router.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    ...chatBot.getStats(),
    timestamp: new Date().toISOString()
  });
});

// Main chat endpoint
router.post('/', async (req, res) => {
  const { question, sessionId } = req.body || {};

  // Validation
  if (!question || typeof question !== 'string' || !question.trim()) {
    return res.status(400).json({
      error: 'Please provide a valid question in the POST body.'
    });
  }

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
});

module.exports = router;

