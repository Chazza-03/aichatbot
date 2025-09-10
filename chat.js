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
  TEMPERATURE: 0.1
};

// Initialize OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Knowledge base storage
class KnowledgeBase {
  constructor() {
    this.items = [];
    this.magnitudes = [];
    this.isLoaded = false;
  }

  load(filePath = path.join(__dirname, 'knowledge_embeddings.json')) {
    try {
      const raw = fs.readFileSync(filePath, 'utf8');
      this.items = JSON.parse(raw);
      this.magnitudes = this.items.map(item => 
        item.embedding ? this.calculateMagnitude(item.embedding) : 0
      );
      this.isLoaded = true;
      console.log(`✓ Loaded ${this.items.length} KB items from ${filePath}`);
    } catch (err) {
      console.warn(`⚠ Could not load knowledge base: ${err.message}`);
      this.items = [];
      this.magnitudes = [];
      this.isLoaded = false;
    }
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

  findTopMatches(queryEmbedding, k = CONFIG.MAX_CONTEXT_ITEMS) {
    if (!this.isLoaded) return [];

    const scores = this.items.map((item, index) => ({
      item,
      score: item.embedding 
        ? this.cosineSimilarity(queryEmbedding, item.embedding, this.magnitudes[index])
        : -1
    }));

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .filter(match => match.score > CONFIG.SIMILARITY_THRESHOLD);
  }
}

// Pattern matching utilities
class PatternMatcher {
  static GREETING_PATTERNS = [
    /^(hi|hello|hey|howdy|greetings|good\s(morning|afternoon|evening)|yo|sup|what's up|hi there|hello there)\b/i,
    /^\b(hi|hello|hey)\s+(there|chatbot|bot|assistant)\b/i,
    /^thanks?(\s+(you|a lot|so much))?$/i,
    /^thank\s+(you|u)\b/i,
    /^(goodbye|bye|see ya|see you|farewell|cheers)\b/i,
    /^how\s+are\s+you(\s+doing)?\??$/i,
    /^what's\s+up\??$/i,
    /^how's\s+it\s+going\??$/i
  ];

  static CONTACT_PATTERNS = [
    /(who (to )?(contact|speak to|call|email|reach).*(about|for))/i,
    /(contact.*(details|info|information))/i,
    /(phone.*number)/i,
    /(email.*address)/i
  ];

  static SERVICE_KEYWORDS = {
    warehousing: ['warehous', 'storage', 'inventory', 'pick and pack', 'devanning', 'pallet', 'fifo'],
    shipping: ['shipping', 'ship', 'sea freight', 'air freight', 'global shipping', 'ocean freight'],
    customs: ['customs', 'brokerage', 'bonded', 'clearance', 'import', 'export'],
    freight: ['freight', 'road freight', 'transport', 'delivery', 'distribution', 'haulage', 'trucking'],
    accounts: ['account', 'billing', 'invoice', 'payment', 'credit', 'finance', 'statement']
  };

  static isGreeting(message) {
    const trimmed = message.trim().toLowerCase();
    return this.GREETING_PATTERNS.some(pattern => pattern.test(trimmed));
  }

  static isContactQuery(message) {
    return this.CONTACT_PATTERNS.some(pattern => pattern.test(message));
  }

  static detectService(message) {
    const lowerMessage = message.toLowerCase();
    
    for (const [service, keywords] of Object.entries(this.SERVICE_KEYWORDS)) {
      if (keywords.some(keyword => lowerMessage.includes(keyword))) {
        return service;
      }
    }
    
    return null;
  }
}

// Response generators
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

// Context builder with smart prioritization
class ContextBuilder {
  static PRIORITY_WEIGHTS = { high: 3, medium: 2, low: 1 };

  static build(matches, userQuestion) {
    if (!matches.length) return '';

    const seenIntents = new Set();
    const prioritized = this.prioritizeMatches(matches);
    let contextText = '';

    for (const match of prioritized) {
      const { item, score } = match;
      const metadata = item.metadata || {};
      
      // Avoid duplicate intents unless high confidence
      if (seenIntents.has(metadata.intent) && score < 0.7) continue;
      seenIntents.add(metadata.intent);

      contextText += this.formatMatch(item, metadata);
      
      // Early exit for high-confidence matches
      if (score > 0.8 && contextText.length > 500) break;
    }

    return contextText;
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

  static formatMatch(item, metadata) {
    const header = metadata.category 
      ? `${metadata.category}${metadata.sub_category ? ' / ' + metadata.sub_category : ''}`
      : '';
    const question = item.Q || item.question || '';
    const answer = item.A || item.answer || item.text || '';
    
    return `\n[${header}] Q: ${question}\nA: ${answer}\n`;
  }
}

// Contact information handler
class ContactHandler {
  static getServiceContact(service, matches) {
    // Look for service-specific routing
    const routingInfo = matches.find(m => 
      m.item.metadata?.intent === 'contact_routing' &&
      m.item.metadata?.service_mapping
    );

    if (routingInfo?.item.metadata.service_mapping[service]) {
      return routingInfo.item.metadata.service_mapping[service];
    }

    // Fallback to general contact info
    const contactInfo = matches.find(m => 
      m.item.metadata?.category === "contact_information"
    );

    return contactInfo 
      ? `please call ${contactInfo.item.answer || 'our main number'}`
      : "please contact us at +44 (0)121 765 4166";
  }

  static shouldIncludeContact(question, matches, answer) {
    if (PatternMatcher.isGreeting(question)) return false;
    
    return PatternMatcher.isContactQuery(question) ||
           matches.length === 0 ||
           matches[0]?.score < 0.3 ||
           /don'?t know|unsure|not sure|no information|sorry|apologize/i.test(answer);
  }

  static formatContactInfo(question, matches, answer) {
    const detectedService = PatternMatcher.detectService(question);
    const contactInstructions = this.getServiceContact(detectedService, matches);
    
    // Avoid duplicate contact info
    if (answer.includes('+44') || answer.includes('sales@jeavons.co.uk') || answer.includes(contactInstructions)) {
      return answer;
    }

    if (detectedService) {
      return `${answer} For ${detectedService} inquiries, ${contactInstructions}.`;
    } else {
      return `${answer} For more information, please ask more questions!.`;
    }
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

// Main chatbot class
class ChatBot {
  constructor() {
    this.knowledgeBase = new KnowledgeBase();
    this.cache = new QueryCache();
    this.systemPrompt = this.buildSystemPrompt();
    
    // Load KB at startup
    this.knowledgeBase.load();
  }

  buildSystemPrompt() {
    return `
You are a knowledgeable and concise customer support agent for Jeavons Eurotir Ltd.
Use the provided context to answer questions accurately.

GREETING HANDLING:
- If the user just greets you, respond politely but don't provide contact details
- Keep greeting responses brief and professional
- Redirect to business purposes after acknowledging greetings

STRICT CONTACT ROUTING RULES:
1. ONLY provide contact details when:
   - The user explicitly asks for contact information (using words like "contact", "phone", "email", "call", "speak to someone")
   - You cannot answer the question with the provided context
   - The user specifically asks "who can I contact about X" or "who to speak to about Y"

2. When providing contact details:
   - First answer their main question completely
   - THEN provide specific contact instructions using service mapping from context
   - Format: "For [specific service] inquiries, [specific contact instructions]"

3. DO NOT include contact details in general service descriptions or answers to factual questions.
4. NEVER provide contact details in response to simple greetings

5. If no specific routing exists, provide general contact info but be clear about it.

Do not hallucinate contact details or service information - use only what's in the context.
    `.trim();
  }

  async processQuery(question) {
    // Handle greetings
    if (PatternMatcher.isGreeting(question)) {
      return {
        answer: ResponseGenerator.generateGreeting(question),
        matches: [],
        context_used: false,
        detected_service: null,
        is_greeting: true
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

      // Find matches
      const matches = this.knowledgeBase.findTopMatches(queryEmbedding);

      // Build context
      let contextText = ContextBuilder.build(matches, question);

      // Add contact info for explicit contact queries
      if (PatternMatcher.isContactQuery(question)) {
        const contactItem = this.knowledgeBase.items.find(item =>
          item.metadata?.category === "contact_information"
        );
        if (contactItem) {
          contextText += ContextBuilder.formatMatch(contactItem, contactItem.metadata || {});
        }
      }

      // Generate response
      const chatResponse = await openai.chat.completions.create({
        model: CONFIG.CHAT_MODEL,
        messages: [
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: this.buildUserPrompt(contextText, question) }
        ],
        temperature: CONFIG.TEMPERATURE,
        max_tokens: CONFIG.MAX_TOKENS
      });

      let answer = chatResponse.choices?.[0]?.message?.content || 
        "I apologize, but I couldn't generate a response at this time.";

      // Add contact info if needed
      if (ContactHandler.shouldIncludeContact(question, matches, answer)) {
        answer = ContactHandler.formatContactInfo(question, matches, answer);
      }

      const response = {
        answer,
        matches: matches.map(m => ({
          score: m.score,
          Q: m.item.Q || m.item.question,
          A: m.item.A || m.item.answer,
          metadata: m.item.metadata
        })),
        context_used: contextText.length > 0,
        detected_service: PatternMatcher.isContactQuery(question) 
          ? PatternMatcher.detectService(question) 
          : null,
        is_greeting: false
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

  buildUserPrompt(contextText, question) {
    const context = contextText || 'No specific context found for this question.';
    return `Context:\n${context}\n\nUser question: ${question}\n\nPlease provide a helpful and accurate answer based on the available information.`;
  }

  getStats() {
    return {
      knowledgeBaseSize: this.knowledgeBase.items.length,
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
  const { question } = req.body || {};
  
  // Validation
  if (!question || typeof question !== 'string' || !question.trim()) {
    return res.status(400).json({ 
      error: 'Please provide a valid question in the POST body.' 
    });
  }

  try {
    const response = await chatBot.processQuery(question.trim());
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

