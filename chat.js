const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

// Config - overridable via environment vars
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
const CHAT_MODEL = process.env.CHAT_MODEL || 'gpt-4o-mini';
const SIMILARITY_THRESHOLD = parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.4;
const MAX_CONTEXT_ITEMS = parseInt(process.env.MAX_CONTEXT_ITEMS) || 6;

// Load OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Precompute magnitudes for faster cosine similarity
let KB = [];
let precomputedMagnitudes = [];

// Load KB into memory at startup
const KB_PATH = path.join(__dirname, 'knowledge_embeddings.json');
try {
  const raw = fs.readFileSync(KB_PATH, 'utf8');
  KB = JSON.parse(raw);
  console.log(`Loaded ${KB.length} KB items from ${KB_PATH}`);
  
  // Precompute magnitudes for all embeddings
  precomputedMagnitudes = KB.map(item => 
    item.embedding ? magnitude(item.embedding) : 0
  );
} catch (err) {
  console.warn('Could not load knowledge_embeddings.json at startup:', err.message);
  KB = [];
  precomputedMagnitudes = [];
}

// Similarity helpers (optimized)
function dot(a, b) {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function magnitude(a) {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

function cosineSimOptimized(a, b, bMagnitude) {
  if (!a || !b || a.length !== b.length) return -1;
  const aMag = magnitude(a);
  const denom = (aMag * bMagnitude) || 1e-10;
  return dot(a, b) / denom;
}

// More specific service detection function
function detectService(question) {
  const questionLower = question.toLowerCase();
  
  const serviceKeywords = {
    'warehousing': ['warehous', 'storage', 'inventory', 'pick and pack', 'devanning', 'pallet', 'fifo'],
    'shipping': ['shipping', 'ship', 'sea freight', 'air freight', 'global shipping', 'ocean freight'],
    'customs': ['customs', 'brokerage', 'bonded', 'clearance', 'import', 'export'],
    'freight': ['freight', 'road freight', 'transport', 'delivery', 'distribution', 'haulage', 'trucking'],
    'accounts': ['account', 'billing', 'invoice', 'payment', 'credit', 'finance', 'statement']
  };

  for (const [service, keywords] of Object.entries(serviceKeywords)) {
    if (keywords.some(keyword => questionLower.includes(keyword))) {
      return service;
    }
  }
  
  return null; // Return null instead of 'general'
}

// Enhanced contact mapping function
function getServiceContactInfo(service, matches) {
  // Look for service routing information in matches
  const routingInfo = matches.find(m => 
    m.item.metadata?.intent === 'contact_routing' &&
    m.item.metadata?.service_mapping
  );

  if (routingInfo && routingInfo.item.metadata.service_mapping[service]) {
    return routingInfo.item.metadata.service_mapping[service];
  }

  // Fallback: look for any contact info in matches
  const contactInfo = matches.find(m => 
    m.item.metadata?.category === "contact_information"
  );

  if (contactInfo) {
    return `please call ${contactInfo.item.answer || 'our main number'}`;
  }

  return "please contact us at +44 (0)121 765 4166";
}

// Utility: get best matches (top K) - optimized
function getTopKMatches(queryEmbedding, k = MAX_CONTEXT_ITEMS) {
  const queryMag = magnitude(queryEmbedding);
  const scored = KB.map((item, index) => {
    if (!item.embedding) return { item, score: -1 };
    
    const score = cosineSimOptimized(queryEmbedding, item.embedding, precomputedMagnitudes[index]);
    return { item, score };
  });
  
  // Quick partial sort for top K instead of full sort
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).filter(x => x.score > SIMILARITY_THRESHOLD);
}

// Cache for common queries
const queryCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Smart context builder with prioritization
function buildContextText(matches, userQuestion) {
  if (matches.length === 0) return '';

  let contextText = '';
  const seenIntents = new Set();
  
  // Prioritize matches by score and metadata priority
  const prioritizedMatches = matches
    .sort((a, b) => {
      // First by score
      if (b.score !== a.score) return b.score - a.score;
      
      // Then by priority (high > medium > low)
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const aPriority = priorityOrder[a.item.metadata?.priority || 'medium'] || 1;
      const bPriority = priorityOrder[b.item.metadata?.priority || 'medium'] || 1;
      return bPriority - aPriority;
    });

  for (const match of prioritizedMatches) {
    const { item, score } = match;
    const md = item.metadata || {};
    
    // Avoid duplicate intents unless they provide complementary info
    if (seenIntents.has(md.intent) && score < 0.7) continue;
    seenIntents.add(md.intent);

    const header = md.category ? `${md.category}${md.sub_category ? ' / ' + md.sub_category : ''}` : '';
    const q = item.Q || item.question || '';
    const a = item.A || item.answer || item.text || '';
    
    contextText += `\n[${header}] Q: ${q}\nA: ${a}\n`;
    
    // Early exit if we have high-confidence matches
    if (score > 0.8 && contextText.length > 500) break;
  }

  return contextText;
}

// Enhanced system prompt with stricter contact routing
const SYSTEM_PROMPT = `
You are a knowledgeable and concise customer support agent for Jeavons Eurotir Ltd.
Use the provided context to answer questions accurately.

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

4. If no specific routing exists, provide general contact info but be clear about it.

Do not hallucinate contact details or service information - use only what's in the context.
`;

// Express router
const express = require('express');
const router = express.Router();

// API endpoint
router.post('/', async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const { question } = req.body || {};
  if (!question || typeof question !== 'string') {
    res.status(400).json({ error: 'Please provide a question in the POST body.' });
    return;
  }

  // Check cache first
  const cacheKey = question.toLowerCase().trim();
  const cached = queryCache.get(cacheKey);
  if (cached && (Date.now() - cached.timestamp) < CACHE_TTL) {
    return res.status(200).json(cached.response);
  }

  try {
    // 1) Create embedding for the question
    const embResp = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: question
    });
    const qEmb = embResp.data[0].embedding;

    // 2) Find top matches
    const topMatches = getTopKMatches(qEmb);

    // 3) Build context text with smart prioritization
    let contextText = buildContextText(topMatches, question);

    // 4) Check if this is explicitly a contact question
    const isExplicitContactQuestion = /(who (to )?(contact|speak to|call|email|reach).*(about|for))|(contact.*(details|info|information))|(phone.*number)|(email.*address)/i.test(question);
    
    // Add contact info to context only if explicitly requested
    if (isExplicitContactQuestion) {
      const contactInfo = KB.find(item =>
        item.metadata?.category === "contact_information"
      );
      if (contactInfo) {
        contextText += `\n[contact_information] Q: ${contactInfo.Q || contactInfo.question}\nA: ${contactInfo.A || contactInfo.answer}\n`;
      }
    }

    // 5) Construct chat messages
    const userPrompt = `Context:\n${contextText || 'No specific context found for this question.'}\n\nUser question: ${question}\n\nPlease provide a helpful and accurate answer based on the available information.`;

    const chatResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.1, // Lower temperature for more factual responses
      max_tokens: 600
    });

    let answer = chatResp.choices?.[0]?.message?.content || "I apologize, but I couldn't generate a response at this time.";

    // 6) Only add contact info for explicit contact questions or low-confidence answers
    const shouldAddContactInfo = 
      isExplicitContactQuestion || 
      topMatches.length === 0 || 
      topMatches[0].score < 0.3 || // Lower threshold for fallback
      /don'?t know|unsure|not sure|no information|sorry|apologize/i.test(answer);
    
    if (shouldAddContactInfo) {
      const detectedService = detectService(question);
      const contactInstructions = getServiceContactInfo(detectedService, topMatches);
      
      // Only add contact info if not already included
      if (!answer.includes('+44') && !answer.includes('info@jeavons.co.uk') && !answer.includes(contactInstructions)) {
        if (detectedService) {
          answer += ` For ${detectedService} inquiries, ${contactInstructions}.`;
        } else {
          answer += ` For more information, please contact us at +44 (0)121 765 4166.`;
        }
      }
    }

    const response = {
      answer,
      matches: topMatches.map(m => ({
        score: m.score,
        Q: m.item.Q || m.item.question,
        A: m.item.A || m.item.answer,
        metadata: m.item.metadata
      })),
      context_used: contextText.length > 0,
      detected_service: isExplicitContactQuestion ? detectService(question) : null
    };

    // Cache successful responses
    if (topMatches.length > 0) {
      queryCache.set(cacheKey, {
        timestamp: Date.now(),
        response
      });
    }

    res.status(200).json(response);

  } catch (err) {
    console.error('Error /api/chat:', err);
    res.status(500).json({ 
      error: 'Internal server error', 
      answer: "I'm experiencing technical difficulties. Please try again later or contact us directly at +44 (0)121 765 4166."
    });
  }
});

// Export the router
module.exports = router;

