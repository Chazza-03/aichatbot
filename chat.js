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

// Enhanced system prompt with better service mapping
const SYSTEM_PROMPT = `
You are a knowledgeable and concise customer support agent for Jeavons Eurotir Ltd.
Use the provided context to answer questions accurately.

SPECIAL INSTRUCTIONS:
1. For service-specific inquiries (warehousing, shipping, customs, etc.), map to the correct contact from the context
2. If information is incomplete, acknowledge what you can answer and suggest contacting for specifics
3. For historical questions, be precise with dates and facts
4. Always maintain a professional and helpful tone
5. If unsure, provide general contact information but be transparent about limitations

Do not hallucinate or invent information beyond what's in the context.
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

    // Add fallback contact info if no good matches or for specific intents
    if (topMatches.length === 0 || /contact|phone|email|reach/i.test(question)) {
      const contactInfo = KB.find(item =>
        item.metadata?.category === "contact_information" &&
        item.metadata?.sub_category === "phone_menu"
      );
      if (contactInfo) {
        contextText += `\n[contact_information / phone_menu] Q: ${contactInfo.Q}\nA: ${contactInfo.A}\n`;
      }
    }

    // 4) Construct chat messages
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

    // Smart fallback handling
    const shouldAddFallback = 
      topMatches.length === 0 || 
      topMatches[0].score < 0.5 ||
      /don'?t know|unsure|not sure|no information|sorry/i.test(answer);
    
    if (shouldAddFallback) {
      answer += ` For more specific information, please contact us at +44 (0)121 789 8666 or email info@jeavons.co.uk.`;
    }

    const response = {
      answer,
      matches: topMatches.map(m => ({
        score: m.score,
        Q: m.item.Q,
        A: m.item.A,
        metadata: m.item.metadata
      })),
      context_used: contextText.length > 0
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
      answer: "I'm experiencing technical difficulties. Please try again later or contact us directly at +44 (0)121 789 8666."
    });
  }
});

// Export the router
module.exports = router;
