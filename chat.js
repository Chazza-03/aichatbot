const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

// Config - overridable via environment vars
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
const CHAT_MODEL = process.env.CHAT_MODEL || 'gpt-4o-mini';

// Load OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Load KB into memory at startup - UPDATED PATH for root directory
const KB_PATH = path.join(__dirname, 'knowledge_embeddings.json');
let KB = [];
try {
  const raw = fs.readFileSync(KB_PATH, 'utf8');
  KB = JSON.parse(raw);
  console.log(`Loaded ${KB.length} KB items from ${KB_PATH}`);
} catch (err) {
  console.warn('Could not load knowledge_embeddings.json at startup:', err.message);
  KB = [];
}

// Similarity helpers
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
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return -1;
  const denom = (magnitude(a) * magnitude(b)) || 1e-10;
  return dot(a, b) / denom;
}

// Utility: get best matches (top K)
function getTopKMatches(queryEmbedding, k = 5) {
  const scored = KB.map(item => {
    return {
      item,
      score: item.embedding ? cosineSim(queryEmbedding, item.embedding) : -1
    };
  });
  scored.sort((a,b) => b.score - a.score);
  return scored.slice(0, k);
}

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

  try {
    // 1) Create embedding for the question (server-side)
    const embResp = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: question
    });
    const qEmb = embResp.data[0].embedding;

    // 2) Find top matches
    const topMatches = getTopKMatches(qEmb, 6).filter(x => x.score > 0.4);

    // 3) Build context text
    let contextText = '';
    if (topMatches.length === 0) {
      contextText = ''; // no relevant context found
    } else {
      // Include metadata and short Q/A
      for (const m of topMatches) {
        const md = m.item.metadata || {};
        const header = (md.category ? `${md.category}${md.sub_category ? ' / ' + md.sub_category : ''}` : '');
        const q = m.item.Q || m.item.question || '';
        const a = m.item.A || m.item.answer || m.item.text || '';
        contextText += `\n[${header}] Q: ${q}\nA: ${a}\n`;
      }
    }

    // 4) Construct chat messages and call chat model
    const systemPrompt = `You are a concise, factual customer support agent for Jeavons Eurotir Ltd. Use only the provided context where possible. If information is missing, say you don't have that information and offer next steps (contact support). Do not hallucinate dates or facts.`;

    const userPrompt = `Context:\n${contextText || '[no context found]'}\n\nUser question: ${question}\n\nAnswer the question based on the context. Keep the answer concise and helpful.`;

    const chatResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.12,
      max_tokens: 800
    });

    const answer = chatResp.choices?.[0]?.message?.content || "Sorry, I couldn't generate a response.";

    res.status(200).json({
      answer,
      matches: topMatches.map(m => ({ score: m.score, Q: m.item.Q, A: m.item.A, metadata: m.item.metadata }))
    });

  } catch (err) {
    console.error('Error /api/chat:', err);
    res.status(500).json({ error: 'Internal server error', details: String(err) });
  }
});

// Export the router
module.exports = router;
