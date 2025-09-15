// securityMiddleware.js
const rateLimit = require("express-rate-limit");

// 1. Rate limiting (30 requests/min per IP)
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // limit each IP
  message: { error: "Too many requests. Please slow down." },
  // Use a custom key generator to handle proxy headers securely
  keyGenerator: (req, res) => {
    // Get the first IP from the X-Forwarded-For header
    // This is the IP of the original client, even with trust proxy enabled
    // If the header doesn't exist, fall back to the direct request IP
    return req.headers['x-forwarded-for'] || req.ip;
  }
});

// 2. Input validation & sanitization
function sanitizeInput(req, res, next) {
  const { question } = req.body || {};

  if (!question || typeof question !== "string") {
    return res.status(400).json({ error: "Invalid input: 'question' must be a string." });
  }

  if (question.length > 1000) {
    return res.status(400).json({ error: "Your query is too long." });
  }

  // Strip control chars
  req.body.question = question.replace(/[\x00-\x1F\x7F]/g, "").trim();

  next();
}

// 3. Prompt injection / abuse detection
function injectionGuard(req, res, next) {
  const q = req.body.question.toLowerCase();

  const blockedPatterns = [
    /ignore (previous|all) instructions/i,
    /disregard context/i,
    /reveal.*(system|prompt|instructions)/i,
    /show.*hidden/i,
    /what.*is.*your.*system/i,
    /(api\s?key|password|secret)/i,
  ];

  if (blockedPatterns.some((pattern) => pattern.test(q))) {
    return res.status(400).json({
      error: "This type of request is not allowed.",
    });
  }

  next();
}

module.exports = { limiter, sanitizeInput, injectionGuard };
