const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// --- SECURITY UPDATE: Configure CORS Middleware ---
// Define the list of allowed websites (origins) that can talk to this backend
const allowedOrigins = [
  'https://www.jeavonseurotir.co.uk/', // REPLACE WITH YOUR LIVE SITE URL
  'https://jeavonsdev.webchoice-test.co.uk/', // YOUR TEST SITE URL
  // For local development
];

const corsOptions = {
  origin: function (origin, callback) {
    // Allow requests with no origin (like from mobile apps, Postman, or curl)
    if (!origin) return callback(null, true);
    
    // Check if the incoming origin is in the allowed list
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  }
};

app.set('trust proxy', true);

// Middleware
app.use(cors(corsOptions)); // Use the secure CORS configuration
app.use(express.json());

// Import your API routes from root (not from api folder)
const chatRoutes = require('./chat'); // Changed from './api/chat'

// Use your API routes
app.use('/api/chat', chatRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', message: 'AI Backend is running' });
});

// Serve static files from root
app.use(express.static(__dirname));

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
