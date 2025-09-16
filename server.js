const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Set trusted front-end origin from environment variables

const allowedOrigins = process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(',') : ['https://www.jeavonseurotir.co.uk','https://jeavonsdev.webchoice-test.co.uk'];

const corsOptions = {
  origin: (origin, callback) => {
    // Check if the incoming request origin is in the allowed list
    // Or if the origin is undefined 
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE'], // Specify allowed methods
  allowedHeaders: ['Content-Type', 'Authorization'], // Specify allowed headers
  optionsSuccessStatus: 200,
};

app.set('trust proxy', true);

// Use the cors middleware with your security-focused options
app.use(cors(corsOptions));
app.use(express.json());

// Import your API routes from root (not from api folder)
const chatRoutes = require('./chat');

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




