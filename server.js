const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

app.set('trust proxy', true);

// Middleware
app.use(cors());
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
