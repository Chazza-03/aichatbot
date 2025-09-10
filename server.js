const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Import your API routes directly from root
const chatRoutes = require('./chat');

// Use your API routes
app.use('/api/chat', chatRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', message: 'AI Backend is running' });
});

// Serve static files from root
app.use('/data', express.static(__dirname));

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
