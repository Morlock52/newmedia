const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const OpenAI = require('openai');
const fs = require('fs').promises;
const path = require('path');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3001;

// Initialize OpenAI
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Middleware
app.use(cors());
app.use(express.json());

// Rate limiting configuration
const limiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 10, // 10 requests per minute
    message: 'Too many requests from this IP, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
});

// Apply rate limiting to chat endpoint
app.use('/api/chat', limiter);

// Cache for system prompt
let systemPromptCache = null;
let promptLastModified = null;

// Load system prompt with caching
async function loadSystemPrompt() {
    const promptPath = path.join(__dirname, 'prompts', 'media-assistant.md');
    
    try {
        const stats = await fs.stat(promptPath);
        
        // Check if we need to reload the prompt
        if (!systemPromptCache || !promptLastModified || stats.mtime > promptLastModified) {
            systemPromptCache = await fs.readFile(promptPath, 'utf-8');
            promptLastModified = stats.mtime;
            console.log('System prompt loaded/reloaded');
        }
        
        return systemPromptCache;
    } catch (error) {
        console.error('Error loading system prompt:', error);
        // Fallback prompt if file doesn't exist
        return `You are a helpful Media Server Assistant focused exclusively on topics related to media servers, streaming, and content management. 

Your expertise includes:
- Media server software (Jellyfin, Plex, Emby)
- Media management tools (Sonarr, Radarr, Lidarr, Bazarr)
- Troubleshooting common issues
- Recommending content based on preferences
- Helping with configuration and setup
- Explaining error messages and solutions

You should:
- Only discuss media server related topics
- Be helpful and friendly
- Provide clear, actionable advice
- Suggest solutions for common problems
- Stay within your domain of expertise

If asked about unrelated topics, politely redirect the conversation back to media server assistance.`;
    }
}

// Helper function to format messages for OpenAI
function formatMessages(history, userMessage, systemPrompt) {
    const messages = [
        { role: 'system', content: systemPrompt }
    ];
    
    // Add conversation history (limit to last 10 messages for context)
    if (history && history.length > 0) {
        const recentHistory = history.slice(-10);
        recentHistory.forEach(msg => {
            messages.push({
                role: msg.role,
                content: msg.content
            });
        });
    }
    
    // Add current user message
    messages.push({
        role: 'user',
        content: userMessage
    });
    
    return messages;
}

// Helper function to adjust response based on settings
function getModelParameters(settings) {
    const speed = settings?.speed || 0.5;
    const detail = settings?.detail || 0.7;
    
    return {
        model: 'gpt-4-turbo-preview',
        temperature: 0.7 + (speed * 0.3), // Higher speed = higher temperature
        max_tokens: Math.floor(500 + (detail * 1500)), // Higher detail = more tokens
        presence_penalty: 0.1,
        frequency_penalty: 0.1
    };
}

// Chat endpoint
app.post('/api/chat', async (req, res) => {
    const { message, history, settings } = req.body;
    
    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }
    
    try {
        // Load system prompt
        const systemPrompt = await loadSystemPrompt();
        
        // Format messages for OpenAI
        const messages = formatMessages(history, message, systemPrompt);
        
        // Get model parameters based on settings
        const modelParams = getModelParameters(settings);
        
        // Make request to OpenAI
        const completion = await openai.chat.completions.create({
            ...modelParams,
            messages: messages,
            stream: false
        });
        
        const assistantResponse = completion.choices[0].message.content;
        
        // Log token usage for monitoring
        console.log(`Tokens used: ${completion.usage.total_tokens}`);
        
        res.json({
            response: assistantResponse,
            usage: {
                prompt_tokens: completion.usage.prompt_tokens,
                completion_tokens: completion.usage.completion_tokens,
                total_tokens: completion.usage.total_tokens
            }
        });
        
    } catch (error) {
        console.error('OpenAI API error:', error);
        
        // Handle specific OpenAI errors
        if (error.status === 429) {
            res.status(429).json({ 
                error: 'Rate limit exceeded. Please try again later.' 
            });
        } else if (error.status === 401) {
            res.status(500).json({ 
                error: 'Authentication error. Please check API key configuration.' 
            });
        } else {
            res.status(500).json({ 
                error: 'An error occurred while processing your request.' 
            });
        }
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'healthy',
        timestamp: new Date().toISOString(),
        openai_configured: !!process.env.OPENAI_API_KEY
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ 
        error: 'An unexpected error occurred',
        message: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Media Assistant API server running on port ${PORT}`);
    console.log(`OpenAI API Key configured: ${!!process.env.OPENAI_API_KEY}`);
    
    // Check for API key
    if (!process.env.OPENAI_API_KEY) {
        console.warn('WARNING: OPENAI_API_KEY environment variable is not set!');
        console.warn('Set it using: export OPENAI_API_KEY=your-api-key-here');
    }
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server');
    app.close(() => {
        console.log('HTTP server closed');
    });
});

module.exports = app;