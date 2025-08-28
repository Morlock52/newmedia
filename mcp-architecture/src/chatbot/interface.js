/**
 * AI Chatbot Interface
 * Integrates with MCP servers and agent voting system
 * 
 * Features:
 * - OpenAI o1-mini powered conversations
 * - MCP server tool integration
 * - Agent voting for complex decisions
 * - Social media research capabilities
 * - Session management and context
 */

const OpenAI = require('openai');
const winston = require('winston');
const { v4: uuidv4 } = require('uuid');

class ChatbotInterface {
  constructor(options = {}) {
    this.openai = new OpenAI({
      apiKey: options.openaiApiKey
    });
    
    this.model = process.env.OPENAI_MODEL || options.model || 'o1-mini';
    this.agents = options.agents;
    this.votingSystem = options.votingSystem;
    this.mcpServers = options.mcpServers;
    this.io = options.io;
    
    this.sessions = new Map();
    this.conversationHistory = new Map();
    
    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.label({ label: 'ChatbotInterface' }),
        winston.format.json()
      ),
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/chatbot.log' })
      ]
    });

    this.initializeSystemPrompt();
  }

  initializeSystemPrompt() {
    this.systemPrompt = `You are an AI assistant for a comprehensive media server system with the following capabilities:

AVAILABLE SERVICES:
- Jellyfin: Media streaming and library management (port 8096)
- Sonarr: TV show automation (port 8989) 
- Radarr: Movie automation (port 7878)
- Prowlarr: Indexer management (port 9696)
- qBittorrent: Torrent client (port 8080)

MCP TOOLS AVAILABLE:
You have access to Model Context Protocol (MCP) servers for each service that provide:
- Real-time data access and management
- Service control and automation
- Statistics and monitoring
- Configuration management

AI AGENT VOTING SYSTEM:
For complex decisions, you can invoke a democratic voting system with 6 specialized AI agents:
1. Media Curator - Content organization specialist
2. Technical Specialist - Performance and optimization expert  
3. User Advocate - User experience champion
4. Automation Expert - Workflow optimization specialist
5. Security Guardian - Security and privacy protection
6. Trend Analyst - Media trends and social insights

DECISION-MAKING PROCESS:
- Simple queries: Answer directly using MCP tools
- Complex decisions: Create a vote proposal for agent consensus
- User requests: Always prioritize user preferences and safety

CAPABILITIES:
- Search and manage media libraries
- Control playback and downloads
- Monitor system performance
- Automate media acquisition
- Provide recommendations
- Troubleshoot issues
- Gather social media insights

IMPORTANT GUIDELINES:
- Always explain what actions you're taking
- Ask for confirmation before making significant changes
- Provide clear status updates during operations
- Use agent voting for decisions affecting system behavior
- Respect user privacy and security
- Suggest optimizations and improvements

You should be helpful, informative, and proactive in managing the media server ecosystem.`;
  }

  async processMessage(message, sessionId = null) {
    try {
      // Get or create session
      const session = this.getOrCreateSession(sessionId);
      
      // Add user message to conversation history
      this.addToHistory(session.id, 'user', message);

      // Analyze message intent and determine if voting is needed
      const analysis = await this.analyzeMessageIntent(message, session);
      
      let response;
      
      if (analysis.requiresVoting) {
        // Complex decision - use agent voting
        response = await this.handleVotingDecision(message, analysis, session);
      } else if (analysis.mcpCalls.length > 0) {
        // Direct MCP operation
        response = await this.handleMCPOperation(message, analysis, session);
      } else {
        // General conversation
        response = await this.handleGeneralConversation(message, session);
      }

      // Add assistant response to history
      this.addToHistory(session.id, 'assistant', response.content);

      // Update session
      session.lastActivity = new Date();
      session.messageCount++;

      return {
        sessionId: session.id,
        content: response.content,
        type: response.type || 'general',
        agentVotes: response.agentVotes || null,
        mcpCalls: response.mcpCalls || [],
        suggestions: response.suggestions || [],
        timestamp: new Date()
      };

    } catch (error) {
      this.logger.error('Error processing message:', error);
      
      return {
        sessionId: sessionId || 'error',
        content: 'I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists.',
        type: 'error',
        error: error.message,
        timestamp: new Date()
      };
    }
  }

  getOrCreateSession(sessionId) {
    if (sessionId && this.sessions.has(sessionId)) {
      return this.sessions.get(sessionId);
    }

    const newSessionId = sessionId || uuidv4();
    const session = {
      id: newSessionId,
      createdAt: new Date(),
      lastActivity: new Date(),
      messageCount: 0,
      context: {}
    };

    this.sessions.set(newSessionId, session);
    this.conversationHistory.set(newSessionId, []);

    return session;
  }

  addToHistory(sessionId, role, content) {
    if (!this.conversationHistory.has(sessionId)) {
      this.conversationHistory.set(sessionId, []);
    }

    const history = this.conversationHistory.get(sessionId);
    history.push({
      role,
      content,
      timestamp: new Date()
    });

    // Keep only last 20 messages to manage token usage
    if (history.length > 20) {
      history.splice(0, history.length - 20);
    }
  }

  async analyzeMessageIntent(message, session) {
    const analysisPrompt = `Analyze this user message and determine the appropriate response strategy:

User Message: "${message}"

Context: Media server management system with Jellyfin, Sonarr, Radarr, Prowlarr, and qBittorrent.

Please respond with a JSON object containing:
{
  "intent": "query|action|configuration|troubleshoot|recommendation",
  "complexity": "simple|moderate|complex",
  "requiresVoting": boolean,
  "mcpCalls": [
    {
      "service": "jellyfin|sonarr|radarr|prowlarr|qbittorrent",
      "tool": "tool_name",
      "parameters": {}
    }
  ],
  "votingTopic": "description if voting required",
  "confidence": 0.95
}

Guidelines:
- requiresVoting: true for system changes, configuration updates, or decisions affecting multiple services
- mcpCalls: specific tool calls needed to fulfill the request
- complex: requires multiple steps or affects system behavior
- simple: information queries or status checks`;

    try {
      const response = await this.openai.chat.completions.create({
        model: process.env.OPENAI_INTENT_MODEL || 'gpt-4o-mini', // Use faster model for analysis or override via env
        messages: [
          { role: 'system', content: 'You are an intent analysis system for a media server assistant.' },
          { role: 'user', content: analysisPrompt }
        ],
        temperature: 0.1,
        max_tokens: 500
      });

      const content = response.choices[0].message.content;
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (error) {
      this.logger.error('Intent analysis failed:', error);
    }

    // Fallback analysis
    const hasActionWords = /\b(add|remove|delete|configure|setup|install|update|change|modify|download|search|play|stop|start|restart)\b/i.test(message);
    const hasComplexWords = /\b(optimize|recommend|best|should|configure|setup|migration|integration)\b/i.test(message);

    return {
      intent: hasActionWords ? 'action' : 'query',
      complexity: hasComplexWords ? 'complex' : 'simple',
      requiresVoting: hasActionWords && hasComplexWords,
      mcpCalls: [],
      votingTopic: hasActionWords && hasComplexWords ? message : null,
      confidence: 0.7
    };
  }

  async handleVotingDecision(message, analysis, session) {
    this.logger.info('Initiating agent voting for decision:', analysis.votingTopic);

    // Create voting proposal
    const proposal = {
      title: `User Request: ${analysis.votingTopic}`,
      description: message,
      context: `Session: ${session.id}, Intent: ${analysis.intent}`,
      systemState: await this.gatherSystemState(),
      timeoutMinutes: 3,
      includeSocialResearch: analysis.intent === 'recommendation'
    };

    // Submit to voting system
    const voteId = await this.votingSystem.createVote(proposal);
    
    // Wait for voting to complete (with timeout)
    const result = await this.waitForVoteCompletion(voteId, 180000); // 3 minutes

    // Generate response based on voting result
    const responseContent = await this.generateVotingResponse(message, result, session);

    return {
      content: responseContent,
      type: 'voting-decision',
      agentVotes: result,
      mcpCalls: []
    };
  }

  async handleMCPOperation(message, analysis, session) {
    this.logger.info('Executing MCP operations:', analysis.mcpCalls);

    const mcpResults = [];
    
    for (const mcpCall of analysis.mcpCalls) {
      try {
        const server = this.mcpServers[mcpCall.service];
        if (!server) {
          throw new Error(`MCP server ${mcpCall.service} not available`);
        }

        // Execute the MCP tool call
        const result = await server.callTool(mcpCall.tool, mcpCall.parameters);
        mcpResults.push({
          service: mcpCall.service,
          tool: mcpCall.tool,
          result,
          success: true
        });

      } catch (error) {
        this.logger.error(`MCP call failed:`, error);
        mcpResults.push({
          service: mcpCall.service,
          tool: mcpCall.tool,
          error: error.message,
          success: false
        });
      }
    }

    // Generate response based on MCP results
    const responseContent = await this.generateMCPResponse(message, mcpResults, session);

    return {
      content: responseContent,
      type: 'mcp-operation',
      mcpCalls: mcpResults
    };
  }

  async handleGeneralConversation(message, session) {
    const history = this.conversationHistory.get(session.id) || [];
    
    // Prepare conversation with system state
    const systemState = await this.gatherSystemState();
    const contextPrompt = `Current System Status:
${JSON.stringify(systemState, null, 2)}

Recent conversation history available for context.`;

    const messages = [
      { role: 'system', content: this.systemPrompt },
      { role: 'system', content: contextPrompt },
      ...history.slice(-10).map(h => ({ role: h.role, content: h.content })),
      { role: 'user', content: message }
    ];

    try {
      let content;
      if (this.model && this.model.startsWith('o3')) {
        const response = await this.openai.responses.create({
          model: this.model,
          input: messages
        });
        content = this.extractResponseText(response);
      } else {
        const response = await this.openai.chat.completions.create({
          model: this.model,
          messages,
          temperature: 0.7,
          max_tokens: 1000
        });
        content = response.choices[0].message.content;
      }
      
      // Extract any suggestions from the response
      const suggestions = this.extractSuggestions(content);

      return {
        content,
        type: 'conversation',
        suggestions
      };

    } catch (error) {
      this.logger.error('OpenAI API error:', error);
      throw error;
    }
  }

  async gatherSystemState() {
    const state = {};

    // Gather status from each MCP server
    for (const [serviceName, server] of Object.entries(this.mcpServers)) {
      try {
        if (server.isRunning) {
          state[serviceName] = {
            status: 'running',
            port: server.port,
            lastActivity: server.lastActivity,
            requestCount: server.requestCount || 0
          };
        } else {
          state[serviceName] = { status: 'stopped' };
        }
      } catch (error) {
        state[serviceName] = { status: 'error', error: error.message };
      }
    }

    // Add voting system status
    if (this.votingSystem) {
      state.votingSystem = this.votingSystem.getSystemStats();
    }

    return state;
  }

  extractResponseText(response) {
    try {
      if (response.output_text) return response.output_text;
      if (response.output && Array.isArray(response.output)) {
        const texts = [];
        for (const item of response.output) {
          if (item && Array.isArray(item.content)) {
            for (const c of item.content) {
              if (c.type === 'output_text' && c.text) texts.push(c.text);
              if (c.type === 'text' && c.text) texts.push(c.text);
            }
          }
        }
        return texts.join('\n').trim();
      }
    } catch (e) {
      // ignore
    }
    return '';
  }

  async waitForVoteCompletion(voteId, timeoutMs) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Vote timeout'));
      }, timeoutMs);

      const checkInterval = setInterval(async () => {
        try {
          const vote = this.votingSystem.voteHistory.find(v => v.id === voteId);
          if (vote && vote.status === 'completed') {
            clearTimeout(timeout);
            clearInterval(checkInterval);
            resolve(vote.result);
          }
        } catch (error) {
          clearTimeout(timeout);
          clearInterval(checkInterval);
          reject(error);
        }
      }, 1000);
    });
  }

  async generateVotingResponse(message, voteResult, session) {
    const prompt = `Generate a response to the user based on the AI agent voting results:

User Request: "${message}"

Voting Results:
- Decision: ${voteResult.decision}
- Consensus: ${(voteResult.consensus * 100).toFixed(1)}%
- Confidence: ${(voteResult.confidence * 100).toFixed(1)}%
- Summary: ${voteResult.summary}

Agent Insights:
- Risks: ${voteResult.details.risks.join(', ') || 'None identified'}
- Benefits: ${voteResult.details.benefits.join(', ') || 'None identified'}
- Alternatives: ${voteResult.details.alternatives.join(', ') || 'None suggested'}

Please provide a clear explanation of:
1. What the agents decided
2. Why they made this decision
3. Next steps or recommendations
4. Any important considerations

Keep the response helpful and actionable.`;

    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are explaining AI agent voting results to a user.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 600
      });

      return response.choices[0].message.content;

    } catch (error) {
      this.logger.error('Failed to generate voting response:', error);
      return `The AI agents have reached a decision: **${voteResult.decision.toUpperCase()}** with ${(voteResult.consensus * 100).toFixed(1)}% consensus.\n\nSummary: ${voteResult.summary}\n\nWould you like me to proceed with this recommendation?`;
    }
  }

  async generateMCPResponse(message, mcpResults, session) {
    const prompt = `Generate a response based on MCP tool execution results:

User Request: "${message}"

MCP Results:
${mcpResults.map(result => `
Service: ${result.service}
Tool: ${result.tool}
Success: ${result.success}
${result.success ? `Result: ${JSON.stringify(result.result, null, 2)}` : `Error: ${result.error}`}
`).join('\n')}

Please provide a clear summary of what was accomplished and any relevant information for the user.`;

    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are summarizing media server operation results.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 600
      });

      return response.choices[0].message.content;

    } catch (error) {
      this.logger.error('Failed to generate MCP response:', error);
      
      const successCount = mcpResults.filter(r => r.success).length;
      return `Executed ${successCount}/${mcpResults.length} operations successfully. ${successCount > 0 ? 'Check the results above.' : 'Please try again or check system status.'}`;
    }
  }

  extractSuggestions(content) {
    // Simple regex to extract suggestions
    const suggestionPatterns = [
      /I suggest (.*?)(?:\.|$)/gi,
      /You might want to (.*?)(?:\.|$)/gi,
      /Consider (.*?)(?:\.|$)/gi,
      /I recommend (.*?)(?:\.|$)/gi
    ];

    const suggestions = [];
    
    suggestionPatterns.forEach(pattern => {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        if (match[1] && match[1].length > 10) {
          suggestions.push(match[1].trim());
        }
      }
    });

    return suggestions.slice(0, 3); // Limit to 3 suggestions
  }

  getSessionStats() {
    return {
      activeSessions: this.sessions.size,
      totalMessages: Array.from(this.conversationHistory.values())
        .reduce((sum, history) => sum + history.length, 0),
      averageSessionLength: this.sessions.size > 0 ? 
        Array.from(this.sessions.values())
          .reduce((sum, session) => sum + session.messageCount, 0) / this.sessions.size : 0
    };
  }

  cleanup() {
    // Clean up old sessions (older than 1 hour)
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    
    for (const [sessionId, session] of this.sessions.entries()) {
      if (session.lastActivity < oneHourAgo) {
        this.sessions.delete(sessionId);
        this.conversationHistory.delete(sessionId);
      }
    }
  }
}

module.exports = ChatbotInterface;