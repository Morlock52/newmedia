/**
 * AI Agent Voting System
 * Implements democratic decision-making for media server management
 * 
 * Features:
 * - Multiple voting protocols (binary, ranked, weighted)
 * - OpenAI o1-mini powered agents with specialized roles
 * - Real-time consensus tracking
 * - Vote aggregation and analysis
 * - Social media research integration
 */

const OpenAI = require('openai');
const winston = require('winston');
const { v4: uuidv4 } = require('uuid');

class VotingSystem {
  constructor(options = {}) {
    this.openai = new OpenAI({
      apiKey: options.openaiApiKey
    });
    
    this.model = options.model || 'o1-mini';
    this.io = options.io;
    
    this.agents = new Map();
    this.activeVotes = new Map();
    this.voteHistory = [];
    this.consensusThreshold = options.consensusThreshold || 0.7;
    
    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.label({ label: 'VotingSystem' }),
        winston.format.json()
      ),
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/voting-system.log' })
      ]
    });

    this.initializeAgents();
  }

  initializeAgents() {
    // Define specialized agent roles with voting weights
    const agentDefinitions = [
      {
        id: 'media-curator',
        name: 'Media Curator',
        role: 'Content organization and metadata management specialist',
        weight: 1.2,
        expertise: ['content-organization', 'metadata', 'library-management'],
        systemPrompt: `You are a Media Curator AI agent specializing in content organization and library management. 
        Your expertise includes:
        - Organizing media libraries efficiently
        - Managing metadata and tagging
        - Content discovery and recommendation
        - Quality assessment of media files
        
        When voting on decisions, consider:
        - User experience and accessibility
        - Content organization best practices
        - Metadata accuracy and completeness
        - Library structure and navigation
        
        Always provide reasoning for your votes and suggest alternatives when appropriate.`
      },
      {
        id: 'technical-specialist',
        name: 'Technical Specialist',
        role: 'Server performance and technical optimization expert',
        weight: 1.3,
        expertise: ['performance', 'system-optimization', 'troubleshooting'],
        systemPrompt: `You are a Technical Specialist AI agent focused on server performance and system optimization.
        Your expertise includes:
        - Server performance monitoring and optimization
        - Resource allocation and management
        - Technical troubleshooting
        - System architecture and scaling
        
        When voting on decisions, prioritize:
        - System performance and reliability
        - Resource efficiency
        - Scalability considerations
        - Technical feasibility
        
        Provide technical analysis and performance impact assessments in your votes.`
      },
      {
        id: 'user-advocate',
        name: 'User Advocate',
        role: 'User experience and accessibility champion',
        weight: 1.1,
        expertise: ['user-experience', 'accessibility', 'interface-design'],
        systemPrompt: `You are a User Advocate AI agent championing user experience and accessibility.
        Your expertise includes:
        - User interface design and usability
        - Accessibility standards and compliance
        - User behavior analysis
        - Feature adoption and user satisfaction
        
        When voting on decisions, emphasize:
        - User-friendly interfaces and workflows
        - Accessibility for all users
        - Intuitive navigation and discovery
        - User feedback and satisfaction
        
        Always consider the end-user impact in your voting decisions.`
      },
      {
        id: 'automation-expert',
        name: 'Automation Expert',
        role: 'Process automation and workflow optimization specialist',
        weight: 1.2,
        expertise: ['automation', 'workflow-optimization', 'integration'],
        systemPrompt: `You are an Automation Expert AI agent specializing in process automation and workflow optimization.
        Your expertise includes:
        - Automated media management workflows
        - API integrations and data synchronization
        - Scheduled tasks and maintenance
        - Cross-service coordination
        
        When voting on decisions, focus on:
        - Automation opportunities and efficiency
        - Workflow streamlining
        - Integration possibilities
        - Maintenance and operational overhead
        
        Evaluate proposals for their automation potential and operational impact.`
      },
      {
        id: 'security-guardian',
        name: 'Security Guardian',
        role: 'Security and privacy protection specialist',
        weight: 1.4,
        expertise: ['security', 'privacy', 'access-control'],
        systemPrompt: `You are a Security Guardian AI agent focused on security and privacy protection.
        Your expertise includes:
        - Access control and user management
        - Data privacy and protection
        - Security vulnerability assessment
        - Compliance and best practices
        
        When voting on decisions, prioritize:
        - Security implications and risks
        - Privacy protection measures
        - Access control and permissions
        - Compliance with security standards
        
        Always assess security risks and recommend protective measures in your votes.`
      },
      {
        id: 'trend-analyst',
        name: 'Trend Analyst',
        role: 'Media trends and social insights specialist',
        weight: 1.0,
        expertise: ['trend-analysis', 'social-media', 'content-trends'],
        systemPrompt: `You are a Trend Analyst AI agent specializing in media trends and social insights.
        Your expertise includes:
        - Current media and entertainment trends
        - Social media sentiment analysis
        - Content popularity prediction
        - Cultural and demographic insights
        
        When voting on decisions, consider:
        - Current trends and user preferences
        - Social media feedback and sentiment
        - Emerging technologies and features
        - Cultural relevance and appeal
        
        Provide trend analysis and social context in your voting decisions.`
      }
    ];

    // Initialize agents
    agentDefinitions.forEach(agentDef => {
      this.agents.set(agentDef.id, {
        ...agentDef,
        isActive: true,
        voteHistory: [],
        lastActivity: new Date(),
        totalVotes: 0,
        averageConfidence: 0
      });
    });

    this.logger.info(`Initialized ${this.agents.size} voting agents`);
  }

  async createVote(proposal) {
    const voteId = uuidv4();
    const vote = {
      id: voteId,
      proposal,
      status: 'active',
      createdAt: new Date(),
      deadline: new Date(Date.now() + (proposal.timeoutMinutes || 5) * 60 * 1000),
      votes: new Map(),
      socialContext: null,
      result: null
    };

    this.activeVotes.set(voteId, vote);

    // Gather social context if requested
    if (proposal.includeSocialResearch) {
      try {
        vote.socialContext = await this.gatherSocialContext(proposal);
      } catch (error) {
        this.logger.error('Failed to gather social context:', error);
      }
    }

    // Notify all agents to vote
    this.notifyAgentsToVote(voteId);

    // Emit vote creation to WebSocket clients
    if (this.io) {
      this.io.to('voting-updates').emit('vote-created', {
        voteId,
        proposal: proposal.title || proposal.description,
        deadline: vote.deadline,
        agentCount: this.agents.size
      });
    }

    this.logger.info(`Created vote: ${voteId}`, { proposal: proposal.title });

    return voteId;
  }

  async notifyAgentsToVote(voteId) {
    const vote = this.activeVotes.get(voteId);
    if (!vote) return;

    const votingPromises = Array.from(this.agents.entries()).map(async ([agentId, agent]) => {
      try {
        const agentVote = await this.getAgentVote(agentId, vote);
        vote.votes.set(agentId, agentVote);
        
        // Update agent statistics
        agent.totalVotes++;
        agent.voteHistory.push({
          voteId,
          decision: agentVote.decision,
          confidence: agentVote.confidence,
          timestamp: new Date()
        });
        agent.lastActivity = new Date();

        // Emit individual vote to WebSocket clients
        if (this.io) {
          this.io.to('voting-updates').emit('agent-voted', {
            voteId,
            agentId,
            agentName: agent.name,
            decision: agentVote.decision,
            confidence: agentVote.confidence
          });
        }

      } catch (error) {
        this.logger.error(`Agent ${agentId} failed to vote:`, error);
        
        // Record a failed vote
        vote.votes.set(agentId, {
          decision: 'abstain',
          confidence: 0,
          reasoning: 'Failed to generate vote due to error',
          error: error.message,
          timestamp: new Date()
        });
      }
    });

    await Promise.allSettled(votingPromises);

    // Calculate results
    const result = this.calculateVoteResult(vote);
    vote.result = result;
    vote.status = 'completed';

    // Move to history
    this.voteHistory.push(vote);
    this.activeVotes.delete(voteId);

    // Emit final result
    if (this.io) {
      this.io.to('voting-updates').emit('vote-completed', {
        voteId,
        result,
        consensus: result.consensus,
        summary: result.summary
      });
    }

    this.logger.info(`Vote completed: ${voteId}`, { result });

    return result;
  }

  async getAgentVote(agentId, vote) {
    const agent = this.agents.get(agentId);
    if (!agent || !agent.isActive) {
      throw new Error(`Agent ${agentId} is not available`);
    }

    const prompt = this.buildVotingPrompt(agent, vote);

    try {
      let content;
      if (this.model && this.model.startsWith('o3')) {
        // Use Responses API for O3 family
        const input = `SYSTEM:\n${agent.systemPrompt}\n\nUSER:\n${prompt}`;
        const response = await this.openai.responses.create({
          model: this.model,
          input
        });
        content = this.extractResponseText(response);
      } else {
        const response = await this.openai.chat.completions.create({
          model: this.model,
          messages: [
            { role: 'system', content: agent.systemPrompt },
            { role: 'user', content: prompt }
          ],
          temperature: 0.3,
          max_tokens: 1000
        });
        content = response.choices[0].message.content;
      }

      const parsedVote = this.parseAgentResponse(content);

      return {
        ...parsedVote,
        agentId,
        timestamp: new Date(),
        model: this.model,
        rawResponse: content
      };

    } catch (error) {
      this.logger.error(`OpenAI API error for agent ${agentId}:`, error);
      throw error;
    }
  }

  buildVotingPrompt(agent, vote) {
    const { proposal } = vote;
    
    let prompt = `
VOTING REQUEST
==============

Proposal: ${proposal.title || 'Media Server Decision'}
Description: ${proposal.description}

Context:
${proposal.context ? `- ${proposal.context}` : '- No additional context provided'}

Current System State:
${proposal.systemState ? JSON.stringify(proposal.systemState, null, 2) : '- System state not provided'}

Your Expertise: ${agent.expertise.join(', ')}

`;

    if (vote.socialContext) {
      prompt += `
Social Media Research:
${JSON.stringify(vote.socialContext, null, 2)}

`;
    }

    if (proposal.options && proposal.options.length > 0) {
      prompt += `
Available Options:
${proposal.options.map((option, index) => `${index + 1}. ${option}`).join('\n')}

Please vote for one of the numbered options above.
`;
    }

    prompt += `
VOTING INSTRUCTIONS:
===================

Please provide your vote in the following JSON format:

{
  "decision": "approve|reject|option-1|option-2|etc",
  "confidence": 0.85,
  "reasoning": "Your detailed reasoning here",
  "alternatives": ["Alternative suggestion 1", "Alternative suggestion 2"],
  "risks": ["Risk 1", "Risk 2"],
  "benefits": ["Benefit 1", "Benefit 2"],
  "priority": "high|medium|low"
}

Key Guidelines:
- Use your expertise in ${agent.expertise.join(', ')} to inform your decision
- Provide confidence as a decimal between 0.0 and 1.0
- Give detailed reasoning that others can understand
- Suggest alternatives if you reject the proposal
- Identify potential risks and benefits
- Assign priority level based on impact

Your vote matters in the collective decision-making process!
`;

    return prompt;
  }

  parseAgentResponse(content) {
    try {
      // Try to extract JSON from the response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        
        // Validate required fields
        if (!parsed.decision || typeof parsed.confidence !== 'number') {
          throw new Error('Invalid vote format');
        }

        return {
          decision: parsed.decision,
          confidence: Math.max(0, Math.min(1, parsed.confidence)),
          reasoning: parsed.reasoning || 'No reasoning provided',
          alternatives: parsed.alternatives || [],
          risks: parsed.risks || [],
          benefits: parsed.benefits || [],
          priority: parsed.priority || 'medium'
        };
      }
    } catch (error) {
      this.logger.error('Failed to parse agent response:', error);
    }

    // Fallback parsing
    return {
      decision: 'abstain',
      confidence: 0.5,
      reasoning: 'Failed to parse vote response',
      alternatives: [],
      risks: [],
      benefits: [],
      priority: 'medium'
    };
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

  calculateVoteResult(vote) {
    const votes = Array.from(vote.votes.values());
    
    if (votes.length === 0) {
      return {
        decision: 'no-consensus',
        consensus: 0,
        summary: 'No votes received',
        details: {}
      };
    }

    // Count decisions with weights
    const decisionCounts = new Map();
    const totalWeight = Array.from(this.agents.values())
      .reduce((sum, agent) => sum + agent.weight, 0);
    
    let weightedScores = new Map();

    votes.forEach(vote => {
      const agent = this.agents.get(vote.agentId);
      const weight = agent ? agent.weight : 1.0;
      const weightedVote = weight * vote.confidence;

      if (!decisionCounts.has(vote.decision)) {
        decisionCounts.set(vote.decision, 0);
        weightedScores.set(vote.decision, 0);
      }

      decisionCounts.set(vote.decision, decisionCounts.get(vote.decision) + 1);
      weightedScores.set(vote.decision, weightedScores.get(vote.decision) + weightedVote);
    });

    // Find winning decision
    let winningDecision = 'no-consensus';
    let maxScore = 0;
    let consensus = 0;

    for (const [decision, score] of weightedScores.entries()) {
      const normalizedScore = score / totalWeight;
      if (normalizedScore > maxScore) {
        maxScore = normalizedScore;
        winningDecision = decision;
      }
    }

    consensus = maxScore;

    // Aggregate reasoning and insights
    const allReasoning = votes.map(v => v.reasoning).filter(r => r);
    const allRisks = votes.flatMap(v => v.risks || []);
    const allBenefits = votes.flatMap(v => v.benefits || []);
    const allAlternatives = votes.flatMap(v => v.alternatives || []);

    const averageConfidence = votes.reduce((sum, v) => sum + v.confidence, 0) / votes.length;

    return {
      decision: winningDecision,
      consensus,
      confidence: averageConfidence,
      reachedThreshold: consensus >= this.consensusThreshold,
      summary: this.generateDecisionSummary(winningDecision, consensus, votes),
      details: {
        totalVotes: votes.length,
        decisionBreakdown: Object.fromEntries(decisionCounts),
        weightedScores: Object.fromEntries(weightedScores),
        reasoning: allReasoning,
        risks: [...new Set(allRisks)],
        benefits: [...new Set(allBenefits)],
        alternatives: [...new Set(allAlternatives)],
        averageConfidence,
        consensusThreshold: this.consensusThreshold
      }
    };
  }

  generateDecisionSummary(decision, consensus, votes) {
    const consensusLevel = consensus >= 0.8 ? 'strong' : 
                          consensus >= 0.6 ? 'moderate' : 'weak';
    
    const supportingAgents = votes.filter(v => v.decision === decision).length;
    const totalAgents = votes.length;

    return `${consensusLevel.toUpperCase()} CONSENSUS: ${decision.toUpperCase()} (${supportingAgents}/${totalAgents} agents, ${(consensus * 100).toFixed(1)}% weighted agreement)`;
  }

  async gatherSocialContext(proposal) {
    // This would integrate with social media APIs to gather relevant context
    // For now, return a placeholder structure
    return {
      platforms: ['twitter', 'reddit', 'github'],
      sentiment: 'neutral',
      mentions: 0,
      trends: [],
      timestamp: new Date()
    };
  }

  getActiveAgents() {
    return Array.from(this.agents.entries()).map(([id, agent]) => ({
      id,
      name: agent.name,
      role: agent.role,
      expertise: agent.expertise,
      weight: agent.weight,
      isActive: agent.isActive,
      totalVotes: agent.totalVotes,
      lastActivity: agent.lastActivity
    }));
  }

  getRecentVotes(limit = 10) {
    return this.voteHistory
      .slice(-limit)
      .map(vote => ({
        id: vote.id,
        proposal: vote.proposal.title || vote.proposal.description,
        decision: vote.result?.decision,
        consensus: vote.result?.consensus,
        createdAt: vote.createdAt,
        voteCount: vote.votes.size
      }));
  }

  getSystemStats() {
    const totalVotes = this.voteHistory.length;
    const completedVotes = this.voteHistory.filter(v => v.status === 'completed').length;
    const averageConsensus = this.voteHistory
      .filter(v => v.result?.consensus)
      .reduce((sum, v) => sum + v.result.consensus, 0) / completedVotes || 0;

    return {
      totalAgents: this.agents.size,
      activeAgents: Array.from(this.agents.values()).filter(a => a.isActive).length,
      totalVotes,
      completedVotes,
      activeVotes: this.activeVotes.size,
      averageConsensus: Math.round(averageConsensus * 100) / 100,
      consensusThreshold: this.consensusThreshold
    };
  }

  async submitDecision(proposal) {
    return await this.createVote(proposal);
  }
}

module.exports = VotingSystem;