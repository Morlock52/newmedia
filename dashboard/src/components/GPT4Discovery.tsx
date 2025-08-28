import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import OpenAI from 'openai';

interface SearchResult {
  id: string;
  title: string;
  type: 'movie' | 'tv' | 'documentary' | 'anime' | 'music' | 'podcast';
  year: number;
  rating: number;
  genres: string[];
  description: string;
  thumbnail: string;
  duration: number;
  language: string;
  relevanceScore: number;
  reasoning: string;
  cast?: string[];
  director?: string;
  network?: string;
  availability: {
    service: string;
    quality: string[];
    downloadable: boolean;
  }[];
  tags: string[];
  mood: string[];
  similarTo: string[];
}

interface SearchContext {
  mood?: string;
  time?: 'short' | 'medium' | 'long' | 'any';
  genre?: string[];
  language?: string;
  rating?: 'any' | 'family' | 'teen' | 'mature';
  freshness?: 'any' | 'new' | 'classic' | 'recent';
  platform?: string[];
  watchWith?: 'solo' | 'friends' | 'family' | 'date';
}

interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  searchResults?: SearchResult[];
  context?: SearchContext;
  suggestions?: string[];
}

interface GPT4DiscoveryProps {
  onSelectMedia?: (media: SearchResult) => void;
  enableVoiceInput?: boolean;
  enableContextAwareness?: boolean;
  maxResults?: number;
  apiKey?: string;
}

const GPT4Discovery: React.FC<GPT4DiscoveryProps> = ({
  onSelectMedia,
  enableVoiceInput = true,
  enableContextAwareness = true,
  maxResults = 12,
  apiKey
}) => {
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [currentContext, setCurrentContext] = useState<SearchContext>({});
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [searchMode, setSearchMode] = useState<'natural' | 'advanced' | 'quick'>('natural');
  const [filters, setFilters] = useState<SearchContext>({});
  const [recentSearches, setRecentSearches] = useState<SearchResult[]>([]);
  const [trendingTopics, setTrendingTopics] = useState<string[]>([]);

  const openaiRef = useRef<OpenAI | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognition | null>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Initialize OpenAI and speech recognition
  useEffect(() => {
    if (apiKey) {
      openaiRef.current = new OpenAI({
        apiKey: apiKey,
        dangerouslyAllowBrowser: true // Note: In production, use a server-side proxy
      });
    }

    if (enableVoiceInput && 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      speechRecognitionRef.current = new SpeechRecognition();
      speechRecognitionRef.current.continuous = false;
      speechRecognitionRef.current.interimResults = false;
      speechRecognitionRef.current.lang = 'en-US';

      speechRecognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputText(transcript);
        setIsListening(false);
      };

      speechRecognitionRef.current.onerror = () => {
        setIsListening(false);
      };

      speechRecognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    // Load trending topics and suggestions
    loadTrendingTopics();
    loadSearchSuggestions();

    // Add welcome message
    addMessage({
      role: 'assistant',
      content: "Hi! I'm your AI-powered content discovery assistant. Ask me anything like:\n\n• \"Find me a sci-fi movie like Blade Runner\"\n• \"What's good to watch on a rainy evening?\"\n• \"Show me documentaries about space\"\n• \"I want something funny and light\"\n\nI understand natural language and can help you discover exactly what you're in the mood for!",
      suggestions: [
        "What's trending right now?",
        "Find me something to watch with friends",
        "I want to learn something new",
        "Show me award-winning movies"
      ]
    });
  }, [apiKey, enableVoiceInput]);

  // Auto-scroll conversation
  useEffect(() => {
    if (conversationRef.current) {
      conversationRef.current.scrollTop = conversationRef.current.scrollHeight;
    }
  }, [conversation]);

  const loadTrendingTopics = () => {
    // Simulate trending topics (in production, fetch from API)
    const topics = [
      "AI documentaries", "Korean dramas", "Marvel movies", "True crime",
      "Cooking shows", "Space exploration", "Comedy specials", "Anime series",
      "Historical dramas", "Psychological thrillers", "Nature documentaries",
      "Stand-up comedy", "Sci-fi series", "Romance movies", "Action films"
    ];
    setTrendingTopics(topics);
  };

  const loadSearchSuggestions = () => {
    const suggestions = [
      "What's new on Netflix?",
      "Find me a feel-good movie",
      "Show me something like Stranger Things",
      "I want to watch something educational",
      "What's good for a date night?",
      "Find critically acclaimed series",
      "Show me award-winning documentaries",
      "I'm in the mood for comedy",
      "What's trending in anime?",
      "Find family-friendly content"
    ];
    setSuggestions(suggestions);
  };

  const addMessage = (message: Omit<ConversationMessage, 'id' | 'timestamp'>) => {
    const newMessage: ConversationMessage = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      ...message
    };

    setConversation(prev => [...prev, newMessage]);
  };

  const handleSearch = async (query: string, context?: SearchContext) => {
    if (!query.trim()) return;

    setIsSearching(true);
    
    // Add user message
    addMessage({
      role: 'user',
      content: query,
      context: { ...currentContext, ...context }
    });

    // Add to search history
    setSearchHistory(prev => [query, ...prev.slice(0, 9)]);

    try {
      // Use GPT-4 to understand the query and generate search parameters
      const searchParams = await analyzeQuery(query, { ...currentContext, ...context });
      
      // Perform the actual search
      const results = await performMediaSearch(searchParams);
      
      // Generate AI response
      const response = await generateResponse(query, results, searchParams);
      
      // Add assistant message with results
      addMessage({
        role: 'assistant',
        content: response.message,
        searchResults: results,
        suggestions: response.suggestions
      });

      // Update context for next searches
      if (enableContextAwareness) {
        setCurrentContext(prev => ({ ...prev, ...searchParams.context }));
      }

      // Add to recent searches
      setRecentSearches(prev => [...results.slice(0, 3), ...prev.slice(0, 7)]);

    } catch (error) {
      console.error('Search failed:', error);
      addMessage({
        role: 'assistant',
        content: "I'm sorry, I encountered an error while searching. Please try again or rephrase your request."
      });
    } finally {
      setIsSearching(false);
    }
  };

  const analyzeQuery = async (query: string, context: SearchContext) => {
    if (!openaiRef.current) {
      // Fallback to simple keyword extraction
      return extractKeywords(query, context);
    }

    try {
      const response = await openaiRef.current.chat.completions.create({
        model: "gpt-4",
        messages: [
          {
            role: "system",
            content: `You are a media search assistant. Analyze user queries and extract search parameters in JSON format. Include:
            - type: array of media types (movie, tv, documentary, anime, music, podcast)
            - genres: array of genres
            - mood: string describing the mood/tone
            - timeframe: preferred duration (short, medium, long, any)
            - rating: content rating preference
            - keywords: important search terms
            - intent: what the user is looking for
            - context: additional context from conversation
            
            Consider the conversation context: ${JSON.stringify(context)}`
          },
          {
            role: "user",
            content: query
          }
        ],
        temperature: 0.3,
        max_tokens: 500
      });

      const analysis = JSON.parse(response.choices[0].message.content || '{}');
      return {
        ...analysis,
        context: { ...context, ...analysis.context }
      };
    } catch (error) {
      console.error('Query analysis failed:', error);
      return extractKeywords(query, context);
    }
  };

  const extractKeywords = (query: string, context: SearchContext) => {
    const lowerQuery = query.toLowerCase();
    
    // Simple keyword extraction
    const types = [];
    if (lowerQuery.includes('movie') || lowerQuery.includes('film')) types.push('movie');
    if (lowerQuery.includes('tv') || lowerQuery.includes('series') || lowerQuery.includes('show')) types.push('tv');
    if (lowerQuery.includes('documentary') || lowerQuery.includes('doc')) types.push('documentary');
    if (lowerQuery.includes('anime')) types.push('anime');
    if (lowerQuery.includes('music')) types.push('music');
    if (lowerQuery.includes('podcast')) types.push('podcast');

    const genres = [];
    const genreKeywords = {
      'action': ['action', 'fight', 'adventure'],
      'comedy': ['comedy', 'funny', 'humor', 'laugh'],
      'drama': ['drama', 'emotional', 'serious'],
      'horror': ['horror', 'scary', 'fear', 'thriller'],
      'sci-fi': ['sci-fi', 'science fiction', 'space', 'future'],
      'romance': ['romance', 'love', 'romantic'],
      'documentary': ['documentary', 'learn', 'educational']
    };

    Object.entries(genreKeywords).forEach(([genre, keywords]) => {
      if (keywords.some(keyword => lowerQuery.includes(keyword))) {
        genres.push(genre);
      }
    });

    return {
      type: types.length > 0 ? types : ['movie', 'tv'],
      genres,
      keywords: query.split(' ').filter(word => word.length > 2),
      intent: 'search',
      context
    };
  };

  const performMediaSearch = async (searchParams: any): Promise<SearchResult[]> => {
    // Simulate API search (in production, connect to your media database/APIs)
    const mockResults: SearchResult[] = [
      {
        id: '1',
        title: 'Blade Runner 2049',
        type: 'movie',
        year: 2017,
        rating: 8.0,
        genres: ['sci-fi', 'thriller'],
        description: 'A young blade runner discovers a secret that leads him to Rick Deckard, a former blade runner who has been missing for thirty years.',
        thumbnail: '/thumbnails/blade-runner-2049.jpg',
        duration: 164,
        language: 'English',
        relevanceScore: 0.95,
        reasoning: 'Perfect match for sci-fi and cyberpunk themes',
        cast: ['Ryan Gosling', 'Harrison Ford'],
        director: 'Denis Villeneuve',
        availability: [
          { service: 'Netflix', quality: ['4K', '1080p'], downloadable: true },
          { service: 'Prime Video', quality: ['4K', '1080p', '720p'], downloadable: false }
        ],
        tags: ['cyberpunk', 'future', 'AI', 'dystopian'],
        mood: ['thoughtful', 'atmospheric'],
        similarTo: ['Blade Runner', 'Ex Machina', 'Ghost in the Shell']
      },
      {
        id: '2',
        title: 'The Expanse',
        type: 'tv',
        year: 2015,
        rating: 8.5,
        genres: ['sci-fi', 'drama'],
        description: 'A hardboiled detective and a rogue ship\'s officer uncover a conspiracy that threatens the Solar System.',
        thumbnail: '/thumbnails/the-expanse.jpg',
        duration: 45,
        language: 'English',
        relevanceScore: 0.88,
        reasoning: 'Excellent space opera with realistic physics',
        network: 'Amazon Prime',
        availability: [
          { service: 'Prime Video', quality: ['4K', '1080p'], downloadable: true }
        ],
        tags: ['space', 'politics', 'realistic'],
        mood: ['intense', 'complex'],
        similarTo: ['Battlestar Galactica', 'Firefly']
      },
      {
        id: '3',
        title: 'Ghost in the Shell: SAC_2045',
        type: 'anime',
        year: 2020,
        rating: 7.2,
        genres: ['sci-fi', 'action', 'anime'],
        description: 'In 2045, cyborg operatives work to counter new forms of cyber-crime and terrorism.',
        thumbnail: '/thumbnails/gits-sac-2045.jpg',
        duration: 24,
        language: 'Japanese',
        relevanceScore: 0.82,
        reasoning: 'Classic cyberpunk anime series',
        availability: [
          { service: 'Netflix', quality: ['1080p'], downloadable: true }
        ],
        tags: ['cyberpunk', 'philosophy', 'action'],
        mood: ['intense', 'philosophical'],
        similarTo: ['Akira', 'Psycho-Pass']
      }
    ];

    // Filter results based on search parameters
    let filteredResults = mockResults;

    if (searchParams.type?.length > 0) {
      filteredResults = filteredResults.filter(result => 
        searchParams.type.includes(result.type)
      );
    }

    if (searchParams.genres?.length > 0) {
      filteredResults = filteredResults.filter(result =>
        result.genres.some(genre => searchParams.genres.includes(genre))
      );
    }

    // Sort by relevance score
    filteredResults.sort((a, b) => b.relevanceScore - a.relevanceScore);

    return filteredResults.slice(0, maxResults);
  };

  const generateResponse = async (query: string, results: SearchResult[], searchParams: any) => {
    if (!openaiRef.current) {
      return {
        message: `I found ${results.length} results for your search. Here are the top recommendations:`,
        suggestions: [
          "Show me more like this",
          "Find something different",
          "What else is trending?"
        ]
      };
    }

    try {
      const response = await openaiRef.current.chat.completions.create({
        model: "gpt-4",
        messages: [
          {
            role: "system",
            content: `You are a friendly media recommendation assistant. Respond naturally to user queries about movies, TV shows, and other content. Be conversational and helpful. Focus on why the recommendations are good matches.`
          },
          {
            role: "user",
            content: `I searched for: "${query}". Here are the results: ${JSON.stringify(results.map(r => ({
              title: r.title,
              type: r.type,
              year: r.year,
              genres: r.genres,
              rating: r.rating,
              reasoning: r.reasoning
            })))}`
          }
        ],
        temperature: 0.7,
        max_tokens: 300
      });

      return {
        message: response.choices[0].message.content || 'Here are your recommendations:',
        suggestions: [
          "Show me more like this",
          "Find something in a different genre",
          "What's new this week?",
          "Surprise me with something random"
        ]
      };
    } catch (error) {
      console.error('Response generation failed:', error);
      return {
        message: `I found ${results.length} great matches for your search! Here are my top recommendations:`,
        suggestions: [
          "Show me more options",
          "Find something different",
          "What else do you recommend?"
        ]
      };
    }
  };

  const startVoiceInput = () => {
    if (speechRecognitionRef.current && !isListening) {
      setIsListening(true);
      speechRecognitionRef.current.start();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputText(suggestion);
    inputRef.current?.focus();
  };

  const clearConversation = () => {
    setConversation([]);
    setCurrentContext({});
    addMessage({
      role: 'assistant',
      content: "Conversation cleared! What would you like to discover today?",
      suggestions: suggestions.slice(0, 4)
    });
  };

  const quickSearch = (term: string) => {
    setInputText(term);
    handleSearch(term);
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
      border: '2px solid #00ffff',
      borderRadius: '15px',
      padding: '20px',
      color: '#ffffff',
      fontFamily: 'monospace',
      height: '700px',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '20px',
        paddingBottom: '15px',
        borderBottom: '1px solid rgba(0,255,255,0.3)'
      }}>
        <h2 style={{
          color: '#00ffff',
          textShadow: '0 0 10px #00ffff',
          margin: 0,
          fontSize: '20px'
        }}>
          🧠 GPT-4 Content Discovery
        </h2>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {/* Search Mode */}
          <select
            value={searchMode}
            onChange={(e) => setSearchMode(e.target.value as any)}
            style={{
              padding: '5px 8px',
              background: 'rgba(0,0,0,0.7)',
              border: '1px solid #ff00ff',
              borderRadius: '5px',
              color: '#ffffff',
              fontSize: '11px'
            }}
          >
            <option value="natural">Natural Language</option>
            <option value="advanced">Advanced Search</option>
            <option value="quick">Quick Find</option>
          </select>

          <button
            onClick={clearConversation}
            style={{
              padding: '8px 12px',
              background: 'rgba(255,255,0,0.2)',
              border: '1px solid #ffff00',
              borderRadius: '5px',
              color: '#ffff00',
              cursor: 'pointer',
              fontSize: '11px'
            }}
          >
            🔄 Clear
          </button>
        </div>
      </div>

      {/* Quick Access */}
      <div style={{ marginBottom: '15px' }}>
        <div style={{ 
          fontSize: '12px', 
          color: '#cccccc', 
          marginBottom: '8px' 
        }}>
          Quick Discovery:
        </div>
        <div style={{ 
          display: 'flex', 
          gap: '8px', 
          flexWrap: 'wrap' 
        }}>
          {trendingTopics.slice(0, 6).map((topic, index) => (
            <button
              key={index}
              onClick={() => quickSearch(`Show me ${topic}`)}
              style={{
                padding: '4px 8px',
                background: 'rgba(255,0,255,0.2)',
                border: '1px solid rgba(255,0,255,0.3)',
                borderRadius: '12px',
                color: '#ff00ff',
                cursor: 'pointer',
                fontSize: '10px',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,0,255,0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255,0,255,0.2)';
              }}
            >
              {topic}
            </button>
          ))}
        </div>
      </div>

      {/* Conversation */}
      <div 
        ref={conversationRef}
        style={{ 
          flex: 1, 
          overflowY: 'auto',
          marginBottom: '15px',
          padding: '10px',
          background: 'rgba(0,0,0,0.3)',
          borderRadius: '8px',
          border: '1px solid rgba(255,255,255,0.1)'
        }}
      >
        <AnimatePresence>
          {conversation.map((message, index) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              style={{
                marginBottom: '20px',
                display: 'flex',
                flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                alignItems: 'flex-start',
                gap: '10px'
              }}
            >
              {/* Avatar */}
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: message.role === 'user' ? 
                  'linear-gradient(45deg, #00ffff, #ff00ff)' : 
                  'linear-gradient(45deg, #ff00ff, #ffff00)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '14px',
                flexShrink: 0
              }}>
                {message.role === 'user' ? '👤' : '🧠'}
              </div>

              {/* Message Content */}
              <div style={{
                maxWidth: '80%',
                background: message.role === 'user' ? 
                  'rgba(0,255,255,0.1)' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${message.role === 'user' ? 'rgba(0,255,255,0.3)' : 'rgba(255,255,255,0.1)'}`,
                borderRadius: '12px',
                padding: '12px'
              }}>
                {/* Text Content */}
                <div style={{
                  fontSize: '13px',
                  lineHeight: '1.4',
                  marginBottom: message.searchResults || message.suggestions ? '10px' : '0',
                  whiteSpace: 'pre-line'
                }}>
                  {message.content}
                </div>

                {/* Search Results */}
                {message.searchResults && message.searchResults.length > 0 && (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                    gap: '10px',
                    marginBottom: '10px'
                  }}>
                    {message.searchResults.map((result, idx) => (
                      <motion.div
                        key={result.id}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: idx * 0.1 }}
                        onClick={() => onSelectMedia && onSelectMedia(result)}
                        style={{
                          background: 'rgba(0,0,0,0.5)',
                          border: '1px solid rgba(0,255,255,0.3)',
                          borderRadius: '8px',
                          padding: '10px',
                          cursor: 'pointer',
                          transition: 'all 0.3s ease'
                        }}
                        whileHover={{
                          scale: 1.02,
                          boxShadow: '0 5px 15px rgba(0,255,255,0.3)'
                        }}
                      >
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'flex-start',
                          marginBottom: '8px'
                        }}>
                          <h4 style={{
                            color: '#ffffff',
                            margin: 0,
                            fontSize: '13px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            flex: 1
                          }}>
                            {result.title}
                          </h4>
                          <span style={{
                            background: 'rgba(255,255,0,0.2)',
                            color: '#ffff00',
                            padding: '2px 6px',
                            borderRadius: '10px',
                            fontSize: '9px',
                            marginLeft: '8px'
                          }}>
                            {(result.relevanceScore * 100).toFixed(0)}%
                          </span>
                        </div>

                        <div style={{
                          fontSize: '10px',
                          color: '#cccccc',
                          marginBottom: '6px'
                        }}>
                          {result.type.toUpperCase()} • {result.year} • ⭐{result.rating}
                        </div>

                        <div style={{
                          fontSize: '10px',
                          color: '#aaaaaa',
                          marginBottom: '8px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical'
                        }}>
                          {result.description}
                        </div>

                        <div style={{
                          display: 'flex',
                          flexWrap: 'wrap',
                          gap: '4px',
                          marginBottom: '6px'
                        }}>
                          {result.genres.slice(0, 3).map(genre => (
                            <span
                              key={genre}
                              style={{
                                background: 'rgba(255,0,255,0.2)',
                                color: '#ff00ff',
                                padding: '1px 4px',
                                borderRadius: '6px',
                                fontSize: '8px'
                              }}
                            >
                              {genre}
                            </span>
                          ))}
                        </div>

                        <div style={{
                          fontSize: '9px',
                          color: '#ffff00',
                          fontStyle: 'italic'
                        }}>
                          {result.reasoning}
                        </div>

                        {/* Availability */}
                        <div style={{
                          marginTop: '6px',
                          display: 'flex',
                          gap: '4px',
                          flexWrap: 'wrap'
                        }}>
                          {result.availability.slice(0, 2).map((avail, aIdx) => (
                            <span
                              key={aIdx}
                              style={{
                                background: 'rgba(0,255,0,0.2)',
                                color: '#00ff00',
                                padding: '1px 4px',
                                borderRadius: '6px',
                                fontSize: '8px'
                              }}
                            >
                              {avail.service}
                            </span>
                          ))}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}

                {/* Suggestions */}
                {message.suggestions && message.suggestions.length > 0 && (
                  <div style={{
                    display: 'flex',
                    gap: '6px',
                    flexWrap: 'wrap'
                  }}>
                    {message.suggestions.map((suggestion, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSuggestionClick(suggestion)}
                        style={{
                          padding: '4px 8px',
                          background: 'rgba(0,255,255,0.2)',
                          border: '1px solid rgba(0,255,255,0.3)',
                          borderRadius: '10px',
                          color: '#00ffff',
                          cursor: 'pointer',
                          fontSize: '9px',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(0,255,255,0.4)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'rgba(0,255,255,0.2)';
                        }}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                )}

                {/* Timestamp */}
                <div style={{
                  fontSize: '8px',
                  color: '#666666',
                  marginTop: '8px',
                  textAlign: message.role === 'user' ? 'right' : 'left'
                }}>
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading indicator */}
        {isSearching && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              color: '#ffff00',
              fontSize: '12px'
            }}
          >
            <div style={{
              width: '20px',
              height: '20px',
              border: '2px solid rgba(255,255,0,0.3)',
              borderTop: '2px solid #ffff00',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
            Analyzing your request with AI...
          </motion.div>
        )}
      </div>

      {/* Input Area */}
      <div style={{
        display: 'flex',
        gap: '10px',
        alignItems: 'center',
        padding: '10px',
        background: 'rgba(0,0,0,0.3)',
        borderRadius: '8px',
        border: '1px solid rgba(255,255,255,0.1)'
      }}>
        <input
          ref={inputRef}
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSearch(inputText);
              setInputText('');
            }
          }}
          placeholder="Ask me anything about movies, shows, or content..."
          disabled={isSearching}
          style={{
            flex: 1,
            padding: '12px 15px',
            background: 'rgba(255,255,255,0.05)',
            border: '2px solid rgba(0,255,255,0.3)',
            borderRadius: '25px',
            color: '#ffffff',
            fontSize: '13px',
            fontFamily: 'monospace',
            outline: 'none',
            transition: 'border-color 0.3s ease'
          }}
          onFocus={(e) => {
            e.target.style.borderColor = 'rgba(0,255,255,0.6)';
          }}
          onBlur={(e) => {
            e.target.style.borderColor = 'rgba(0,255,255,0.3)';
          }}
        />

        {/* Voice Input */}
        {enableVoiceInput && speechRecognitionRef.current && (
          <button
            onClick={startVoiceInput}
            disabled={isListening || isSearching}
            style={{
              padding: '10px',
              background: isListening ? 'rgba(255,0,0,0.3)' : 'rgba(255,0,255,0.2)',
              border: `2px solid ${isListening ? '#ff0000' : '#ff00ff'}`,
              borderRadius: '50%',
              color: isListening ? '#ff0000' : '#ff00ff',
              cursor: isListening || isSearching ? 'not-allowed' : 'pointer',
              fontSize: '16px',
              width: '45px',
              height: '45px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.3s ease'
            }}
          >
            {isListening ? '🔴' : '🎤'}
          </button>
        )}

        {/* Send Button */}
        <button
          onClick={() => {
            handleSearch(inputText);
            setInputText('');
          }}
          disabled={!inputText.trim() || isSearching}
          style={{
            padding: '10px 15px',
            background: inputText.trim() && !isSearching ? 
              'linear-gradient(45deg, #00ffff, #ff00ff)' : 
              'rgba(128,128,128,0.3)',
            border: 'none',
            borderRadius: '25px',
            color: inputText.trim() && !isSearching ? '#000000' : '#666666',
            cursor: inputText.trim() && !isSearching ? 'pointer' : 'not-allowed',
            fontSize: '14px',
            fontWeight: 'bold',
            transition: 'all 0.3s ease'
          }}
        >
          {isSearching ? '🔍' : '🚀'} Send
        </button>
      </div>

      {/* Status Bar */}
      <div style={{
        marginTop: '10px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '10px',
        color: '#666666'
      }}>
        <div>
          {searchHistory.length > 0 && (
            <span>Recent: {searchHistory[0]}</span>
          )}
        </div>
        <div>
          {enableContextAwareness && Object.keys(currentContext).length > 0 && (
            <span>Context: Active</span>
          )}
        </div>
        <div>
          {apiKey ? '🧠 AI Powered' : '💡 Basic Mode'}
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default GPT4Discovery;