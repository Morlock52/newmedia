import React, { useState, useRef, useEffect } from 'react'
import {
  Card,
  CardContent,
  Typography,
  TextField,
  IconButton,
  Box,
  Chip,
  Avatar,
  LinearProgress,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Fade,
  Tooltip,
} from '@mui/material'
import {
  Send,
  Mic,
  Stop,
  AutoAwesome,
  Psychology,
  SmartToy,
  Clear,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery, useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'

interface Message {
  id: string
  type: 'user' | 'assistant'
  text: string
  timestamp: Date
  suggestions?: string[]
}

interface AIResponse {
  text: string
  suggestions: string[]
  action?: {
    type: string
    payload: any
  }
}

const quickPrompts = [
  'Show system status',
  'What\'s using the most resources?',
  'Recent media additions',
  'Optimize performance',
  'Troubleshoot errors',
  'Backup recommendations',
]

const AIAssistant: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      text: 'Hello! I\'m your AI media assistant. How can I help you manage your media server today?',
      timestamp: new Date(),
      suggestions: quickPrompts.slice(0, 3),
    },
  ])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Initialize speech recognition if available
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = false
      recognitionRef.current.interimResults = true
      recognitionRef.current.lang = 'en-US'

      recognitionRef.current.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((result: any) => result[0].transcript)
          .join('')
        setInput(transcript)
      }

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
        toast.error('Speech recognition failed')
      }

      recognitionRef.current.onend = () => {
        setIsListening(false)
      }
    }
  }, [])

  const sendMessageMutation = useMutation({
    mutationFn: async (message: string): Promise<AIResponse> => {
      // Simulate API call to AI service
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000))
      
      // Mock responses based on keywords
      const lowerMessage = message.toLowerCase()
      
      if (lowerMessage.includes('status')) {
        return {
          text: 'All services are running smoothly! Jellyfin is streaming to 3 devices, Sonarr has 2 shows queued for download, and system resources are at 45% utilization.',
          suggestions: ['Show detailed metrics', 'Check individual services', 'View active streams'],
        }
      } else if (lowerMessage.includes('resource')) {
        return {
          text: 'Jellyfin is currently using the most resources (CPU: 35%, Memory: 2.1GB) due to active transcoding. Consider enabling hardware acceleration for better performance.',
          suggestions: ['Enable hardware acceleration', 'View transcoding settings', 'Optimize Jellyfin'],
        }
      } else if (lowerMessage.includes('media') || lowerMessage.includes('recent')) {
        return {
          text: 'In the last 24 hours, I\'ve added 5 new movies and 12 TV episodes to your library. The latest additions include "Dune: Part Two" and 3 episodes of "The Last of Us".',
          suggestions: ['Show all recent additions', 'Scan for new media', 'View download queue'],
        }
      } else if (lowerMessage.includes('optimize')) {
        return {
          text: 'I recommend the following optimizations: 1) Enable GPU transcoding, 2) Increase cache size for Jellyfin, 3) Schedule library scans during off-peak hours.',
          suggestions: ['Apply optimizations', 'Show advanced settings', 'Schedule maintenance'],
        }
      } else {
        return {
          text: `I understand you're asking about "${message}". Let me help you with that. Based on your media server setup, I can assist with service management, troubleshooting, and optimization.`,
          suggestions: quickPrompts.slice(3, 6),
        }
      }
    },
    onSuccess: (response) => {
      const assistantMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        text: response.text,
        timestamp: new Date(),
        suggestions: response.suggestions,
      }
      setMessages(prev => [...prev, assistantMessage])
      setIsTyping(false)
    },
    onError: (error) => {
      toast.error('Failed to get AI response')
      setIsTyping(false)
    },
  })

  const handleSend = () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      text: input,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsTyping(true)

    sendMessageMutation.mutate(input)
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion)
    handleSend()
  }

  const toggleListening = () => {
    if (!recognitionRef.current) {
      toast.error('Speech recognition not available')
      return
    }

    if (isListening) {
      recognitionRef.current.stop()
    } else {
      recognitionRef.current.start()
      setIsListening(true)
      toast.success('Listening...')
    }
  }

  const clearChat = () => {
    setMessages([messages[0]])
    toast.success('Chat cleared')
  }

  return (
    <Card sx={{ height: 400, display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
              <SmartToy fontSize="small" />
            </Avatar>
            <Typography variant="h6">AI Assistant</Typography>
            <Chip
              icon={<Psychology />}
              label="Neural Network Active"
              size="small"
              color="primary"
              variant="outlined"
            />
          </Box>
          <Tooltip title="Clear chat">
            <IconButton size="small" onClick={clearChat}>
              <Clear />
            </IconButton>
          </Tooltip>
        </Box>
      </CardContent>

      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        <List>
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <ListItem
                  alignItems="flex-start"
                  sx={{
                    flexDirection: message.type === 'user' ? 'row-reverse' : 'row',
                    gap: 1,
                  }}
                >
                  <ListItemAvatar>
                    <Avatar
                      sx={{
                        bgcolor: message.type === 'user' ? 'secondary.main' : 'primary.main',
                        width: 32,
                        height: 32,
                      }}
                    >
                      {message.type === 'user' ? '👤' : '🤖'}
                    </Avatar>
                  </ListItemAvatar>
                  <Box
                    sx={{
                      backgroundColor: message.type === 'user'
                        ? 'rgba(0, 136, 255, 0.1)'
                        : 'rgba(255, 255, 255, 0.05)',
                      borderRadius: 2,
                      p: 2,
                      maxWidth: '80%',
                      border: '1px solid',
                      borderColor: message.type === 'user'
                        ? 'rgba(0, 136, 255, 0.3)'
                        : 'rgba(255, 255, 255, 0.1)',
                    }}
                  >
                    <ListItemText
                      primary={message.text}
                      secondary={message.timestamp.toLocaleTimeString()}
                    />
                    {message.suggestions && (
                      <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {message.suggestions.map((suggestion, idx) => (
                          <Chip
                            key={idx}
                            label={suggestion}
                            size="small"
                            onClick={() => handleSuggestionClick(suggestion)}
                            sx={{
                              cursor: 'pointer',
                              '&:hover': {
                                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                              },
                            }}
                          />
                        ))}
                      </Box>
                    )}
                  </Box>
                </ListItem>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {isTyping && (
            <Fade in={isTyping}>
              <ListItem>
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                    🤖
                  </Avatar>
                </ListItemAvatar>
                <Box
                  sx={{
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: 2,
                    p: 2,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AutoAwesome fontSize="small" color="primary" />
                    <Typography variant="body2" color="text.secondary">
                      AI is thinking...
                    </Typography>
                  </Box>
                  <LinearProgress sx={{ mt: 1, width: 100 }} />
                </Box>
              </ListItem>
            </Fade>
          )}
          
          <div ref={messagesEndRef} />
        </List>
      </Box>

      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            size="small"
            placeholder="Ask me anything about your media server..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            disabled={isTyping}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 3,
              },
            }}
          />
          <Tooltip title={isListening ? 'Stop listening' : 'Voice input'}>
            <IconButton
              onClick={toggleListening}
              color={isListening ? 'error' : 'default'}
              sx={{
                backgroundColor: isListening ? 'rgba(255, 51, 102, 0.1)' : 'rgba(255, 255, 255, 0.05)',
              }}
            >
              {isListening ? <Stop /> : <Mic />}
            </IconButton>
          </Tooltip>
          <IconButton
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            color="primary"
            sx={{
              backgroundColor: 'rgba(0, 255, 136, 0.1)',
              '&:hover': {
                backgroundColor: 'rgba(0, 255, 136, 0.2)',
              },
            }}
          >
            <Send />
          </IconButton>
        </Box>

        {!messages.length && (
          <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {quickPrompts.map((prompt, index) => (
              <Chip
                key={index}
                label={prompt}
                size="small"
                onClick={() => {
                  setInput(prompt)
                  handleSend()
                }}
                sx={{
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                  },
                }}
              />
            ))}
          </Box>
        )}
      </Box>
    </Card>
  )
}

export default AIAssistant