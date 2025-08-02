// NLP Processor Module
// Natural Language Processing for command detection and intent recognition

class NLPProcessor {
    constructor(options = {}) {
        this.options = {
            confidenceThreshold: 0.7,
            fuzzyMatchThreshold: 0.8,
            ...options
        };

        // Intent patterns
        this.intents = {
            weather: {
                patterns: [
                    /what(?:'s| is) the weather/i,
                    /how(?:'s| is) the weather/i,
                    /weather forecast/i,
                    /temperature today/i,
                    /is it (?:going to )?rain/i,
                    /will it (?:be )?sunny/i,
                    /weather in (.+)/i
                ],
                entities: ['location', 'time'],
                examples: ["What's the weather?", "Weather forecast for tomorrow", "Is it going to rain?"]
            },
            time: {
                patterns: [
                    /what time is it/i,
                    /what(?:'s| is) the time/i,
                    /current time/i,
                    /time in (.+)/i,
                    /what time is it in (.+)/i
                ],
                entities: ['location'],
                examples: ["What time is it?", "Current time in Tokyo", "What's the time?"]
            },
            timer: {
                patterns: [
                    /set (?:a )?timer for (.+)/i,
                    /start (?:a )?timer for (.+)/i,
                    /timer (.+)/i,
                    /remind me in (.+)/i,
                    /countdown (.+)/i
                ],
                entities: ['duration'],
                examples: ["Set a timer for 5 minutes", "Start a timer for 1 hour", "Remind me in 30 seconds"]
            },
            alarm: {
                patterns: [
                    /set (?:an )?alarm (?:for|at) (.+)/i,
                    /wake me (?:up )?at (.+)/i,
                    /alarm at (.+)/i,
                    /create (?:an )?alarm for (.+)/i
                ],
                entities: ['time'],
                examples: ["Set an alarm for 7 AM", "Wake me up at 6:30", "Alarm at 8 PM"]
            },
            search: {
                patterns: [
                    /search (?:for )?(.+)/i,
                    /google (.+)/i,
                    /look up (.+)/i,
                    /find (?:information (?:about|on) )?(.+)/i,
                    /what is (.+)/i,
                    /who is (.+)/i,
                    /define (.+)/i
                ],
                entities: ['query'],
                examples: ["Search for pizza recipes", "Google climate change", "What is machine learning?"]
            },
            calculate: {
                patterns: [
                    /what(?:'s| is) (.+) (?:plus|minus|times|divided by|multiplied by) (.+)/i,
                    /calculate (.+)/i,
                    /(.+) (?:plus|minus|times|divided by|multiplied by|\+|\-|\*|\/) (.+)/i,
                    /solve (.+)/i
                ],
                entities: ['expression', 'operand1', 'operand2', 'operator'],
                examples: ["What's 5 plus 3?", "Calculate 100 divided by 4", "25 times 4"]
            },
            translate: {
                patterns: [
                    /translate (.+) to (.+)/i,
                    /how do you say (.+) in (.+)/i,
                    /(.+) in (.+)/i,
                    /what(?:'s| is) (.+) in (.+)/i
                ],
                entities: ['text', 'targetLanguage'],
                examples: ["Translate hello to Spanish", "How do you say goodbye in French?", "Thank you in Japanese"]
            },
            note: {
                patterns: [
                    /take (?:a )?note(?:\:)? (.+)/i,
                    /note(?:\:)? (.+)/i,
                    /remember(?:\:)? (.+)/i,
                    /save(?:\:)? (.+)/i,
                    /write down(?:\:)? (.+)/i
                ],
                entities: ['content'],
                examples: ["Take a note: buy milk", "Remember to call mom", "Note: meeting at 3 PM"]
            },
            reminder: {
                patterns: [
                    /remind me (?:to )?(.+) (?:at|in|on) (.+)/i,
                    /reminder(?:\:)? (.+) (?:at|in|on) (.+)/i,
                    /set (?:a )?reminder (?:to )?(.+) (?:at|in|on) (.+)/i
                ],
                entities: ['task', 'time'],
                examples: ["Remind me to take medicine at 8 PM", "Set a reminder to call John tomorrow"]
            },
            volume: {
                patterns: [
                    /(?:set )?volume (?:to )?(.+)/i,
                    /turn (?:the )?volume (up|down)/i,
                    /(?:make it |turn it )?(louder|quieter)/i,
                    /mute/i,
                    /unmute/i
                ],
                entities: ['level', 'direction'],
                examples: ["Volume to 50%", "Turn volume up", "Make it louder", "Mute"]
            },
            music: {
                patterns: [
                    /play (.+)/i,
                    /play (?:some )?music/i,
                    /(?:play |put on )(.+) by (.+)/i,
                    /pause (?:the )?(?:music|song)/i,
                    /stop (?:the )?(?:music|song)/i,
                    /next (?:song|track)/i,
                    /previous (?:song|track)/i,
                    /skip/i
                ],
                entities: ['song', 'artist', 'action'],
                examples: ["Play some music", "Play Bohemian Rhapsody by Queen", "Next song", "Pause"]
            },
            navigation: {
                patterns: [
                    /(?:navigate|directions|route) to (.+)/i,
                    /how (?:do I |to )get to (.+)/i,
                    /take me to (.+)/i,
                    /where is (.+)/i,
                    /find (.+) near(?:by| me)/i
                ],
                entities: ['destination', 'location'],
                examples: ["Navigate to Times Square", "How do I get to the airport?", "Where is the nearest coffee shop?"]
            },
            call: {
                patterns: [
                    /call (.+)/i,
                    /dial (.+)/i,
                    /phone (.+)/i,
                    /ring (.+)/i
                ],
                entities: ['contact'],
                examples: ["Call mom", "Dial 555-1234", "Phone John Smith"]
            },
            message: {
                patterns: [
                    /(?:send |text )(?:a )?message to (.+) (?:saying |that says )?(.+)/i,
                    /text (.+) (?:saying |that says )?(.+)/i,
                    /message (.+) (?:saying |that says )?(.+)/i,
                    /tell (.+) (?:that )?(.+)/i
                ],
                entities: ['recipient', 'content'],
                examples: ["Send a message to John saying I'll be late", "Text mom that I'm on my way"]
            },
            email: {
                patterns: [
                    /(?:send |compose )(?:an )?email to (.+) (?:about |with subject )?(.+)/i,
                    /email (.+) (?:about |with subject )?(.+)/i,
                    /(?:write |draft) (?:an )?email to (.+)/i
                ],
                entities: ['recipient', 'subject', 'content'],
                examples: ["Send an email to boss about meeting", "Email John with subject Project Update"]
            },
            help: {
                patterns: [
                    /help/i,
                    /what can you do/i,
                    /(?:show |list )commands/i,
                    /(?:show |list )(?:available )?features/i,
                    /how (?:do I |to )(.+)/i
                ],
                entities: ['topic'],
                examples: ["Help", "What can you do?", "Show commands", "How do I set a timer?"]
            }
        };

        // Entity extractors
        this.entityExtractors = {
            location: (text) => {
                const locationPattern = /(?:in|at|from|to) ([A-Za-z\s]+?)(?:\.|,|$)/i;
                const match = text.match(locationPattern);
                return match ? match[1].trim() : null;
            },
            time: (text) => {
                const timePatterns = [
                    /(\d{1,2}:\d{2}(?:\s?[AP]M)?)/i,
                    /(\d{1,2}\s?[AP]M)/i,
                    /(morning|afternoon|evening|night|noon|midnight)/i,
                    /(today|tomorrow|yesterday)/i,
                    /(monday|tuesday|wednesday|thursday|friday|saturday|sunday)/i
                ];
                
                for (const pattern of timePatterns) {
                    const match = text.match(pattern);
                    if (match) return match[1];
                }
                return null;
            },
            duration: (text) => {
                const durationPattern = /(\d+)\s*(second|minute|hour|day|week|month|year)s?/gi;
                const matches = text.matchAll(durationPattern);
                const durations = [];
                
                for (const match of matches) {
                    durations.push({
                        value: parseInt(match[1]),
                        unit: match[2].toLowerCase()
                    });
                }
                
                return durations.length > 0 ? durations : null;
            },
            number: (text) => {
                const numberPattern = /\b(\d+(?:\.\d+)?)\b/g;
                const matches = text.match(numberPattern);
                return matches ? matches.map(n => parseFloat(n)) : null;
            },
            query: (text) => {
                // Remove common search prefixes
                const cleanedText = text.replace(/^(search for|google|look up|find|what is|who is|define)\s*/i, '');
                return cleanedText.trim();
            },
            expression: (text) => {
                const mathPattern = /[\d\s\+\-\*\/\(\)\.]+/g;
                const match = text.match(mathPattern);
                return match ? match[0].trim() : null;
            }
        };

        // Sentiment analysis
        this.sentimentWords = {
            positive: ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy', 'love', 'best', 'perfect'],
            negative: ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'sad', 'angry', 'disappointed', 'frustrating']
        };
    }

    processText(text) {
        const result = {
            originalText: text,
            normalizedText: this.normalizeText(text),
            intent: null,
            entities: {},
            confidence: 0,
            alternatives: [],
            sentiment: this.analyzeSentiment(text),
            language: this.detectLanguage(text)
        };

        // Try to match intents
        const matches = this.matchIntents(result.normalizedText);
        
        if (matches.length > 0) {
            // Use the best match
            result.intent = matches[0].intent;
            result.confidence = matches[0].confidence;
            result.entities = matches[0].entities;
            
            // Add alternatives if confidence is below threshold
            if (matches.length > 1) {
                result.alternatives = matches.slice(1, 4).map(m => ({
                    intent: m.intent,
                    confidence: m.confidence
                }));
            }
        }

        return result;
    }

    normalizeText(text) {
        return text
            .toLowerCase()
            .replace(/[.,!?;:'"]/g, '')
            .replace(/\s+/g, ' ')
            .trim();
    }

    matchIntents(text) {
        const matches = [];

        for (const [intentName, intentData] of Object.entries(this.intents)) {
            for (const pattern of intentData.patterns) {
                const match = text.match(pattern);
                if (match) {
                    const entities = this.extractEntities(text, match, intentData.entities);
                    matches.push({
                        intent: intentName,
                        confidence: 1.0, // Exact regex match
                        entities: entities,
                        matchType: 'exact'
                    });
                }
            }
        }

        // If no exact matches, try fuzzy matching
        if (matches.length === 0) {
            const fuzzyMatches = this.fuzzyMatchIntents(text);
            matches.push(...fuzzyMatches);
        }

        // Sort by confidence
        matches.sort((a, b) => b.confidence - a.confidence);

        return matches;
    }

    fuzzyMatchIntents(text) {
        const matches = [];
        const textWords = text.split(' ');

        for (const [intentName, intentData] of Object.entries(this.intents)) {
            // Check examples for similarity
            for (const example of intentData.examples || []) {
                const similarity = this.calculateSimilarity(text, example.toLowerCase());
                if (similarity >= this.options.fuzzyMatchThreshold) {
                    matches.push({
                        intent: intentName,
                        confidence: similarity,
                        entities: this.extractEntities(text, null, intentData.entities),
                        matchType: 'fuzzy'
                    });
                    break;
                }
            }
        }

        return matches;
    }

    calculateSimilarity(text1, text2) {
        // Simple word-based similarity
        const words1 = text1.split(' ');
        const words2 = text2.split(' ');
        const allWords = new Set([...words1, ...words2]);
        
        let commonWords = 0;
        for (const word of words1) {
            if (words2.includes(word)) {
                commonWords++;
            }
        }

        return commonWords / Math.max(words1.length, words2.length);
    }

    extractEntities(text, regexMatch, entityTypes) {
        const entities = {};

        for (const entityType of entityTypes) {
            if (this.entityExtractors[entityType]) {
                const value = this.entityExtractors[entityType](text);
                if (value) {
                    entities[entityType] = value;
                }
            }
        }

        // Extract from regex groups if available
        if (regexMatch && regexMatch.length > 1) {
            for (let i = 1; i < regexMatch.length; i++) {
                if (regexMatch[i] && entityTypes[i - 1]) {
                    entities[entityTypes[i - 1]] = regexMatch[i];
                }
            }
        }

        return entities;
    }

    analyzeSentiment(text) {
        const words = text.toLowerCase().split(' ');
        let positiveCount = 0;
        let negativeCount = 0;

        for (const word of words) {
            if (this.sentimentWords.positive.includes(word)) {
                positiveCount++;
            } else if (this.sentimentWords.negative.includes(word)) {
                negativeCount++;
            }
        }

        const total = positiveCount + negativeCount;
        if (total === 0) {
            return { sentiment: 'neutral', confidence: 1.0 };
        }

        const positiveRatio = positiveCount / total;
        if (positiveRatio > 0.6) {
            return { sentiment: 'positive', confidence: positiveRatio };
        } else if (positiveRatio < 0.4) {
            return { sentiment: 'negative', confidence: 1 - positiveRatio };
        } else {
            return { sentiment: 'neutral', confidence: 0.5 };
        }
    }

    detectLanguage(text) {
        // Simple language detection based on character sets
        const scripts = {
            latin: /[a-zA-Z]/,
            cyrillic: /[\u0400-\u04FF]/,
            arabic: /[\u0600-\u06FF]/,
            chinese: /[\u4E00-\u9FFF]/,
            japanese: /[\u3040-\u309F\u30A0-\u30FF]/,
            korean: /[\uAC00-\uD7AF]/,
            devanagari: /[\u0900-\u097F]/,
            thai: /[\u0E00-\u0E7F]/
        };

        const detected = [];
        for (const [script, pattern] of Object.entries(scripts)) {
            if (pattern.test(text)) {
                detected.push(script);
            }
        }

        // Map scripts to languages (simplified)
        const scriptToLang = {
            latin: 'en',
            cyrillic: 'ru',
            arabic: 'ar',
            chinese: 'zh',
            japanese: 'ja',
            korean: 'ko',
            devanagari: 'hi',
            thai: 'th'
        };

        return detected.length > 0 ? scriptToLang[detected[0]] : 'en';
    }

    // Add custom intent
    addIntent(name, patterns, entities = [], examples = []) {
        this.intents[name] = {
            patterns: patterns.map(p => typeof p === 'string' ? new RegExp(p, 'i') : p),
            entities: entities,
            examples: examples
        };
    }

    // Add entity extractor
    addEntityExtractor(name, extractorFn) {
        this.entityExtractors[name] = extractorFn;
    }

    // Get all available intents
    getAvailableIntents() {
        return Object.keys(this.intents);
    }

    // Get intent details
    getIntentDetails(intentName) {
        return this.intents[intentName] || null;
    }

    // Validate command confidence
    isConfidentMatch(result) {
        return result.confidence >= this.options.confidenceThreshold;
    }

    // Generate response suggestions
    generateSuggestions(intent) {
        const intentData = this.intents[intent];
        if (!intentData || !intentData.examples) {
            return [];
        }

        return intentData.examples.slice(0, 3);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NLPProcessor;
}