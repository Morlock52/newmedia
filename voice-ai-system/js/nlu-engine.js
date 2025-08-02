class NaturalLanguageUnderstanding {
    constructor() {
        this.intents = this.initializeIntents();
        this.entities = this.initializeEntities();
        this.synonyms = this.initializeSynonyms();
        this.confidence_threshold = 0.6;
    }

    initializeIntents() {
        return {
            PLAY: {
                patterns: [
                    /^(play|start|begin|resume)(\s+.*)?$/i,
                    /^(hit\s+)?play(\s+button)?$/i,
                    /^(let's\s+)?(start\s+)?(playing|watching)$/i,
                    /^(can\s+you\s+)?(please\s+)?play(\s+this|\s+it|\s+video|\s+media)?$/i
                ],
                examples: ['play', 'start playing', 'play video', 'hit play', 'resume', 'begin']
            },
            PAUSE: {
                patterns: [
                    /^(pause|stop|halt)(\s+.*)?$/i,
                    /^(hit\s+)?pause(\s+button)?$/i,
                    /^(can\s+you\s+)?(please\s+)?(pause|stop)(\s+this|\s+it|\s+video|\s+media)?$/i,
                    /^(hold\s+on|wait)$/i
                ],
                examples: ['pause', 'stop', 'halt', 'hit pause', 'hold on', 'wait']
            },
            VOLUME_UP: {
                patterns: [
                    /^(volume\s+up|increase\s+volume|louder|turn\s+up)$/i,
                    /^(make\s+it\s+)?(louder|higher)$/i,
                    /^(volume\s+)?(up|increase)(\s+volume)?$/i,
                    /^(can\s+you\s+)?(please\s+)?(turn|make)(\s+it|\s+the\s+volume)?\s+(up|louder|higher)$/i
                ],
                examples: ['volume up', 'louder', 'turn up', 'increase volume', 'make it louder']
            },
            VOLUME_DOWN: {
                patterns: [
                    /^(volume\s+down|decrease\s+volume|quieter|turn\s+down)$/i,
                    /^(make\s+it\s+)?(quieter|lower|softer)$/i,
                    /^(volume\s+)?(down|decrease)(\s+volume)?$/i,
                    /^(can\s+you\s+)?(please\s+)?(turn|make)(\s+it|\s+the\s+volume)?\s+(down|quieter|lower|softer)$/i
                ],
                examples: ['volume down', 'quieter', 'turn down', 'decrease volume', 'make it softer']
            },
            MUTE: {
                patterns: [
                    /^(mute|silence|turn\s+off\s+sound)$/i,
                    /^(make\s+it\s+)?(silent|quiet)$/i,
                    /^(turn\s+off|disable)(\s+the)?\s+(audio|sound|volume)$/i,
                    /^(can\s+you\s+)?(please\s+)?mute(\s+this|\s+it)?$/i
                ],
                examples: ['mute', 'silence', 'turn off sound', 'make it silent', 'quiet']
            },
            UNMUTE: {
                patterns: [
                    /^(unmute|turn\s+on\s+sound|restore\s+audio)$/i,
                    /^(turn\s+on|enable)(\s+the)?\s+(audio|sound|volume)$/i,
                    /^(bring\s+back|restore)(\s+the)?\s+(sound|audio)$/i,
                    /^(can\s+you\s+)?(please\s+)?unmute(\s+this|\s+it)?$/i
                ],
                examples: ['unmute', 'turn on sound', 'restore audio', 'bring back sound']
            },
            SKIP_FORWARD: {
                patterns: [
                    /^(skip\s+forward|fast\s+forward|jump\s+ahead)(\s+\d+\s+(seconds?|minutes?))?$/i,
                    /^(forward|ahead)(\s+\d+\s+(seconds?|minutes?))?$/i,
                    /^(go\s+)?(forward|ahead)(\s+\d+\s+(seconds?|minutes?))?$/i,
                    /^(skip\s+)?(\d+\s+(seconds?|minutes?)\s+)?(forward|ahead)$/i
                ],
                examples: ['skip forward', 'fast forward', 'jump ahead', 'forward 10 seconds', 'ahead']
            },
            SKIP_BACKWARD: {
                patterns: [
                    /^(skip\s+back|rewind|jump\s+back|go\s+back)(\s+\d+\s+(seconds?|minutes?))?$/i,
                    /^(back|backward)(\s+\d+\s+(seconds?|minutes?))?$/i,
                    /^(skip\s+)?(\d+\s+(seconds?|minutes?)\s+)?(back|backward)$/i,
                    /^(replay|go\s+back)(\s+\d+\s+(seconds?|minutes?))?$/i
                ],
                examples: ['skip back', 'rewind', 'jump back', 'back 10 seconds', 'go back']
            },
            FULLSCREEN_ENTER: {
                patterns: [
                    /^(fullscreen|full\s+screen|expand)$/i,
                    /^(go\s+)?(fullscreen|full\s+screen)$/i,
                    /^(enter\s+|switch\s+to\s+)?(fullscreen|full\s+screen)(\s+mode)?$/i,
                    /^(make\s+it\s+)?(fullscreen|bigger|larger)$/i
                ],
                examples: ['fullscreen', 'full screen', 'expand', 'make it bigger', 'go fullscreen']
            },
            FULLSCREEN_EXIT: {
                patterns: [
                    /^(exit\s+fullscreen|leave\s+fullscreen|normal\s+screen)$/i,
                    /^(escape|exit)(\s+fullscreen|\s+full\s+screen)?$/i,
                    /^(go\s+back\s+to\s+)?(normal|regular)(\s+size|\s+screen|\s+mode)?$/i,
                    /^(make\s+it\s+)?(smaller|normal\s+size)$/i
                ],
                examples: ['exit fullscreen', 'escape', 'normal screen', 'make it smaller']
            },
            HELP: {
                patterns: [
                    /^(help|what\s+can\s+you\s+do|commands)$/i,
                    /^(show\s+)?(help|commands|options)$/i,
                    /^(what\s+)?(commands\s+)?(can\s+i\s+say|are\s+available)$/i,
                    /^(i\s+need\s+)?help$/i
                ],
                examples: ['help', 'what can you do', 'show commands', 'what commands can I say']
            },
            STATUS: {
                patterns: [
                    /^(status|what's\s+playing|current\s+status)$/i,
                    /^(what's\s+)?(happening|playing|the\s+status)$/i,
                    /^(show\s+)?(status|current\s+state)$/i,
                    /^(tell\s+me\s+)?(what's\s+)?(playing|happening)$/i
                ],
                examples: ['status', "what's playing", 'current status', 'what\'s happening']
            }
        };
    }

    initializeEntities() {
        return {
            TIME_DURATION: {
                patterns: [
                    /(\d+)\s*(seconds?|secs?)/i,
                    /(\d+)\s*(minutes?|mins?)/i,
                    /(\d+)\s*(hours?|hrs?)/i
                ],
                extractor: (match) => {
                    const value = parseInt(match[1]);
                    const unit = match[2].toLowerCase();
                    
                    if (unit.startsWith('sec')) return value;
                    if (unit.startsWith('min')) return value * 60;
                    if (unit.startsWith('hour') || unit.startsWith('hr')) return value * 3600;
                    
                    return value;
                }
            },
            VOLUME_LEVEL: {
                patterns: [
                    /(\d+)%?/i,
                    /(maximum|max|full)/i,
                    /(minimum|min|zero)/i,
                    /(half|fifty\s*percent)/i
                ],
                extractor: (match) => {
                    const text = match[0].toLowerCase();
                    
                    if (text.includes('max') || text.includes('full')) return 100;
                    if (text.includes('min') || text.includes('zero')) return 0;
                    if (text.includes('half') || text.includes('fifty')) return 50;
                    
                    const number = parseInt(match[1]);
                    return Math.min(100, Math.max(0, number));
                }
            }
        };
    }

    initializeSynonyms() {
        return {
            'start': ['play', 'begin', 'commence'],
            'stop': ['pause', 'halt', 'cease'],
            'louder': ['volume up', 'increase volume', 'turn up'],
            'quieter': ['volume down', 'decrease volume', 'turn down'],
            'silent': ['mute', 'quiet', 'no sound'],
            'forward': ['ahead', 'skip', 'fast forward'],
            'backward': ['back', 'rewind', 'reverse'],
            'bigger': ['fullscreen', 'expand', 'larger'],
            'smaller': ['normal', 'windowed', 'regular']
        };
    }

    processCommand(text) {
        if (!text || typeof text !== 'string') {
            return {
                success: false,
                error: 'Invalid input text',
                confidence: 0
            };
        }

        const cleanText = this.preprocessText(text);
        const result = this.parseIntent(cleanText);
        
        if (result.confidence >= this.confidence_threshold) {
            const entities = this.extractEntities(cleanText);
            return {
                success: true,
                intent: result.intent,
                entities: entities,
                confidence: result.confidence,
                originalText: text,
                processedText: cleanText
            };
        }

        return {
            success: false,
            error: 'Could not understand command',
            confidence: result.confidence,
            originalText: text,
            processedText: cleanText,
            suggestions: this.getSuggestions(cleanText)
        };
    }

    preprocessText(text) {
        return text
            .toLowerCase()
            .trim()
            .replace(/[^\w\s]/g, ' ')  // Remove punctuation
            .replace(/\s+/g, ' ')      // Normalize whitespace
            .trim();
    }

    parseIntent(text) {
        let bestMatch = {
            intent: null,
            confidence: 0
        };

        for (const [intentName, intentData] of Object.entries(this.intents)) {
            for (const pattern of intentData.patterns) {
                const match = text.match(pattern);
                
                if (match) {
                    // Calculate confidence based on match quality
                    const confidence = this.calculateConfidence(text, match, pattern);
                    
                    if (confidence > bestMatch.confidence) {
                        bestMatch = {
                            intent: intentName,
                            confidence: confidence,
                            matchedPattern: pattern.source,
                            matchedText: match[0]
                        };
                    }
                }
            }
        }

        // Also check for synonym matches
        const synonymMatch = this.checkSynonyms(text);
        if (synonymMatch.confidence > bestMatch.confidence) {
            bestMatch = synonymMatch;
        }

        return bestMatch;
    }

    calculateConfidence(text, match, pattern) {
        // Base confidence for any match
        let confidence = 0.6;
        
        // Boost confidence for exact matches
        if (match[0] === text) {
            confidence += 0.3;
        }
        
        // Boost confidence for longer matches
        const matchRatio = match[0].length / text.length;
        confidence += matchRatio * 0.2;
        
        // Boost confidence for matches at the beginning
        if (match.index === 0) {
            confidence += 0.1;
        }

        return Math.min(1.0, confidence);
    }

    checkSynonyms(text) {
        let bestMatch = { intent: null, confidence: 0 };

        for (const [key, synonyms] of Object.entries(this.synonyms)) {
            for (const synonym of synonyms) {
                if (text.includes(synonym)) {
                    // Find the intent that would match this synonym
                    const intentMatch = this.parseIntent(synonym);
                    if (intentMatch.confidence > bestMatch.confidence) {
                        bestMatch = {
                            intent: intentMatch.intent,
                            confidence: intentMatch.confidence * 0.8 // Slightly lower confidence for synonyms
                        };
                    }
                }
            }
        }

        return bestMatch;
    }

    extractEntities(text) {
        const entities = {};

        for (const [entityName, entityData] of Object.entries(this.entities)) {
            for (const pattern of entityData.patterns) {
                const match = text.match(pattern);
                if (match) {
                    entities[entityName] = {
                        value: entityData.extractor(match),
                        raw: match[0],
                        position: match.index
                    };
                }
            }
        }

        return entities;
    }

    getSuggestions(text) {
        const suggestions = [];
        const words = text.split(' ');
        
        // Find similar intents
        for (const [intentName, intentData] of Object.entries(this.intents)) {
            for (const example of intentData.examples) {
                const similarity = this.calculateSimilarity(text, example);
                if (similarity > 0.3) {
                    suggestions.push({
                        command: example,
                        intent: intentName,
                        similarity: similarity
                    });
                }
            }
        }

        // Sort by similarity and return top 3
        return suggestions
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 3)
            .map(s => s.command);
    }

    calculateSimilarity(str1, str2) {
        const words1 = str1.toLowerCase().split(' ');
        const words2 = str2.toLowerCase().split(' ');
        
        let matches = 0;
        for (const word1 of words1) {
            if (words2.includes(word1)) {
                matches++;
            }
        }
        
        return matches / Math.max(words1.length, words2.length);
    }

    setConfidenceThreshold(threshold) {
        this.confidence_threshold = Math.max(0, Math.min(1, threshold));
    }

    getAvailableCommands() {
        const commands = [];
        
        for (const [intentName, intentData] of Object.entries(this.intents)) {
            commands.push({
                intent: intentName,
                examples: intentData.examples,
                description: this.getIntentDescription(intentName)
            });
        }
        
        return commands;
    }

    getIntentDescription(intent) {
        const descriptions = {
            PLAY: 'Start or resume media playback',
            PAUSE: 'Pause media playback',
            VOLUME_UP: 'Increase the volume',
            VOLUME_DOWN: 'Decrease the volume',
            MUTE: 'Mute the audio',
            UNMUTE: 'Unmute the audio',
            SKIP_FORWARD: 'Skip forward in the media',
            SKIP_BACKWARD: 'Skip backward in the media',
            FULLSCREEN_ENTER: 'Enter fullscreen mode',
            FULLSCREEN_EXIT: 'Exit fullscreen mode',
            HELP: 'Show available commands',
            STATUS: 'Show current playback status'
        };
        
        return descriptions[intent] || 'Unknown command';
    }

    // Test the NLU engine with various inputs
    test() {
        const testCases = [
            'play video',
            'pause please',
            'volume up',
            'make it louder',
            'skip forward 10 seconds',
            'go back 30 seconds',
            'fullscreen',
            'mute',
            'help',
            'what can you do'
        ];

        const results = testCases.map(testCase => ({
            input: testCase,
            result: this.processCommand(testCase)
        }));

        return results;
    }
}