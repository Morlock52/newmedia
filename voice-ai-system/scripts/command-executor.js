// Command Executor Module
// Executes voice commands based on detected intents

class CommandExecutor {
    constructor(options = {}) {
        this.options = {
            confirmBeforeExecute: false,
            simulationMode: false,
            ...options
        };

        this.commandHandlers = {};
        this.activeTimers = new Map();
        this.notes = [];
        this.reminders = [];
        
        this.initializeHandlers();
    }

    initializeHandlers() {
        // Weather command handler
        this.commandHandlers.weather = async (entities) => {
            const location = entities.location || 'your location';
            
            if (this.options.simulationMode) {
                return {
                    success: true,
                    response: `The weather in ${location} is sunny with a temperature of 72°F (22°C). There's a 10% chance of rain.`,
                    data: {
                        location: location,
                        temperature: 72,
                        condition: 'sunny',
                        humidity: 45,
                        windSpeed: 8
                    }
                };
            }

            // In real implementation, this would call a weather API
            try {
                const response = await this.fetchWeatherData(location);
                return {
                    success: true,
                    response: response.message,
                    data: response.data
                };
            } catch (error) {
                return {
                    success: false,
                    response: `Sorry, I couldn't get the weather for ${location}.`,
                    error: error.message
                };
            }
        };

        // Time command handler
        this.commandHandlers.time = async (entities) => {
            const location = entities.location;
            const now = new Date();
            
            if (!location) {
                const timeString = now.toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
                
                return {
                    success: true,
                    response: `The current time is ${timeString}.`,
                    data: {
                        time: timeString,
                        timestamp: now.toISOString()
                    }
                };
            }

            // Handle time in different locations
            try {
                const timeData = await this.getTimeInLocation(location);
                return {
                    success: true,
                    response: `The current time in ${location} is ${timeData.time}.`,
                    data: timeData
                };
            } catch (error) {
                return {
                    success: false,
                    response: `Sorry, I couldn't get the time for ${location}.`,
                    error: error.message
                };
            }
        };

        // Timer command handler
        this.commandHandlers.timer = async (entities) => {
            const durations = entities.duration;
            
            if (!durations || durations.length === 0) {
                return {
                    success: false,
                    response: "I didn't understand the timer duration. Please specify a time like '5 minutes' or '1 hour'."
                };
            }

            const totalMs = this.calculateDurationMs(durations);
            const timerId = Date.now().toString();
            
            const timer = {
                id: timerId,
                duration: totalMs,
                startTime: Date.now(),
                endTime: Date.now() + totalMs,
                description: this.formatDuration(durations)
            };

            this.activeTimers.set(timerId, timer);

            // Set actual timer
            setTimeout(() => {
                this.handleTimerComplete(timerId);
            }, totalMs);

            return {
                success: true,
                response: `Timer set for ${timer.description}. I'll notify you when it's done.`,
                data: {
                    timerId: timerId,
                    duration: timer.description,
                    endTime: new Date(timer.endTime).toLocaleTimeString()
                }
            };
        };

        // Alarm command handler
        this.commandHandlers.alarm = async (entities) => {
            const timeStr = entities.time;
            
            if (!timeStr) {
                return {
                    success: false,
                    response: "Please specify a time for the alarm."
                };
            }

            const alarmTime = this.parseAlarmTime(timeStr);
            if (!alarmTime) {
                return {
                    success: false,
                    response: "I couldn't understand the time. Please try again with a format like '7:30 AM' or '3 PM'."
                };
            }

            return {
                success: true,
                response: `Alarm set for ${alarmTime.toLocaleTimeString()}.`,
                data: {
                    alarmTime: alarmTime.toISOString(),
                    displayTime: alarmTime.toLocaleTimeString()
                }
            };
        };

        // Search command handler
        this.commandHandlers.search = async (entities) => {
            const query = entities.query;
            
            if (!query) {
                return {
                    success: false,
                    response: "What would you like me to search for?"
                };
            }

            if (this.options.simulationMode) {
                return {
                    success: true,
                    response: `Here are the search results for "${query}".`,
                    data: {
                        query: query,
                        results: [
                            { title: `Result 1 for ${query}`, url: '#', snippet: 'Sample result snippet...' },
                            { title: `Result 2 for ${query}`, url: '#', snippet: 'Another result snippet...' },
                            { title: `Result 3 for ${query}`, url: '#', snippet: 'Third result snippet...' }
                        ]
                    }
                };
            }

            // In real implementation, this would use a search API
            return {
                success: true,
                response: `Searching for "${query}"...`,
                action: 'search',
                data: { query: query }
            };
        };

        // Calculate command handler
        this.commandHandlers.calculate = async (entities) => {
            const expression = entities.expression;
            
            if (!expression) {
                return {
                    success: false,
                    response: "Please provide a mathematical expression to calculate."
                };
            }

            try {
                // Safe evaluation of mathematical expressions
                const result = this.evaluateMathExpression(expression);
                
                return {
                    success: true,
                    response: `${expression} equals ${result}.`,
                    data: {
                        expression: expression,
                        result: result
                    }
                };
            } catch (error) {
                return {
                    success: false,
                    response: "I couldn't calculate that expression. Please try again with a simpler format.",
                    error: error.message
                };
            }
        };

        // Translate command handler
        this.commandHandlers.translate = async (entities) => {
            const text = entities.text;
            const targetLang = entities.targetLanguage;
            
            if (!text || !targetLang) {
                return {
                    success: false,
                    response: "Please specify what to translate and the target language."
                };
            }

            if (this.options.simulationMode) {
                const translations = {
                    spanish: 'Hola',
                    french: 'Bonjour',
                    german: 'Hallo',
                    italian: 'Ciao',
                    portuguese: 'Olá',
                    japanese: 'こんにちは',
                    chinese: '你好',
                    korean: '안녕하세요'
                };

                const langKey = targetLang.toLowerCase();
                const translation = translations[langKey] || `[Translation to ${targetLang}]`;

                return {
                    success: true,
                    response: `"${text}" in ${targetLang} is "${translation}".`,
                    data: {
                        originalText: text,
                        translatedText: translation,
                        sourceLanguage: 'english',
                        targetLanguage: targetLang
                    }
                };
            }

            // In real implementation, this would use a translation API
            return {
                success: true,
                response: `Translating "${text}" to ${targetLang}...`,
                action: 'translate',
                data: { text: text, targetLanguage: targetLang }
            };
        };

        // Note command handler
        this.commandHandlers.note = async (entities) => {
            const content = entities.content;
            
            if (!content) {
                return {
                    success: false,
                    response: "What would you like me to note down?"
                };
            }

            const note = {
                id: Date.now().toString(),
                content: content,
                timestamp: new Date().toISOString(),
                tags: this.extractTags(content)
            };

            this.notes.push(note);

            return {
                success: true,
                response: `I've saved your note: "${content}".`,
                data: note
            };
        };

        // Reminder command handler
        this.commandHandlers.reminder = async (entities) => {
            const task = entities.task;
            const time = entities.time;
            
            if (!task || !time) {
                return {
                    success: false,
                    response: "Please specify what to remind you about and when."
                };
            }

            const reminderTime = this.parseReminderTime(time);
            if (!reminderTime) {
                return {
                    success: false,
                    response: "I couldn't understand the time for the reminder."
                };
            }

            const reminder = {
                id: Date.now().toString(),
                task: task,
                time: reminderTime.toISOString(),
                displayTime: reminderTime.toLocaleString(),
                created: new Date().toISOString()
            };

            this.reminders.push(reminder);

            return {
                success: true,
                response: `I'll remind you to ${task} at ${reminder.displayTime}.`,
                data: reminder
            };
        };

        // Volume command handler
        this.commandHandlers.volume = async (entities) => {
            const level = entities.level;
            const direction = entities.direction;
            
            if (level) {
                // Specific volume level
                const volumeLevel = parseInt(level);
                if (isNaN(volumeLevel) || volumeLevel < 0 || volumeLevel > 100) {
                    return {
                        success: false,
                        response: "Please specify a volume level between 0 and 100."
                    };
                }

                return {
                    success: true,
                    response: `Volume set to ${volumeLevel}%.`,
                    action: 'setVolume',
                    data: { level: volumeLevel }
                };
            } else if (direction) {
                // Relative volume change
                const change = direction === 'up' || direction === 'louder' ? 10 : -10;
                
                return {
                    success: true,
                    response: `Volume turned ${direction}.`,
                    action: 'adjustVolume',
                    data: { change: change }
                };
            }

            return {
                success: false,
                response: "Please specify how to adjust the volume."
            };
        };

        // Music command handler
        this.commandHandlers.music = async (entities) => {
            const song = entities.song;
            const artist = entities.artist;
            const action = entities.action;
            
            if (action) {
                // Music control actions
                const actions = {
                    'pause': 'Music paused.',
                    'stop': 'Music stopped.',
                    'next': 'Playing next track.',
                    'previous': 'Playing previous track.',
                    'skip': 'Skipping to next track.'
                };

                return {
                    success: true,
                    response: actions[action] || `Performing ${action} action.`,
                    action: action,
                    data: { action: action }
                };
            }

            if (song || artist) {
                const description = song && artist ? `${song} by ${artist}` : song || `music by ${artist}`;
                
                return {
                    success: true,
                    response: `Playing ${description}.`,
                    action: 'play',
                    data: { song: song, artist: artist }
                };
            }

            return {
                success: true,
                response: "Playing music.",
                action: 'play',
                data: { type: 'general' }
            };
        };

        // Help command handler
        this.commandHandlers.help = async (entities) => {
            const topic = entities.topic;
            
            if (topic) {
                // Specific help topic
                const helpText = this.getHelpForTopic(topic);
                return {
                    success: true,
                    response: helpText,
                    data: { topic: topic }
                };
            }

            // General help
            const commands = Object.keys(this.commandHandlers).sort();
            const commandList = commands.join(', ');
            
            return {
                success: true,
                response: `I can help you with: ${commandList}. Say "help" followed by a command to learn more.`,
                data: { availableCommands: commands }
            };
        };
    }

    async execute(intent, entities = {}) {
        if (!this.commandHandlers[intent]) {
            return {
                success: false,
                response: `I don't know how to handle "${intent}" commands yet.`,
                error: 'Unknown intent'
            };
        }

        try {
            const result = await this.commandHandlers[intent](entities);
            
            // Log execution
            console.log(`Executed command: ${intent}`, { entities, result });
            
            return result;
        } catch (error) {
            console.error(`Error executing command ${intent}:`, error);
            
            return {
                success: false,
                response: `An error occurred while executing the ${intent} command.`,
                error: error.message
            };
        }
    }

    // Helper methods
    calculateDurationMs(durations) {
        const unitToMs = {
            second: 1000,
            minute: 60000,
            hour: 3600000,
            day: 86400000
        };

        let totalMs = 0;
        for (const duration of durations) {
            totalMs += duration.value * (unitToMs[duration.unit] || 0);
        }

        return totalMs;
    }

    formatDuration(durations) {
        return durations.map(d => `${d.value} ${d.unit}${d.value > 1 ? 's' : ''}`).join(' and ');
    }

    parseAlarmTime(timeStr) {
        // Simple time parsing - in production, use a proper date parsing library
        const now = new Date();
        const timeParts = timeStr.match(/(\d{1,2}):?(\d{2})?\s*(AM|PM)?/i);
        
        if (!timeParts) return null;

        let hours = parseInt(timeParts[1]);
        const minutes = parseInt(timeParts[2] || '0');
        const period = timeParts[3];

        if (period) {
            if (period.toUpperCase() === 'PM' && hours !== 12) {
                hours += 12;
            } else if (period.toUpperCase() === 'AM' && hours === 12) {
                hours = 0;
            }
        }

        const alarmTime = new Date(now);
        alarmTime.setHours(hours, minutes, 0, 0);

        // If the time has already passed today, set it for tomorrow
        if (alarmTime <= now) {
            alarmTime.setDate(alarmTime.getDate() + 1);
        }

        return alarmTime;
    }

    parseReminderTime(timeStr) {
        // Handle relative times like "in 5 minutes" or "tomorrow"
        const now = new Date();
        
        // Check for relative time
        const relativeMatch = timeStr.match(/in (\d+) (minute|hour|day)s?/i);
        if (relativeMatch) {
            const value = parseInt(relativeMatch[1]);
            const unit = relativeMatch[2].toLowerCase();
            
            const ms = this.calculateDurationMs([{ value, unit }]);
            return new Date(now.getTime() + ms);
        }

        // Check for specific times
        if (timeStr.toLowerCase().includes('tomorrow')) {
            const tomorrow = new Date(now);
            tomorrow.setDate(tomorrow.getDate() + 1);
            tomorrow.setHours(9, 0, 0, 0); // Default to 9 AM
            return tomorrow;
        }

        // Try parsing as absolute time
        return this.parseAlarmTime(timeStr);
    }

    evaluateMathExpression(expression) {
        // Safe math evaluation - only allows numbers and basic operators
        const sanitized = expression.replace(/[^0-9+\-*/().\s]/g, '');
        
        // Simple evaluation for basic operations
        try {
            // This is a simplified evaluator - in production, use a proper math parser
            const result = Function('"use strict"; return (' + sanitized + ')')();
            return Math.round(result * 100) / 100; // Round to 2 decimal places
        } catch (error) {
            throw new Error('Invalid mathematical expression');
        }
    }

    extractTags(content) {
        // Extract hashtags or keywords
        const tags = [];
        const hashtagMatch = content.match(/#\w+/g);
        if (hashtagMatch) {
            tags.push(...hashtagMatch.map(tag => tag.substring(1)));
        }
        return tags;
    }

    handleTimerComplete(timerId) {
        const timer = this.activeTimers.get(timerId);
        if (!timer) return;

        // Notify about timer completion
        console.log(`Timer ${timerId} completed:`, timer.description);
        
        // Remove from active timers
        this.activeTimers.delete(timerId);

        // In a real app, this would trigger a notification
        if (this.options.onTimerComplete) {
            this.options.onTimerComplete(timer);
        }
    }

    async fetchWeatherData(location) {
        // Simulated weather API call
        if (this.options.simulationMode) {
            return {
                message: `The weather in ${location} is partly cloudy with a temperature of 68°F.`,
                data: {
                    temperature: 68,
                    condition: 'partly cloudy',
                    humidity: 60,
                    windSpeed: 12
                }
            };
        }

        // In production, this would call a real weather API
        throw new Error('Weather API not configured');
    }

    async getTimeInLocation(location) {
        // Simulated timezone lookup
        const timezones = {
            'london': 'Europe/London',
            'paris': 'Europe/Paris',
            'tokyo': 'Asia/Tokyo',
            'new york': 'America/New_York',
            'los angeles': 'America/Los_Angeles'
        };

        const timezone = timezones[location.toLowerCase()];
        if (!timezone) {
            throw new Error('Unknown location');
        }

        const time = new Date().toLocaleTimeString('en-US', {
            timeZone: timezone,
            hour: '2-digit',
            minute: '2-digit'
        });

        return {
            time: time,
            timezone: timezone,
            location: location
        };
    }

    getHelpForTopic(topic) {
        const helpTexts = {
            timer: "To set a timer, say 'Set a timer for X minutes' or 'Timer 30 seconds'. You can use seconds, minutes, or hours.",
            alarm: "To set an alarm, say 'Set an alarm for 7 AM' or 'Wake me up at 6:30 PM'.",
            weather: "Ask about the weather by saying 'What's the weather?' or 'Weather in London'.",
            translate: "To translate text, say 'Translate hello to Spanish' or 'How do you say goodbye in French?'",
            calculate: "For calculations, say 'What's 5 plus 3?' or 'Calculate 100 divided by 4'.",
            search: "To search, say 'Search for pizza recipes' or 'Google climate change'.",
            note: "To take a note, say 'Take a note: buy milk' or 'Remember to call mom'.",
            music: "Control music by saying 'Play some music', 'Next song', 'Pause', or 'Play Bohemian Rhapsody by Queen'."
        };

        return helpTexts[topic.toLowerCase()] || `I don't have specific help for "${topic}". Try asking 'help' to see all available commands.`;
    }

    // Get active timers
    getActiveTimers() {
        return Array.from(this.activeTimers.values());
    }

    // Get all notes
    getNotes() {
        return this.notes;
    }

    // Get all reminders
    getReminders() {
        return this.reminders.filter(r => new Date(r.time) > new Date());
    }

    // Clear all data
    clearAll() {
        this.activeTimers.clear();
        this.notes = [];
        this.reminders = [];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CommandExecutor;
}