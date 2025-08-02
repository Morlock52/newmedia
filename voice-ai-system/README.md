# Voice AI Media Control System

A complete, working voice AI system that uses the Web Speech API and modern voice processing technologies for natural media control. The system includes speech recognition, synthesis, and natural language understanding capabilities.

## Features

### 🎤 Speech Recognition
- Real-time voice command recognition using Web Speech API
- Configurable confidence thresholds
- Support for multiple languages
- Continuous listening mode with interim results
- Automatic error recovery and retry mechanisms

### 🗣️ Speech Synthesis
- Natural voice feedback using Web Speech API
- Multiple voice options with gender detection
- Configurable pitch, rate, and volume
- Context-aware responses for different command types
- Support for multilingual speech synthesis

### 🧠 Natural Language Understanding
- Advanced intent recognition with pattern matching
- Entity extraction for time durations and volume levels
- Synonym support for natural command variations
- Confidence scoring and command suggestions
- Extensive command vocabulary covering media controls

### 🎵 Media Control
- Complete media player control (play, pause, stop)
- Volume management (up, down, mute, unmute)
- Seeking controls (skip forward/backward with time specifications)
- Fullscreen mode toggle
- Real-time status reporting and progress tracking

### 📊 Audio Visualization
- Real-time audio waveform visualization
- Frequency spectrum display with color gradients
- Volume level indicators
- Voice activity detection
- Multiple visualization modes (bars, waveform, spectrogram)

### 🎛️ Advanced Features
- Command history with success/failure tracking
- Configurable voice settings (pitch, rate, voice selection)
- Real-time status indicators with visual feedback
- Responsive design for desktop and mobile
- Keyboard shortcuts (Ctrl+Space to toggle listening)
- Comprehensive error handling and user feedback

## Supported Voice Commands

### Playback Control
- **Play**: "play", "start", "begin", "resume", "play video"
- **Pause**: "pause", "stop", "halt", "hold on", "wait"

### Volume Control
- **Volume Up**: "volume up", "louder", "turn up", "increase volume"
- **Volume Down**: "volume down", "quieter", "turn down", "decrease volume"
- **Mute**: "mute", "silence", "turn off sound", "make it silent"
- **Unmute**: "unmute", "turn on sound", "restore audio"

### Navigation
- **Skip Forward**: "skip forward", "fast forward", "jump ahead", "forward 10 seconds"
- **Skip Backward**: "skip back", "rewind", "jump back", "back 30 seconds"

### Display Control
- **Fullscreen**: "fullscreen", "full screen", "expand", "make it bigger"
- **Exit Fullscreen**: "exit fullscreen", "escape", "normal screen"

### System Commands
- **Help**: "help", "what can you do", "show commands"
- **Status**: "status", "what's playing", "current status"

## Technical Architecture

### Core Components

1. **SpeechRecognitionEngine** (`speech-recognition.js`)
   - Manages Web Speech API speech recognition
   - Handles microphone permissions and audio input
   - Provides confidence-based result filtering
   - Implements error recovery and reconnection logic

2. **SpeechSynthesisEngine** (`speech-synthesis.js`)
   - Manages Web Speech API speech synthesis
   - Supports multiple voices and languages
   - Provides contextual response generation
   - Handles voice selection and configuration

3. **NaturalLanguageUnderstanding** (`nlu-engine.js`)
   - Intent recognition using regex patterns
   - Entity extraction for parameters (time, volume)
   - Synonym matching and command suggestions
   - Confidence scoring and validation

4. **MediaController** (`media-controller.js`)
   - Comprehensive media element control
   - Fullscreen API integration
   - Progress tracking and status reporting
   - Event-driven architecture for real-time updates

5. **AudioVisualizer** (`audio-visualizer.js`)
   - Real-time audio visualization using Canvas API
   - Multiple visualization modes and effects
   - Voice activity detection
   - Responsive design with device pixel ratio support

6. **VoiceAISystem** (`app.js`)
   - Main application orchestrator
   - Component integration and coordination
   - User interface management
   - Command history and analytics

### Browser Compatibility

- **Chrome/Edge**: Full support (recommended)
- **Firefox**: Full support with some API differences
- **Safari**: Limited support (iOS requires user gesture)
- **Mobile browsers**: Varies by platform and browser

### Prerequisites

- Modern web browser with Web Speech API support
- Microphone access permission
- HTTPS connection (required for microphone access)

## Installation and Setup

1. **Clone or download** the project files
2. **Serve the files** using a web server (required for microphone access):
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Using Node.js (with serve package)
   npx serve .
   
   # Using PHP
   php -S localhost:8000
   ```
3. **Open** `https://localhost:8000` in your browser
4. **Grant microphone permission** when prompted
5. **Click "Start Listening"** to begin voice control

## Usage Guide

### Getting Started
1. Open the application in a supported browser
2. Grant microphone permissions when prompted
3. Click the "Start Listening" button
4. Speak naturally using the supported voice commands
5. Watch the audio visualizer for real-time feedback

### Voice Settings Configuration
- **Voice Selection**: Choose from available system voices
- **Pitch Control**: Adjust voice pitch (0.5 - 2.0)
- **Speech Rate**: Control speaking speed (0.5 - 2.0)
- **Confidence Threshold**: Set recognition sensitivity (0.5 - 1.0)

### Best Practices
- Speak clearly and at a moderate pace
- Use natural language - the system understands variations
- Wait for visual feedback before issuing new commands
- Check the command history for recognition accuracy
- Adjust confidence threshold if having recognition issues

## API Reference

### VoiceAISystem Class

```javascript
// Initialize the system
const voiceAI = new VoiceAISystem();

// Test voice commands programmatically
await voiceAI.testVoiceCommand("play video");

// Get system status
const status = voiceAI.getSystemStatus();

// Speak text directly
await voiceAI.speak("Hello, this is a test");

// Start/stop listening manually
await voiceAI.startListening();
voiceAI.stopListening();
```

### Command Processing Flow

1. **Audio Input** → Speech Recognition Engine
2. **Transcript** → Natural Language Understanding
3. **Intent + Entities** → Media Controller
4. **Action Result** → Speech Synthesis (feedback)
5. **History Logging** → Command History Display

## Customization

### Adding New Commands

1. **Define Intent Patterns** in `nlu-engine.js`:
   ```javascript
   NEW_COMMAND: {
       patterns: [
           /^(new|custom)\s+command$/i,
           /^do\s+something\s+new$/i
       ],
       examples: ['new command', 'do something new']
   }
   ```

2. **Implement Handler** in `media-controller.js`:
   ```javascript
   case 'NEW_COMMAND':
       return this.handleNewCommand();
   ```

3. **Add Voice Feedback** in `speech-synthesis.js`:
   ```javascript
   async speakNewCommand() {
       return this.speak("New command executed successfully");
   }
   ```

### Extending Media Controls

The `MediaController` class can be extended to support additional media types:

```javascript
// Add support for audio playlists
addPlaylistSupport() {
    // Implementation for playlist controls
}

// Add support for streaming services
addStreamingSupport() {
    // Implementation for streaming API integration
}
```

## Performance Optimization

- **Audio Context Management**: Automatically suspends when not in use
- **Canvas Rendering Optimization**: Uses requestAnimationFrame for smooth visuals
- **Memory Management**: Limits command history and cleans up resources
- **Event Debouncing**: Prevents excessive processing during continuous input
- **Responsive Design**: Adapts to different screen sizes and device capabilities

## Troubleshooting

### Common Issues

1. **Microphone Access Denied**
   - Ensure HTTPS connection
   - Check browser permissions settings
   - Try refreshing the page and granting permission again

2. **Speech Recognition Not Working**
   - Verify browser compatibility
   - Check microphone hardware
   - Adjust confidence threshold in settings

3. **Commands Not Recognized**
   - Speak more clearly
   - Try alternative command phrases
   - Check command history for patterns
   - Lower confidence threshold

4. **Audio Visualization Not Showing**
   - Ensure microphone permission is granted
   - Check browser console for WebGL/Canvas errors
   - Try refreshing the page

### Debug Mode

Enable debug logging by opening browser developer tools. The system provides comprehensive logging for:
- Speech recognition events
- Command processing steps
- Media control actions
- Audio processing status

## Security Considerations

- **Microphone Privacy**: Audio is processed locally, never sent to external servers
- **HTTPS Required**: Secure connection needed for microphone access
- **Permission Management**: Users have full control over microphone access
- **Local Processing**: All speech recognition and synthesis occurs in the browser

## Browser Support Matrix

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| Speech Recognition | ✅ | ✅ | ⚠️ | ✅ |
| Speech Synthesis | ✅ | ✅ | ✅ | ✅ |
| Audio Visualization | ✅ | ✅ | ✅ | ✅ |
| Fullscreen API | ✅ | ✅ | ✅ | ✅ |
| Media Controls | ✅ | ✅ | ✅ | ✅ |

✅ Full Support | ⚠️ Limited Support | ❌ Not Supported

## Future Enhancements

- **Wake Word Detection**: "Hey Assistant" activation
- **Multi-language Support**: Dynamic language switching
- **Cloud Speech Services**: Integration with Google/Azure Speech APIs
- **Voice Training**: Personal voice model adaptation
- **Advanced NLU**: Machine learning-based intent recognition
- **Plugin Architecture**: Extensible command system
- **Mobile App**: Native mobile application
- **IoT Integration**: Smart home device control

## Contributing

This is a complete, standalone voice AI system. To contribute:

1. Test the system with different browsers and devices
2. Report issues with specific browser/OS combinations
3. Suggest new voice commands or media controls
4. Optimize performance for different hardware configurations
5. Improve accessibility features

## License

This project is provided as a complete working example for educational and development purposes. Feel free to use, modify, and distribute according to your needs.

---

**Live Demo**: Open `index.html` in a modern web browser with microphone access to experience the full voice AI system.

**System Requirements**: Modern browser, microphone access, HTTPS connection (for production use).