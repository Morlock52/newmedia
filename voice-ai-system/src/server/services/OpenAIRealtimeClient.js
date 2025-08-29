import EventEmitter from 'events';
import WebSocket from 'ws';

// Minimal OpenAI Realtime WebSocket client wrapper
// Designed to be used by SpeechService and LLMProcessor when OPENAI_REALTIME_ENABLED=true
export class OpenAIRealtimeClient extends EventEmitter {
  constructor(options = {}) {
    super();
    this.apiKey = options.apiKey || process.env.OPENAI_API_KEY;
    this.model = options.model || process.env.OPENAI_REALTIME_MODEL || 'gpt-4o-realtime-preview';
    this.endpoint = options.endpoint || `wss://api.openai.com/v1/realtime?model=${this.model}`;
    this.ws = null;
    this.connected = false;
    this._buffer = [];
  }

  connect() {
    if (!this.apiKey) throw new Error('OPENAI API key is required for realtime client');

    if (this.connected) return Promise.resolve();

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.endpoint, {
          headers: {
            Authorization: `Bearer ${this.apiKey}`
          }
        });

        this.ws.on('open', () => {
          this.connected = true;
          this.emit('open');
          resolve();
        });

        this.ws.on('message', (msg) => {
          let d = msg.toString();
          try { d = JSON.parse(d); } catch (e) {}
          // Normalize common realtime event shapes
          // Emit raw message
          this.emit('message', d);

          // Heuristic: handle transcripts and response deltas
          if (d?.type === 'transcript' && d.text) {
            if (d.partial) this.emit('transcript.partial', d);
            else this.emit('transcript.final', d);
          }

          if (d?.type === 'response.delta' && d.response) {
            // delta may come in pieces
            this.emit('response.delta', d.response);
            if (d.response.final) this.emit('response.final', d.response);
          }
        });

        this.ws.on('close', (code, reason) => {
          this.connected = false;
          this.emit('close', { code, reason });
        });

        this.ws.on('error', (err) => {
          this.connected = false;
          this.emit('error', err);
        });
      } catch (e) {
        reject(e);
      }
    });
  }

  sendEvent(event) {
    if (!this.connected || !this.ws) {
      this._buffer.push(event);
      return;
    }
    try {
      const payload = typeof event === 'string' ? event : JSON.stringify(event);
      this.ws.send(payload);
    } catch (e) {
      this.emit('error', e);
    }
  }

  // Send audio bytes as base64 chunks (OpenAI realtime usually expects specific framing; this is minimal)
  async sendAudioChunk(base64Chunk) {
    const event = { type: 'input_audio_buffer.append', audio: base64Chunk };
    this.sendEvent(event);
  }

  // Signal end of audio buffer
  async flushAudio() {
    this.sendEvent({ type: 'input_audio_buffer.commit' });
  }

  // Send a text prompt and request a realtime response stream
  async requestResponse(prompt, options = {}) {
    const id = `req_${Date.now()}`;
    const event = {
      type: 'response.create',
      response: {
        instructions: prompt,
        modalities: ['text'],
        ...options
      },
      id
    };
    this.sendEvent(event);
    return id;
  }

  close() {
    if (this.ws) this.ws.close();
    this.ws = null;
    this.connected = false;
  }
}

export default OpenAIRealtimeClient;
