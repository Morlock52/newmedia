# OpenAI Realtime Integration

This project includes optional integration with the OpenAI Realtime API (WebSocket / WebRTC style) to provide low-latency streaming for voice transcription and LLM responses.

Usage

- Enable realtime mode by setting the environment variable:

  OPENAI_REALTIME_ENABLED=true

- Required keys:

  OPENAI_API_KEY - your OpenAI API key with Realtime access
  OPENAI_REALTIME_MODEL - model name (default: gpt-4o-realtime-preview)

Behavior

- When OPENAI_REALTIME_ENABLED is true, SpeechService and LLMProcessor will attempt to connect to the OpenAI Realtime WebSocket endpoint and stream audio / responses.
- If the realtime client fails to connect or times out, the services will gracefully fall back to configured providers (Azure, Google, ElevenLabs, local) so your system remains functional.

Notes

- Realtime requires appropriate OpenAI account access and may incur additional costs. Make sure your account has Realtime enabled and the model specified is available to you.
- The Realtime client included here is a minimal wrapper and expects OpenAI realtime framing. For production usage, review OpenAI Realtime API docs and ensure compliance with their event formats.
