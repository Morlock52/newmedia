#!/usr/bin/env node
// Simple runtime integration test for OpenAI Realtime (gated by OPENAI_API_KEY)
import OpenAIRealtimeClient from '../../voice-ai-system/src/server/services/OpenAIRealtimeClient.js';

const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_REALTIME_MODEL || 'gpt-4o-realtime-preview';

if (!API_KEY) {
  console.log('OPENAI_API_KEY not set — skipping realtime integration test');
  process.exit(0);
}

(async () => {
  const client = new OpenAIRealtimeClient({ apiKey: API_KEY, model: MODEL });
  try {
    console.log('Connecting to OpenAI Realtime...');
    await client.connect();

    console.log('Requesting realtime response...');
    const reqId = await client.requestResponse('Say hello and then pong.', { modalities: ['text'] });

    let acc = '';
    const timeout = setTimeout(() => {
      console.error('Realtime test timeout');
      process.exit(2);
    }, 30000);

    client.on('response.delta', (d) => {
      if (d?.text) {
        acc += d.text;
        console.log('delta:', d.text);
      }
    });

    client.on('response.final', (d) => {
      clearTimeout(timeout);
      acc += d.text || '';
      console.log('final response:', acc);
      process.exit(0);
    });

  } catch (e) {
    console.error('Realtime integration test failed:', e);
    process.exit(3);
  }
})();
