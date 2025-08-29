const { OpenAIRealtimeClient } = require('../../voice-ai-system/src/server/services/OpenAIRealtimeClient.js');

describe('OpenAIRealtimeClient (smoke)', () => {
  it('constructs without crashing', () => {
    const client = new OpenAIRealtimeClient({ apiKey: process.env.OPENAI_API_KEY || 'test', model: 'test-model' });
    expect(client).toBeDefined();
  });
});
