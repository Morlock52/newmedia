#!/usr/bin/env node
import fetch from 'node-fetch';
import wrtc from 'wrtc';

const SIGNAL_URL = process.env.VOICE_SERVER_SIGNAL_URL || 'http://localhost:8080/api/realtime/signal';
const MODEL = process.env.OPENAI_REALTIME_MODEL || 'gpt-4o-realtime-preview';

async function run() {
  if (!process.env.OPENAI_API_KEY) {
    console.log('OPENAI_API_KEY not set, skipping WebRTC client test');
    process.exit(0);
  }

  const pc = new wrtc.RTCPeerConnection();

  // create audio source and track
  const source = new wrtc.nonstandard.RTCAudioSource();
  const track = source.createTrack();
  pc.addTrack(track);

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const resp = await fetch(SIGNAL_URL, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sdp: offer.sdp, type: offer.type, sessionId: 'test-session' })
  });
  const ans = await resp.json();
  await pc.setRemoteDescription({ type: ans.type, sdp: ans.sdp });

  // send a few silent audio frames
  const sampleRate = 16000;
  const frames = 3;
  for (let f=0; f<frames; f++) {
    const samples = new Int16Array(160);
    source.onData({ samples, sampleRate });
    await new Promise(r => setTimeout(r, 100));
  }

  console.log('Sent audio frames; waiting for transcription events...');
  // Wait for a bit (server should forward transcripts via datachannel or other channels)
  await new Promise(r => setTimeout(r, 10000));
  console.log('WebRTC test completed.');
  process.exit(0);
}

run().catch(e => { console.error('webrtc test failed', e); process.exit(2); });
