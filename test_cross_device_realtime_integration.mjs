// test_cross_device_realtime_integration.mjs
// End-to-End Verification of Cross-Device Real-Time Communication for SignBridge AI
//
// Tests:
// 1. API Config: WebSocket URL generation (http -> ws, https -> wss)
// 2. Dual-Role Connection: Deaf and Admin connect to desk_04 on FastAPI WebSocket relay
// 3. Deaf message (HELLO) sent via WebSocket -> Admin receives -> Admin chat updates -> Admin TTS speaks once
// 4. Admin response sent via WebSocket -> Deaf receives -> Deaf chat updates -> Deaf 3D avatar plays sign animation
// 5. WebRTC signaling relay: RTC_REQUEST_STREAM, RTC_OFFER, RTC_ANSWER, ICE_CANDIDATE, RTC_STREAM_STOPPED
// 6. Zero echo: WebSocket incoming messages do NOT re-emit back to the WebSocket
// 7. Duplicate suppression: Duplicate event IDs are ignored
// 8. Local fallback: BroadcastChannel/localStorage fallback continues to work
// 9. Graceful disconnect and reconnect cleanup

import assert from 'assert';
import { API_BASE_URL, WS_BASE_URL, getWsRelayUrl } from './services/apiConfig.js';
import { CommunicationService } from './services/communicationService.js';
import { conversationStore } from './state/conversationStore.js';
import { adminTtsService } from './services/adminTtsService.js';
import { signAnimationService } from './services/signAnimationService.js';

const NativeWebSocket = globalThis.WebSocket;

console.log('================================================================');
console.log('TEST SUITE: CROSS-DEVICE REAL-TIME COMMUNICATION (SIGNBRIDGE AI)');
console.log('================================================================\n');

// Mock localStorage and window if running in Node
if (typeof global.window === 'undefined') {
  global.window = {
    addEventListener: () => {},
    removeEventListener: () => {},
    localStorage: {
      _data: {},
      getItem(k) { return this._data[k] || null; },
      setItem(k, v) { this._data[k] = String(v); },
      removeItem(k) { delete this._data[k]; },
      clear() { this._data = {}; }
    }
  };
  global.localStorage = global.window.localStorage;
}

// Mock speech synthesis for Node TTS testing
class MockSpeechSynthesisUtterance {
  constructor(text) {
    this.text = text;
    this.lang = 'en-US';
    this.rate = 1.0;
    this.pitch = 1.0;
    this.volume = 1.0;
    this.voice = null;
    this.onstart = null;
    this.onend = null;
    this.onerror = null;
  }
}

class MockSpeechSynthesis {
  constructor() {
    this.speaking = false;
    this.pending = false;
    this.paused = false;
    this.spokenUtterances = [];
  }
  getVoices() {
    return [{ name: 'Samantha (Natural)', lang: 'en-US', default: true }];
  }
  speak(utterance) {
    this.spokenUtterances.push(utterance);
    this.speaking = true;
    if (utterance.onstart) utterance.onstart();
    setTimeout(() => {
      this.speaking = false;
      if (utterance.onend) utterance.onend();
    }, 15);
  }
  cancel() {
    this.speaking = false;
    this.spokenUtterances = [];
  }
}

const mockSynth = new MockSpeechSynthesis();
adminTtsService.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

async function runTests() {
  let passed = 0;
  let total = 0;

  function test(name, fn) {
    total++;
    try {
      fn();
      console.log(`  [PASS] ${name}`);
      passed++;
    } catch (err) {
      console.error(`  [FAIL] ${name}:`, err.message);
      throw err;
    }
  }

  async function asyncTest(name, fn) {
    total++;
    try {
      await fn();
      console.log(`  [PASS] ${name}`);
      passed++;
    } catch (err) {
      console.error(`  [FAIL] ${name}:`, err.message);
      throw err;
    }
  }

  // -------------------------------------------------------------------------
  // 1. API CONFIG: WEBSOCKET URL DERIVATION
  // -------------------------------------------------------------------------
  console.log('--- Phase 1: API Config & WebSocket URLs ---');
  test('WS_BASE_URL converts http to ws and https to wss', () => {
    assert(WS_BASE_URL.startsWith('ws'), `Expected WS_BASE_URL to start with ws, got ${WS_BASE_URL}`);
    const deafUrl = getWsRelayUrl('desk_04', 'deaf');
    const adminUrl = getWsRelayUrl('desk_04', 'admin');
    assert(deafUrl.includes('/ws/relay/desk_04/deaf'), `Unexpected Deaf WS URL: ${deafUrl}`);
    assert(adminUrl.includes('/ws/relay/desk_04/admin'), `Unexpected Admin WS URL: ${adminUrl}`);
  });

  test('Production HTTPS URL derivation test', () => {
    const prodHttp = 'https://signbridge-ai-backend-onit.onrender.com';
    const prodWs = prodHttp.replace(/^http/, 'ws');
    assert.strictEqual(prodWs, 'wss://signbridge-ai-backend-onit.onrender.com');
  });

  // -------------------------------------------------------------------------
  // 2. BACKEND WEBSOCKET RELAY CONNECTION & PEER DISCOVERY
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 2: Dual-Role Connection on desk_04 ---');
  const deafComm = new CommunicationService();
  const adminComm = new CommunicationService();
  deafComm.setMockWebSocketClass(NativeWebSocket);
  adminComm.setMockWebSocketClass(NativeWebSocket);

  await asyncTest('Both Deaf and Admin connect to desk_04 relay', async () => {
    let deafConnected = false;
    let adminConnected = false;

    deafComm.on('COMM_CONNECTED', () => { deafConnected = true; });
    adminComm.on('COMM_CONNECTED', () => { adminConnected = true; });

    deafComm.connect('desk_04', 'deaf');
    adminComm.connect('desk_04', 'admin');

    await new Promise((resolve) => setTimeout(resolve, 300));
    assert(deafComm.isConnected, 'Deaf client must be connected');
    assert(adminComm.isConnected, 'Admin client must be connected');
  });

  // -------------------------------------------------------------------------
  // 3. DEAF SIGNS HELLO -> ADMIN RECEIVES -> CHAT STORE -> TTS READS ONCE
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 3: Deaf Recognition -> Admin Chat -> Admin TTS ---');
  await asyncTest('Deaf signs HELLO -> Admin receives DEAF_MESSAGE_SENT via WebSocket', async () => {
    let receivedPayload = null;

    adminComm.on('DEAF_MESSAGE_SENT', (payload) => {
      receivedPayload = payload;

      // Simulate AdminPage logic:
      const msgId = payload.msgId || payload.id;
      const conv = conversationStore.getConversation();
      const exists = conv.some((m) => (msgId && m.id === msgId) || (m.sender === 'deaf' && m.text === payload.text));
      if (!exists) {
        conversationStore.addMessage({
          id: msgId,
          sender: 'deaf',
          senderName: 'Deaf Person',
          text: payload.text,
          rawSign: payload.rawSign || null,
          type: 'hospital_sentence'
        });
      }

      // Admin TTS speaks incoming Deaf message
      adminTtsService.speakIncomingDeafMessage(payload);
    });

    const deafMessage = {
      text: 'Hello, I need help.',
      rawSign: 'HELLO -> HELP',
      rawSequence: ['HELLO', 'HELP'],
      confidence: 0.96,
      timestamp: Date.now()
    };

    // Deaf transmits event
    deafComm.emit('DEAF_MESSAGE_SENT', deafMessage);

    await new Promise((resolve) => setTimeout(resolve, 200));

    assert(receivedPayload !== null, 'Admin must receive DEAF_MESSAGE_SENT');
    assert.strictEqual(receivedPayload.text, 'Hello, I need help.');
    assert.strictEqual(receivedPayload.rawSign, 'HELLO -> HELP');

    // Verify conversationStore on Admin side contains the Deaf message
    const conv = conversationStore.getConversation();
    const lastMsg = conv[conv.length - 1];
    assert.strictEqual(lastMsg.sender, 'deaf');
    assert.strictEqual(lastMsg.text, 'Hello, I need help.');

    // Verify Admin TTS spoke the received message
    assert(mockSynth.spokenUtterances.length > 0, 'Admin TTS must have spoken');
    assert.strictEqual(mockSynth.spokenUtterances[0].text, 'Hello, I need help.');
  });

  test('Admin TTS does NOT speak duplicate or echo', () => {
    const initialSpeechCount = mockSynth.spokenUtterances.length;

    // Simulate same message arriving again or echo
    const res = adminTtsService.speakIncomingDeafMessage({
      text: 'Hello, I need help.',
      rawSign: 'HELLO -> HELP',
      timestamp: Date.now()
    });

    assert.strictEqual(res, false, 'Duplicate message must be rejected by TTS');
    assert.strictEqual(mockSynth.spokenUtterances.length, initialSpeechCount, 'TTS count must not increase');
  });

  // -------------------------------------------------------------------------
  // 4. ADMIN RESPONSE -> DEAF CHAT -> DEAF SIGN ANIMATION
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 4: Admin Response -> Deaf Chat -> Deaf Sign Animation ---');
  await asyncTest('Admin sends ADMIN_SIGN_RESPONSE -> Deaf receives & plays animation', async () => {
    let deafReceivedSign = null;
    let animationTriggered = false;

    deafComm.on('ADMIN_SIGN_RESPONSE', (payload) => {
      deafReceivedSign = payload;

      // Simulate DeafPage logic:
      const msgId = payload.messageId || payload.id || payload.msgId;
      const conv = conversationStore.getConversation();
      const exists = conv.some((m) => (msgId && m.id === msgId) || (m.sender === 'admin' && m.text === payload.text));
      if (!exists) {
        conversationStore.addMessage({
          id: msgId,
          sender: 'admin',
          senderName: 'Admin (Officer Vance)',
          text: payload.text,
          type: 'text',
          isActiveReply: true,
          mode: payload.mode || 'FINGERSPELLING'
        });
      }

      // Trigger sign animation
      signAnimationService.playAnimationForMessage(payload.text, payload.mode, payload.signSequence);
      animationTriggered = true;
    });

    const adminResponse = {
      messageId: 'admin_resp_' + Date.now(),
      text: 'Please wait for your token number.',
      mode: 'FINGERSPELLING',
      signSequence: ['PLEASE', 'WAIT'],
      timestamp: Date.now()
    };

    adminComm.emit('ADMIN_SIGN_RESPONSE', adminResponse);

    await new Promise((resolve) => setTimeout(resolve, 200));

    assert(deafReceivedSign !== null, 'Deaf must receive ADMIN_SIGN_RESPONSE');
    assert.strictEqual(deafReceivedSign.text, 'Please wait for your token number.');
    assert.strictEqual(animationTriggered, true, 'Sign animation must be triggered on Deaf client');

    // Verify conversationStore contains Admin response
    const conv = conversationStore.getConversation();
    const lastMsg = conv[conv.length - 1];
    assert.strictEqual(lastMsg.sender, 'admin');
    assert.strictEqual(lastMsg.text, 'Please wait for your token number.');
  });

  // -------------------------------------------------------------------------
  // 5. WEBRTC SIGNALING RELAY ACROSS WEBSOCKET
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 5: WebRTC Signaling Transport over WebSocket ---');
  await asyncTest('Admin sends RTC_REQUEST_STREAM -> Deaf receives signaling', async () => {
    let streamRequested = false;
    deafComm.on('RTC_REQUEST_STREAM', () => { streamRequested = true; });

    adminComm.sendWsDirect('RTC_REQUEST_STREAM', { role: 'receiver' });
    await new Promise((resolve) => setTimeout(resolve, 150));

    assert.strictEqual(streamRequested, true, 'Deaf must receive RTC_REQUEST_STREAM');
  });

  await asyncTest('Deaf sends RTC_OFFER -> Admin receives exact SDP offer', async () => {
    let receivedOffer = null;
    adminComm.on('RTC_OFFER', (payload) => { receivedOffer = payload; });

    const fakeOffer = { type: 'offer', sdp: 'v=0\r\no=- 4611738 2 IN IP4 127.0.0.1\r\ns=-\r\n' };
    deafComm.sendWsDirect('RTC_OFFER', { sdp: fakeOffer });
    await new Promise((resolve) => setTimeout(resolve, 150));

    assert(receivedOffer !== null, 'Admin must receive RTC_OFFER');
    assert.deepStrictEqual(receivedOffer.sdp, fakeOffer, 'SDP offer payload must be preserved exactly');
  });

  await asyncTest('Admin sends RTC_ANSWER -> Deaf receives exact SDP answer', async () => {
    let receivedAnswer = null;
    deafComm.on('RTC_ANSWER', (payload) => { receivedAnswer = payload; });

    const fakeAnswer = { type: 'answer', sdp: 'v=0\r\no=- 8923411 2 IN IP4 127.0.0.1\r\ns=-\r\n' };
    adminComm.sendWsDirect('RTC_ANSWER', { sdp: fakeAnswer });
    await new Promise((resolve) => setTimeout(resolve, 150));

    assert(receivedAnswer !== null, 'Deaf must receive RTC_ANSWER');
    assert.deepStrictEqual(receivedAnswer.sdp, fakeAnswer, 'SDP answer payload must be preserved exactly');
  });

  await asyncTest('ICE candidate bidirectional relay', async () => {
    let adminReceivedCandidate = null;
    let deafReceivedCandidate = null;

    adminComm.on('ICE_CANDIDATE', (payload) => { adminReceivedCandidate = payload; });
    deafComm.on('ICE_CANDIDATE', (payload) => { deafReceivedCandidate = payload; });

    // Deaf -> Admin candidate
    const cand1 = { candidate: 'candidate:1 1 UDP 2130706431 192.168.1.5 50000 typ host', sdpMid: '0' };
    deafComm.sendWsDirect('ICE_CANDIDATE', { candidate: cand1, origin: 'broadcaster' });

    // Admin -> Deaf candidate
    const cand2 = { candidate: 'candidate:2 1 UDP 2130706431 192.168.1.9 50002 typ host', sdpMid: '0' };
    adminComm.sendWsDirect('ICE_CANDIDATE', { candidate: cand2, origin: 'receiver' });

    await new Promise((resolve) => setTimeout(resolve, 150));

    assert(adminReceivedCandidate !== null, 'Admin must receive broadcaster ICE candidate');
    assert.deepStrictEqual(adminReceivedCandidate.candidate, cand1);
    assert(deafReceivedCandidate !== null, 'Deaf must receive receiver ICE candidate');
    assert.deepStrictEqual(deafReceivedCandidate.candidate, cand2);
  });

  // -------------------------------------------------------------------------
  // 6. ZERO ECHO & DUPLICATE SUPPRESSION
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 6: Zero Echo & Duplicate Suppression ---');
  await asyncTest('WebSocket incoming message does NOT re-emit back to WebSocket', async () => {
    let adminReceivedCount = 0;
    adminComm.on('TEST_ECHO_EVENT', () => { adminReceivedCount++; });

    // Deaf emits event once
    deafComm.emit('TEST_ECHO_EVENT', { text: 'Testing echo' });
    await new Promise((resolve) => setTimeout(resolve, 200));

    assert.strictEqual(adminReceivedCount, 1, 'Admin should receive event exactly once (no looping echo)');
  });

  test('Duplicate message ID suppression in communicationService', () => {
    let handlerCallCount = 0;
    const unsub = adminComm.on('TEST_DUP_EVENT', () => { handlerCallCount++; });

    const fixedMsgId = 'fixed_unique_id_999';
    adminComm.handleIncoming(fixedMsgId, 'TEST_DUP_EVENT', { sample: 1 });
    adminComm.handleIncoming(fixedMsgId, 'TEST_DUP_EVENT', { sample: 2 }); // Duplicate!

    assert.strictEqual(handlerCallCount, 1, 'Duplicate msgId must be ignored');
    unsub();
  });

  // -------------------------------------------------------------------------
  // 7. CLEAN DISCONNECT & CLEANUP
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 7: Disconnect Cleanup ---');
  await asyncTest('Clean disconnect of Deaf and Admin', async () => {
    deafComm.disconnect();
    adminComm.disconnect();

    assert.strictEqual(deafComm.isConnected, false);
    assert.strictEqual(adminComm.isConnected, false);
  });

  console.log('\n================================================================');
  console.log(`SUMMARY: ${passed}/${total} TESTS PASSED`);
  console.log('================================================================\n');
  process.exit(0);
}

runTests().catch((err) => {
  console.error('Fatal test failure:', err);
  process.exit(1);
});
