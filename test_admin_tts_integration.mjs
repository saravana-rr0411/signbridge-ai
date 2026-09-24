// test_admin_tts_integration.mjs
// Focused Test Suite for Admin Console Real-Time Text-to-Speech (TTS) Integration
import assert from 'assert';
import { AdminTtsService, adminTtsService } from './services/adminTtsService.js';
import { communicationService } from './services/communicationService.js';
import { hospitalConversationService } from './services/conversation/hospitalConversationService.js';
import { conversationStore } from './state/conversationStore.js';

console.log('====================================================');
console.log('TESTING ADMIN PAGE REAL-TIME TEXT-TO-SPEECH (TTS)');
console.log('====================================================');

// Mock SpeechSynthesis and SpeechSynthesisUtterance for Node runtime
class MockSpeechSynthesisUtterance {
  constructor(text) {
    this.text = text;
    this.lang = '';
    this.rate = 1;
    this.pitch = 1;
    this.volume = 1;
    this.voice = null;
    this.onend = null;
    this.onerror = null;
  }
}

class MockSpeechSynthesis {
  constructor() {
    this.spokenUtterances = [];
    this.currentUtterance = null;
    this.cancelledCount = 0;
    this.speaking = false;
    this.paused = false;
  }

  getVoices() {
    return [
      { name: 'Google US English', lang: 'en-US' },
      { name: 'Samantha', lang: 'en-US' },
      { name: 'Alex', lang: 'en-US' }
    ];
  }

  speak(utterance) {
    this.spokenUtterances.push(utterance);
    this.currentUtterance = utterance;
    this.speaking = true;
  }

  cancel() {
    this.cancelledCount++;
    this.speaking = false;
    this.currentUtterance = null;
  }

  pause() {
    this.paused = true;
  }

  resume() {
    this.paused = false;
  }

  finishCurrent() {
    if (this.currentUtterance && typeof this.currentUtterance.onend === 'function') {
      const u = this.currentUtterance;
      this.currentUtterance = null;
      this.speaking = false;
      u.onend();
    }
  }
}

// ----------------------------------------------------------------------------
// TEST 1: Utterance Configuration & Voice Selection
// ----------------------------------------------------------------------------
console.log('\n--- SECTION 1: BROWSER SPEECH API CONFIGURATION ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0; // immediate for test
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  tts.speakIncomingDeafMessage({
    id: 'test_cfg_1',
    sender: 'deaf',
    text: 'Hello, I need help.'
  });

  assert.strictEqual(mockSynth.spokenUtterances.length, 1, 'One utterance spoken');
  const utt = mockSynth.spokenUtterances[0];
  assert.strictEqual(utt.text, 'Hello, I need help.', 'Exact text configured');
  assert.strictEqual(utt.lang, 'en-US', 'Language is en-US');
  assert.strictEqual(utt.rate, 0.95, 'Speech rate is natural 0.95');
  assert.strictEqual(utt.pitch, 1.0, 'Pitch is 1.0');
  assert.strictEqual(utt.volume, 1.0, 'Volume is 1.0');
  assert(utt.voice !== null, 'Voice was selected from available voices');
  assert.strictEqual(utt.voice.lang, 'en-US', 'Voice language is en-US');
  console.log('✅ [PASS] Native SpeechSynthesisUtterance configured with en-US, rate 0.95, pitch 1.0, volume 1.0');
}

// ----------------------------------------------------------------------------
// CASE 1: Deaf signs HELLO -> Admin speaks "Hello."
// ----------------------------------------------------------------------------
console.log('\n--- CASE 1: DEAF SIGNS HELLO ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  tts.speakIncomingDeafMessage({
    id: 'msg_hello_1',
    sender: 'deaf',
    text: 'Hello.',
    rawSign: 'HELLO',
    rawSequence: ['HELLO']
  });

  assert.strictEqual(mockSynth.spokenUtterances.length, 1);
  assert.strictEqual(mockSynth.spokenUtterances[0].text, 'Hello.');
  console.log('✅ [PASS] CASE 1: Deaf signs HELLO -> Admin speaks "Hello."');
}

// ----------------------------------------------------------------------------
// CASE 2: Deaf signs HELP -> Admin speaks "I need help."
// ----------------------------------------------------------------------------
console.log('\n--- CASE 2: DEAF SIGNS HELP ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  tts.speakIncomingDeafMessage({
    id: 'msg_help_1',
    sender: 'deaf',
    text: 'I need help.',
    rawSign: 'HELP',
    rawSequence: ['HELP']
  });

  assert.strictEqual(mockSynth.spokenUtterances.length, 1);
  assert.strictEqual(mockSynth.spokenUtterances[0].text, 'I need help.');
  console.log('✅ [PASS] CASE 2: Deaf signs HELP -> Admin speaks "I need help."');
}

// ----------------------------------------------------------------------------
// CASE 3: Deaf signs HELLO + HELP -> Admin speaks complete sentence ONCE
// ----------------------------------------------------------------------------
console.log('\n--- CASE 3: DEAF SIGNS HELLO + HELP (SUPERSEDING SEQUENCE) ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  // 1. Initial gesture prefix HELLO arrives
  tts.speakIncomingDeafMessage({
    id: 'msg_seq_hello',
    sender: 'deaf',
    text: 'Hello.',
    rawSign: 'HELLO',
    rawSequence: ['HELLO']
  });

  // Verify HELLO started speaking
  assert.strictEqual(mockSynth.spokenUtterances.length, 1);
  assert.strictEqual(mockSynth.spokenUtterances[0].text, 'Hello.');

  // 2. Accumulated complete sentence arrives before HELLO finishes
  tts.speakIncomingDeafMessage({
    id: 'msg_seq_hello_help',
    sender: 'deaf',
    text: 'Hello, I need help.',
    rawSign: 'HELLO -> HELP',
    rawSequence: ['HELLO', 'HELP']
  });

  // speechSynthesis.cancel() must have been called to interrupt partial speech
  assert(mockSynth.cancelledCount >= 1, 'Partial speech cancelled for superseding sentence');
  assert.strictEqual(mockSynth.spokenUtterances.length, 2);
  assert.strictEqual(mockSynth.spokenUtterances[1].text, 'Hello, I need help.');
  console.log('✅ [PASS] CASE 3: Deaf signs HELLO + HELP -> Complete sentence "Hello, I need help." supersedes prefix');
}

// ----------------------------------------------------------------------------
// CASE 4: Deaf signs SICK + DOCTOR -> Admin speaks complete contextual message
// ----------------------------------------------------------------------------
console.log('\n--- CASE 4: DEAF SIGNS SICK + DOCTOR ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  tts.speakIncomingDeafMessage({
    id: 'msg_sick_doc',
    sender: 'deaf',
    text: 'I am sick. I need the doctor.',
    rawSign: 'SICK -> DOCTOR',
    rawSequence: ['SICK', 'DOCTOR']
  });

  assert.strictEqual(mockSynth.spokenUtterances.length, 1);
  assert.strictEqual(mockSynth.spokenUtterances[0].text, 'I am sick. I need the doctor.');
  console.log('✅ [PASS] CASE 4: Deaf signs SICK + DOCTOR -> Admin speaks "I am sick. I need the doctor."');
}

// ----------------------------------------------------------------------------
// CASE 5: Duplicate communication event -> Admin speaks only ONCE
// ----------------------------------------------------------------------------
console.log('\n--- CASE 5: DUPLICATE COMMUNICATION EVENT ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  const eventPayload = {
    id: 'msg_dup_test_100',
    sender: 'deaf',
    text: 'Where should I wait?',
    timestamp: 1727189999000
  };

  // First arrival
  const accepted1 = tts.speakIncomingDeafMessage(eventPayload);
  assert.strictEqual(accepted1, true, 'First event accepted');
  assert.strictEqual(mockSynth.spokenUtterances.length, 1);

  // Duplicate arrival (same id / same payload across BroadcastChannel & localStorage)
  const accepted2 = tts.speakIncomingDeafMessage(eventPayload);
  assert.strictEqual(accepted2, false, 'Duplicate event rejected');
  assert.strictEqual(mockSynth.spokenUtterances.length, 1, 'Utterance was NOT spoken twice');

  // Finish first speech item before speaking next distinct message
  mockSynth.finishCurrent();

  // Duplicate arrival without explicit ID but identical timestamp & text
  const payloadNoId = {
    sender: 'deaf',
    text: 'Where is the consultation room?',
    timestamp: 1727190000000
  };
  const accepted3 = tts.speakIncomingDeafMessage(payloadNoId);
  assert.strictEqual(accepted3, true, 'First timestamped event accepted');
  const accepted4 = tts.speakIncomingDeafMessage(payloadNoId);
  assert.strictEqual(accepted4, false, 'Duplicate timestamped event rejected');
  assert.strictEqual(mockSynth.spokenUtterances.length, 2, 'Total spoken utterances strictly 2');

  console.log('✅ [PASS] CASE 5: Duplicate communication events ignored; Admin speaks strictly ONCE');
}

// ----------------------------------------------------------------------------
// CASE 6: Admin sends response -> Admin outgoing message must NOT trigger TTS
// ----------------------------------------------------------------------------
console.log('\n--- CASE 6: ADMIN OUTGOING MESSAGE EXCLUSION ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  const acceptedAdminMsg = tts.speakIncomingDeafMessage({
    id: 'msg_admin_out_1',
    sender: 'admin',
    senderName: 'Admin (Officer Vance)',
    text: 'Please wait here. The doctor will examine you shortly.'
  });

  assert.strictEqual(acceptedAdminMsg, false, 'Admin outgoing message rejected');
  assert.strictEqual(mockSynth.spokenUtterances.length, 0, 'No utterance created for Admin message');
  console.log('✅ [PASS] CASE 6: Admin outgoing response strictly blocked from triggering Admin TTS');
}

// ----------------------------------------------------------------------------
// CASE 7: Rapid incoming messages -> Controlled sequential queue without overlap
// ----------------------------------------------------------------------------
console.log('\n--- CASE 7: RAPID INCOMING MESSAGES (SAFE QUEUE) ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  // Send 3 distinct messages rapidly
  tts.speakIncomingDeafMessage({ id: 'rapid_1', sender: 'deaf', text: 'Message 1' });
  tts.speakIncomingDeafMessage({ id: 'rapid_2', sender: 'deaf', text: 'Message 2' });
  tts.speakIncomingDeafMessage({ id: 'rapid_3', sender: 'deaf', text: 'Message 3' });

  // Only Message 1 should be actively speaking right now (no concurrent synthesis)
  assert.strictEqual(mockSynth.spokenUtterances.length, 1, 'Only first message speaking');
  assert.strictEqual(mockSynth.spokenUtterances[0].text, 'Message 1');
  assert.strictEqual(tts.getQueue().length, 2, 'Remaining 2 messages held in FIFO queue');

  // Finish Message 1
  mockSynth.finishCurrent();
  assert.strictEqual(mockSynth.spokenUtterances.length, 2, 'Second message started after first finished');
  assert.strictEqual(mockSynth.spokenUtterances[1].text, 'Message 2');
  assert.strictEqual(tts.getQueue().length, 1, 'Remaining 1 message in queue');

  // Finish Message 2
  mockSynth.finishCurrent();
  assert.strictEqual(mockSynth.spokenUtterances.length, 3, 'Third message started after second finished');
  assert.strictEqual(mockSynth.spokenUtterances[2].text, 'Message 3');
  assert.strictEqual(tts.getQueue().length, 0, 'Queue completely drained');

  // Finish Message 3
  mockSynth.finishCurrent();
  assert.strictEqual(tts.isSpeaking(), false, 'TTS state is inactive after queue completes');
  console.log('✅ [PASS] CASE 7: Rapid messages safely queued sequentially without overlapping voices');
}

// ----------------------------------------------------------------------------
// TEST 8: State Change Listener & UI Indicator Integration
// ----------------------------------------------------------------------------
console.log('\n--- SECTION 8: STATE CHANGE LISTENER FOR UI INDICATOR ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  let indicatorState = null;
  const unsub = tts.onStateChange((isSpeaking) => {
    indicatorState = isSpeaking;
  });

  tts.speakIncomingDeafMessage({ id: 'state_test', sender: 'deaf', text: 'Hello.' });
  assert.strictEqual(indicatorState, true, 'Speaking indicator turned ON');

  mockSynth.finishCurrent();
  assert.strictEqual(indicatorState, false, 'Speaking indicator turned OFF after speech');

  unsub();
  console.log('✅ [PASS] State change listener triggers indicator ON/OFF accurately');
}

// ----------------------------------------------------------------------------
// TEST 9: Pre-seeding Historical Messages (Page Load Protection)
// ----------------------------------------------------------------------------
console.log('\n--- SECTION 9: HISTORICAL CONVERSATION PRE-SEEDING ---');
{
  const mockSynth = new MockSpeechSynthesis();
  const tts = new AdminTtsService();
  tts.debounceMs = 0;
  tts.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  const history = [
    { id: 'hist_1', sender: 'deaf', text: 'Old message 1' },
    { id: 'hist_2', sender: 'deaf', text: 'Old message 2' }
  ];

  tts.markExistingAsSpoken(history);

  // Try to speak historical message 1
  const res = tts.speakIncomingDeafMessage(history[0]);
  assert.strictEqual(res, false, 'Historical message not spoken');
  assert.strictEqual(mockSynth.spokenUtterances.length, 0, 'Zero utterances created for pre-seeded history');
  console.log('✅ [PASS] Existing conversation history pre-seeded; old messages not read on mount');
}

// ----------------------------------------------------------------------------
// TEST 10: Full End-to-End Event Bus Wiring (Hospital Phrase -> DEAF_MESSAGE_SENT)
// ----------------------------------------------------------------------------
console.log('\n--- SECTION 10: END-TO-END EVENT BUS WIRING ---');
{
  const mockSynth = new MockSpeechSynthesis();
  adminTtsService.reset();
  adminTtsService.debounceMs = 0;
  adminTtsService.setSpeechSynthesis(mockSynth, MockSpeechSynthesisUtterance);

  // Simulate Admin page event listener subscription
  const unsub = communicationService.on('DEAF_MESSAGE_SENT', (payload) => {
    adminTtsService.speakIncomingDeafMessage(payload);
  });

  // Deaf Person signs via hospital conversation layer
  hospitalConversationService.reset();
  const tNow = Date.now();
  hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: tNow, sendToChat: true });

  assert.strictEqual(mockSynth.spokenUtterances.length, 1, 'Event bus triggered TTS');
  assert.strictEqual(mockSynth.spokenUtterances[0].text, 'Hello.', 'Spoken text matches hospital layer output');

  unsub();
  adminTtsService.reset();
  console.log('✅ [PASS] Full E2E communication bus event triggers Admin TTS seamlessly');
}

console.log('\n====================================================');
console.log('ADMIN TTS TEST SUITE: 10/10 TESTS PASSED');
console.log('====================================================\n');
