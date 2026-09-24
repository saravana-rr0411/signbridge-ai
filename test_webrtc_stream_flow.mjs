// test_webrtc_stream_flow.mjs
// Verification of WebRTC signaling, track publishing, candidate queueing,
// and state guards against renegotiation storms.

import assert from 'assert';
import { webrtcService } from './services/webrtcService.js';
import { communicationService } from './services/communicationService.js';

console.log('================================================================');
console.log('TEST SUITE: WEBRTC VIDEO STREAM & SIGNALING VERIFICATION');
console.log('================================================================\n');

// Mock MediaStreamTrack
class MockMediaStreamTrack {
  constructor(kind = 'video', id = 'track_test_1') {
    this.kind = kind;
    this.id = id;
    this.readyState = 'live';
    this.enabled = true;
  }
  stop() {
    this.readyState = 'ended';
  }
}

// Mock MediaStream
class MockMediaStream {
  constructor(tracks = []) {
    this.id = 'stream_test_' + Math.random().toString(36).substr(2, 6);
    this.active = true;
    this._tracks = [...tracks];
  }
  getTracks() {
    return this._tracks;
  }
  getVideoTracks() {
    return this._tracks.filter(t => t.kind === 'video');
  }
  getAudioTracks() {
    return this._tracks.filter(t => t.kind === 'audio');
  }
  addTrack(track) {
    this._tracks.push(track);
  }
}

// Mock RTCRtpSender
class MockRTCRtpSender {
  constructor(track) {
    this.track = track;
  }
}

// Mock RTCPeerConnection
class MockRTCPeerConnection {
  constructor(config) {
    this.config = config;
    this.localDescription = null;
    this.remoteDescription = null;
    this.signalingState = 'stable';
    this.iceConnectionState = 'new';
    this.connectionState = 'new';
    this.iceGatheringState = 'new';
    this.senders = [];
    this.addedTracks = [];
    this.onicecandidate = null;
    this.ontrack = null;
    this.oniceconnectionstatechange = null;
    this.onsignalingstatechange = null;
    this.onconnectionstatechange = null;
  }

  addTrack(track, stream) {
    this.addedTracks.push({ track, stream });
    const sender = new MockRTCRtpSender(track);
    this.senders.push(sender);
    return sender;
  }

  getSenders() {
    return this.senders;
  }

  async createOffer(options) {
    return {
      type: 'offer',
      sdp: 'v=0\r\no=- 12345 2 IN IP4 127.0.0.1\r\ns=-\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 0.0.0.0\r\na=sendonly\r\n'
    };
  }

  async createAnswer(options) {
    return {
      type: 'answer',
      sdp: 'v=0\r\no=- 67890 2 IN IP4 127.0.0.1\r\ns=-\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 0.0.0.0\r\na=recvonly\r\n'
    };
  }

  async setLocalDescription(desc) {
    this.localDescription = desc;
    if (desc.type === 'offer') {
      this.signalingState = 'have-local-offer';
    } else if (desc.type === 'answer') {
      this.signalingState = 'stable';
    }
    if (this.onsignalingstatechange) this.onsignalingstatechange();
  }

  async setRemoteDescription(desc) {
    this.remoteDescription = desc;
    if (desc.type === 'offer') {
      this.signalingState = 'have-remote-offer';
    } else if (desc.type === 'answer') {
      this.signalingState = 'stable';
    }
    if (this.onsignalingstatechange) this.onsignalingstatechange();
  }

  async addIceCandidate(candidate) {
    return true;
  }

  close() {
    this.signalingState = 'closed';
    this.iceConnectionState = 'closed';
    this.connectionState = 'closed';
  }
}

// Mock RTCSessionDescription & RTCIceCandidate
class MockRTCSessionDescription {
  constructor(init) {
    this.type = init.type;
    this.sdp = init.sdp;
  }
}

class MockRTCIceCandidate {
  constructor(init) {
    this.candidate = init.candidate || init;
    this.sdpMid = init.sdpMid || '0';
    this.sdpMLineIndex = init.sdpMLineIndex || 0;
  }
}

// Mock VideoElement
class MockVideoElement {
  constructor() {
    this.srcObject = null;
    this.autoplay = false;
    this.playsInline = false;
    this.muted = false;
    this.defaultMuted = false;
    this.attributes = {};
    this.style = {};
    this.isConnected = true;
    this.videoWidth = 1280;
    this.videoHeight = 720;
  }

  setAttribute(k, v) {
    this.attributes[k] = v;
  }

  async play() {
    return Promise.resolve();
  }
}

// Inject globals for test environment
global.RTCPeerConnection = MockRTCPeerConnection;
global.RTCSessionDescription = MockRTCSessionDescription;
global.RTCIceCandidate = MockRTCIceCandidate;
global.MediaStream = MockMediaStream;

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
  // 1. Broadcaster track verification
  // -------------------------------------------------------------------------
  console.log('--- Phase 1: Broadcaster Stream Publishing & Video Track Verification ---');

  const videoTrack = new MockMediaStreamTrack('video', 'cam_video_01');
  const mockWebcamStream = new MockMediaStream([videoTrack]);

  let sentSignalingMessages = [];
  const origSendWsDirect = communicationService.sendWsDirect.bind(communicationService);
  communicationService.sendWsDirect = (type, payload) => {
    sentSignalingMessages.push({ type, payload });
    return true;
  };

  await asyncTest('publishStream() initializes Broadcaster role and creates offer with m=video', async () => {
    await webrtcService.publishStream(mockWebcamStream);

    assert.strictEqual(webrtcService.role, 'BROADCASTER');
    assert.strictEqual(webrtcService.stream, mockWebcamStream);
    assert(webrtcService.senderPC !== null, 'senderPC must be instantiated');

    // Verify video senders
    const senders = webrtcService.senderPC.getSenders();
    assert.strictEqual(senders.length, 1, 'senderPC must have 1 track sender');
    assert.strictEqual(senders[0].track.kind, 'video', 'Sender must be for video track');

    // Verify offer created and sent
    const offerMsg = sentSignalingMessages.find(m => m.type === 'RTC_OFFER');
    assert(offerMsg !== null, 'RTC_OFFER must be dispatched via signaling');
    assert(offerMsg.payload.sdp.sdp.includes('m=video'), 'SDP offer must include m=video');
    assert.strictEqual(webrtcService.senderPC.signalingState, 'have-local-offer');
  });

  // -------------------------------------------------------------------------
  // 2. State guard: Prevent renegotiation storm from tearing down connection
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 2: State Guard against Renegotiation Storm ---');

  await asyncTest('startBroadcasterOffer does NOT destroy senderPC when have-local-offer is in flight', async () => {
    const currentPC = webrtcService.senderPC;
    assert.strictEqual(currentPC.signalingState, 'have-local-offer');

    // Simulate incoming RTC_REQUEST_STREAM while offer is awaiting answer
    await webrtcService.handleSignalingMessage({
      type: 'RTC_REQUEST_STREAM',
      id: 'test_req_dup_1'
    });

    // senderPC must NOT have been closed or replaced
    assert.strictEqual(webrtcService.senderPC, currentPC, 'senderPC must NOT be recreated while awaiting answer');
    assert.strictEqual(webrtcService.senderPC.signalingState, 'have-local-offer');
  });

  await asyncTest('startBroadcasterOffer does NOT destroy senderPC when already connected', async () => {
    const currentPC = webrtcService.senderPC;
    currentPC.iceConnectionState = 'connected';
    currentPC.connectionState = 'connected';

    await webrtcService.handleSignalingMessage({
      type: 'RTC_REQUEST_STREAM',
      id: 'test_req_dup_2'
    });

    assert.strictEqual(webrtcService.senderPC, currentPC, 'senderPC must NOT be destroyed when already connected');
  });

  // -------------------------------------------------------------------------
  // 3. Receiver subscription and remote offer handling
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 3: Receiver Subscription & Unified Plan Track Handling ---');

  // Switch to clean instance or reset role for Receiver test
  webrtcService.unpublishStream();
  assert.strictEqual(webrtcService.role, null);

  const mockAdminVideo = new MockVideoElement();
  let statusChanges = [];
  webrtcService.subscribeStream(mockAdminVideo, (status, stream) => {
    statusChanges.push({ status, stream });
  });

  assert.strictEqual(webrtcService.role, 'RECEIVER');

  await asyncTest('handleRemoteOffer creates receiverPC, handles track, and sets autoplay attributes', async () => {
    const sdpOffer = {
      type: 'offer',
      sdp: 'v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n'
    };

    await webrtcService.handleRemoteOffer(sdpOffer);

    assert(webrtcService.receiverPC !== null, 'receiverPC must be created');
    assert.strictEqual(webrtcService.receiverPC.signalingState, 'stable', 'Signaling state must be stable after answer');

    // Verify answer was dispatched
    const answerMsg = sentSignalingMessages.find(m => m.type === 'RTC_ANSWER');
    assert(answerMsg !== null, 'RTC_ANSWER must be dispatched');
    assert(answerMsg.payload.sdp.sdp.includes('m=video'), 'Answer must contain m=video section');

    // Simulate ontrack event with track (Unified Plan fallback test)
    const remoteVideoTrack = new MockMediaStreamTrack('video', 'remote_cam_track_1');
    webrtcService.receiverPC.ontrack({
      track: remoteVideoTrack,
      streams: [] // Empty streams array to test Unified Plan synthesis
    });

    assert(webrtcService.remoteStream !== null, 'remoteStream must be synthesized from track');
    assert.strictEqual(webrtcService.remoteStream.getVideoTracks().length, 1);
    assert.strictEqual(mockAdminVideo.srcObject, webrtcService.remoteStream, 'Video element must receive remote stream');

    // Verify critical autoplay attributes
    assert.strictEqual(mockAdminVideo.muted, true, 'Video element must be muted for autoplay');
    assert.strictEqual(mockAdminVideo.defaultMuted, true, 'Video element must have defaultMuted=true');
    assert.strictEqual(mockAdminVideo.playsInline, true, 'Video element must have playsInline=true');
    assert.strictEqual(mockAdminVideo.attributes['autoplay'], '', 'autoplay attribute must be set');
    assert.strictEqual(mockAdminVideo.attributes['playsinline'], '', 'playsinline attribute must be set');
  });

  // -------------------------------------------------------------------------
  // 4. Candidate queueing and draining
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 4: Candidate Queueing & Clean Cleanup ---');

  await asyncTest('handleIncomingIceCandidate queues candidate if remote description not set', async () => {
    // Reset receiver
    webrtcService.unsubscribeStream('phase4_setup');
    webrtcService.role = 'RECEIVER';
    webrtcService.receiverPC = null; // simulate candidate arriving before PC is initialized

    const testCandidate = { candidate: 'candidate:1 1 UDP 2130706431 1.2.3.4 50000 typ host', sdpMid: '0' };
    await webrtcService.handleIncomingIceCandidate({ candidate: testCandidate });

    assert.strictEqual(webrtcService.receiverQueuedCandidates.length, 1, 'Candidate must be queued');
    assert.deepStrictEqual(webrtcService.receiverQueuedCandidates[0], testCandidate);
  });

  // -------------------------------------------------------------------------
  // 5. Stale RTC_STREAM_STOPPED rejection & Session Generation Guard
  // -------------------------------------------------------------------------
  console.log('\n--- Phase 5: Session Generation & Stale RTC_STREAM_STOPPED Guard ---');

  await asyncTest('Receiver rejects stale RTC_STREAM_STOPPED with mismatched session ID', async () => {
    // Setup active receiver with session
    webrtcService.role = 'RECEIVER';
    webrtcService.activeBroadcasterSessionId = 'sess_active_123';
    webrtcService.receiverPC = new MockRTCPeerConnection();
    webrtcService.receiverPC.iceConnectionState = 'checking';
    webrtcService.receiverPC.connectionState = 'connecting';
    const mockTrack = new MockMediaStreamTrack('video', 'active_track');
    webrtcService.remoteStream = new MockMediaStream([mockTrack]);

    const activePC = webrtcService.receiverPC;

    // Simulate stale RTC_STREAM_STOPPED from past session
    await webrtcService.handleSignalingMessage({
      type: 'RTC_STREAM_STOPPED',
      streamSessionId: 'sess_stale_999',
      origin: 'broadcaster',
      senderRole: 'deaf',
      reason: 'old_unmount'
    });

    // Active connection and stream must be preserved
    assert.strictEqual(webrtcService.receiverPC, activePC, 'receiverPC must NOT be closed by stale stop message');
    assert.strictEqual(webrtcService.activeBroadcasterSessionId, 'sess_active_123', 'activeBroadcasterSessionId must remain unchanged');
    assert.notStrictEqual(webrtcService.remoteStream, null, 'remoteStream must not be cleared');
  });

  await asyncTest('Receiver rejects untargeted RTC_STREAM_STOPPED when connection is actively checking/connecting', async () => {
    const activePC = webrtcService.receiverPC;

    // Simulate untargeted legacy RTC_STREAM_STOPPED without session ID
    await webrtcService.handleSignalingMessage({
      type: 'RTC_STREAM_STOPPED',
      origin: 'broadcaster',
      senderRole: 'deaf'
    });

    // Must be preserved because receiver is active (ice=checking, conn=connecting)
    assert.strictEqual(webrtcService.receiverPC, activePC, 'receiverPC must NOT be closed by untargeted stop when connecting');
    assert.strictEqual(webrtcService.activeBroadcasterSessionId, 'sess_active_123');
  });

  await asyncTest('Receiver accepts matching RTC_STREAM_STOPPED for current session and tears down cleanly', async () => {
    // Simulate valid RTC_STREAM_STOPPED matching active session
    await webrtcService.handleSignalingMessage({
      type: 'RTC_STREAM_STOPPED',
      streamSessionId: 'sess_active_123',
      origin: 'broadcaster',
      senderRole: 'deaf',
      reason: 'user_camera_stopped'
    });

    assert.strictEqual(webrtcService.receiverPC, null, 'receiverPC must be closed and nulled on valid stop');
    assert.strictEqual(webrtcService.activeBroadcasterSessionId, null, 'activeBroadcasterSessionId must be reset');
    assert.strictEqual(webrtcService.remoteStream, null, 'remoteStream must be reset');
  });

  // Clean teardown
  webrtcService.unsubscribeStream('test_complete');
  communicationService.sendWsDirect = origSendWsDirect;

  console.log('\n================================================================');
  console.log(`SUMMARY: ${passed}/${total} WEBRTC TESTS PASSED`);
  console.log('================================================================\n');
}

runTests().catch((err) => {
  console.error('Fatal WebRTC test failure:', err);
  process.exit(1);
});
