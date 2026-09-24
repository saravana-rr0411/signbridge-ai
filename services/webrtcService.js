// services/webrtcService.js - Robust WebRTC Peer-to-Peer Video Stream Relay
// Connects Deaf Person camera stream to Admin "CITIZEN LIVE FEED" across independent browser windows
// Features: Dual-channel signaling (BroadcastChannel + storage event), ICE candidate queuing, auto-reconnect

import { communicationService } from './communicationService.js';

class WebRTCService {
  constructor() {
    this.role = null; // 'BROADCASTER' | 'RECEIVER' | null
    this.stream = null;
    this.remoteStream = null;
    this.videoElement = null;
    this.onStatusChange = null;

    this.senderPC = null;
    this.receiverPC = null;

    this.queuedCandidates = [];
    this.hasRemoteDescription = false;
    this.seenMessageIds = new Set();
    this.heartbeatTimer = null;
    this.requestRetryTimer = null;

    this.initSignaling();
  }

  // Dual-transport signaling: WebSocket (cross-device) + BroadcastChannel & localStorage (same-machine fallback)
  initSignaling() {
    // 1. Cross-Device WebSocket transport via communicationService
    const rtcEvents = [
      'RTC_OFFER',
      'RTC_ANSWER',
      'ICE_CANDIDATE',
      'RTC_REQUEST_STREAM',
      'RTC_STREAM_READY',
      'RTC_STREAM_STOPPED',
      'REQUEST_STREAM',
      'BROADCASTER_ANNOUNCE',
      'STREAM_OFFLINE',
      'PEER_CONNECTED'
    ];

    rtcEvents.forEach((evt) => {
      communicationService.on(evt, (payload) => {
        const data = payload && payload.type ? payload : { ...(payload || {}), type: evt };
        this.handleSignalingMessage(data);
      });
    });

    if (typeof window === 'undefined') return;

    // 2. Same-machine fallback: BroadcastChannel
    try {
      this.channel = new BroadcastChannel('signbridge_webrtc_channel_v3');
      this.channel.onmessage = (event) => {
        if (event.data) {
          this.handleSignalingMessage(event.data);
        }
      };
    } catch (e) {
      console.warn('BroadcastChannel unavailable for WebRTC:', e);
    }

    // 3. Same-machine fallback: storage event
    window.addEventListener('storage', (event) => {
      if (event.key === 'signbridge_rtc_sig_msg' && event.newValue) {
        try {
          const data = JSON.parse(event.newValue);
          this.handleSignalingMessage(data);
        } catch (e) {}
      }
    });
  }

  sendSignaling(message) {
    if (!message) return;

    // Normalize event type to standard WebRTC signaling names
    let eventType = message.type;
    if (eventType === 'REQUEST_STREAM') eventType = 'RTC_REQUEST_STREAM';
    if (eventType === 'BROADCASTER_ANNOUNCE') eventType = 'RTC_STREAM_READY';
    if (eventType === 'STREAM_OFFLINE') eventType = 'RTC_STREAM_STOPPED';

    const msgWithId = {
      ...message,
      type: eventType,
      id: message.id || ('sig_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6)),
      timestamp: message.timestamp || Date.now()
    };
    this.seenMessageIds.add(msgWithId.id);

    // 1. Cross-Device Signaling via Render WebSocket
    communicationService.sendWsDirect(eventType, msgWithId);

    // 2. Same-machine local fallback: BroadcastChannel
    if (this.channel) {
      try {
        this.channel.postMessage(msgWithId);
      } catch (e) {}
    }

    // 3. Same-machine local fallback: localStorage ping
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('signbridge_rtc_sig_msg', JSON.stringify(msgWithId));
      }
    } catch (e) {}
  }

  async handleSignalingMessage(data) {
    if (!data || !data.type) return;
    if (this.seenMessageIds.has(data.id)) return;
    this.seenMessageIds.add(data.id);

    // Prevent memory buildup of message IDs
    if (this.seenMessageIds.size > 200) {
      this.seenMessageIds.clear();
    }

    switch (data.type) {
      // Opposite peer connected to room -> trigger renegotiation if ready
      case 'PEER_CONNECTED':
        if (this.role === 'BROADCASTER' && this.stream && this.stream.active) {
          console.log('[WebRTC Broadcaster] Opposite peer connected. Initiating offer...');
          await this.startBroadcasterOffer();
        } else if (this.role === 'RECEIVER' && !this.remoteStream) {
          console.log('[WebRTC Receiver] Opposite peer connected. Requesting stream...');
          this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
        }
        break;

      // Admin requests stream from Deaf Person broadcaster
      case 'RTC_REQUEST_STREAM':
      case 'REQUEST_STREAM':
        if (this.role === 'BROADCASTER' && this.stream && this.stream.active) {
          console.log('[WebRTC Broadcaster] Received REQUEST_STREAM from Admin. Creating Offer...');
          await this.startBroadcasterOffer();
        }
        break;

      // Deaf broadcaster announces stream presence
      case 'RTC_STREAM_READY':
      case 'BROADCASTER_ANNOUNCE':
        if (this.role === 'RECEIVER' && !this.remoteStream) {
          console.log('[WebRTC Receiver] Detected active broadcaster. Requesting stream...');
          this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
        }
        break;

      // Receiver receives Offer from Broadcaster
      case 'RTC_OFFER':
        if (this.role === 'RECEIVER') {
          console.log('[WebRTC Receiver] Received RTC_OFFER. Handling offer...');
          await this.handleRemoteOffer(data.sdp);
        }
        break;

      // Broadcaster receives Answer from Receiver
      case 'RTC_ANSWER':
        if (this.role === 'BROADCASTER' && this.senderPC) {
          console.log('[WebRTC Broadcaster] Received RTC_ANSWER. Setting remote description...');
          await this.handleRemoteAnswer(data.sdp);
        }
        break;

      // ICE Candidates exchange
      case 'ICE_CANDIDATE':
        await this.handleIncomingIceCandidate(data);
        break;

      // Broadcaster stopped camera
      case 'RTC_STREAM_STOPPED':
      case 'STREAM_OFFLINE':
        if (this.role === 'RECEIVER') {
          console.log('[WebRTC Receiver] Broadcaster stream went offline.');
          this.remoteStream = null;
          if (this.videoElement) {
            this.videoElement.srcObject = null;
          }
          if (this.onStatusChange) {
            this.onStatusChange(false, null);
          }
        }
        break;
    }
  }

  // =========================================================================
  // 1. BROADCASTER ROLE (DEAF PERSON PAGE)
  // =========================================================================
  publishStream(stream) {
    if (!stream) return;
    console.log('[WebRTC Broadcaster] Publishing stream:', stream.id);
    this.role = 'BROADCASTER';
    this.stream = stream;

    // Start periodic broadcaster heartbeat so Admin window knows stream is live
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = setInterval(() => {
      if (this.stream && this.stream.active) {
        this.sendSignaling({ type: 'RTC_STREAM_READY' });
      }
    }, 1500);

    // Announce immediately and initiate offer
    this.sendSignaling({ type: 'RTC_STREAM_READY' });
    this.startBroadcasterOffer();
  }

  async startBroadcasterOffer() {
    if (!this.stream || !this.stream.active) return;

    try {
      if (this.senderPC) {
        try { this.senderPC.close(); } catch (e) {}
      }

      this.queuedCandidates = [];
      this.hasRemoteDescription = false;

      this.senderPC = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ],
        iceCandidatePoolSize: 2
      });

      // Add camera video tracks to connection
      this.stream.getTracks().forEach((track) => {
        this.senderPC.addTrack(track, this.stream);
      });

      // Send local ICE candidates to Admin receiver
      this.senderPC.onicecandidate = (event) => {
        if (event.candidate) {
          this.sendSignaling({
            type: 'ICE_CANDIDATE',
            origin: 'broadcaster',
            candidate: event.candidate.toJSON ? event.candidate.toJSON() : event.candidate
          });
        }
      };

      this.senderPC.oniceconnectionstatechange = () => {
        console.log('[WebRTC Broadcaster] ICE connection state:', this.senderPC.iceConnectionState);
      };

      // Create and send SDP Offer
      const offer = await this.senderPC.createOffer({
        offerToReceiveAudio: false,
        offerToReceiveVideo: false
      });
      await this.senderPC.setLocalDescription(offer);

      this.sendSignaling({
        type: 'RTC_OFFER',
        sdp: offer
      });
    } catch (err) {
      console.error('[WebRTC Broadcaster] Error creating offer:', err);
    }
  }

  async handleRemoteAnswer(sdp) {
    if (!this.senderPC) return;
    try {
      await this.senderPC.setRemoteDescription(new RTCSessionDescription(sdp));
      this.hasRemoteDescription = true;

      // Drain any queued candidates that arrived before the answer
      while (this.queuedCandidates.length > 0) {
        const cand = this.queuedCandidates.shift();
        try {
          await this.senderPC.addIceCandidate(new RTCIceCandidate(cand));
        } catch (e) {
          console.warn('[WebRTC Broadcaster] Queued candidate error:', e);
        }
      }
      console.log('[WebRTC Broadcaster] Remote answer set successfully. WebRTC connected!');
    } catch (err) {
      console.error('[WebRTC Broadcaster] Error setting remote answer:', err);
    }
  }

  unpublishStream() {
    console.log('[WebRTC Broadcaster] Unpublishing stream.');
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    this.sendSignaling({ type: 'RTC_STREAM_STOPPED' });

    if (this.senderPC) {
      try { this.senderPC.close(); } catch (e) {}
      this.senderPC = null;
    }
    this.stream = null;
    this.role = null;
  }

  // =========================================================================
  // 2. RECEIVER ROLE (ADMIN / CITIZEN LIVE FEED)
  // =========================================================================
  subscribeStream(videoElement, onStatusChange) {
    console.log('[WebRTC Receiver] Subscribing Admin video element to stream...');
    this.role = 'RECEIVER';
    this.videoElement = videoElement;
    this.onStatusChange = onStatusChange;

    // Check if stream already arrived or is active in same runtime (e.g. SPA)
    if (this.remoteStream && this.remoteStream.active) {
      this.attachStreamToVideo(this.remoteStream);
      if (this.onStatusChange) this.onStatusChange(true, this.remoteStream);
      return;
    }

    // Request stream from broadcaster immediately
    this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });

    // Retry requesting stream every 1.5s until stream arrives
    if (this.requestRetryTimer) clearInterval(this.requestRetryTimer);
    this.requestRetryTimer = setInterval(() => {
      if (!this.remoteStream || !this.remoteStream.active) {
        this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
      } else {
        clearInterval(this.requestRetryTimer);
        this.requestRetryTimer = null;
      }
    }, 1500);
  }

  async handleRemoteOffer(sdp) {
    try {
      if (this.receiverPC) {
        try { this.receiverPC.close(); } catch (e) {}
      }

      this.queuedCandidates = [];
      this.hasRemoteDescription = false;

      this.receiverPC = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ],
        iceCandidatePoolSize: 2
      });

      // Attach incoming media track directly to Admin video element
      this.receiverPC.ontrack = (event) => {
        console.log('[WebRTC Receiver] Incoming video track received!', event);
        if (event.streams && event.streams[0]) {
          const remoteStream = event.streams[0];
          this.remoteStream = remoteStream;
          this.attachStreamToVideo(remoteStream);

          if (this.requestRetryTimer) {
            clearInterval(this.requestRetryTimer);
            this.requestRetryTimer = null;
          }

          if (this.onStatusChange) {
            this.onStatusChange(true, remoteStream);
          }
        }
      };

      // Send local ICE candidates to Broadcaster
      this.receiverPC.onicecandidate = (event) => {
        if (event.candidate) {
          this.sendSignaling({
            type: 'ICE_CANDIDATE',
            origin: 'receiver',
            candidate: event.candidate.toJSON ? event.candidate.toJSON() : event.candidate
          });
        }
      };

      this.receiverPC.oniceconnectionstatechange = () => {
        console.log('[WebRTC Receiver] ICE connection state:', this.receiverPC.iceConnectionState);
        if (this.receiverPC.iceConnectionState === 'connected' || this.receiverPC.iceConnectionState === 'completed') {
          if (this.onStatusChange) this.onStatusChange(true, this.remoteStream);
        } else if (this.receiverPC.iceConnectionState === 'disconnected' || this.receiverPC.iceConnectionState === 'failed') {
          console.warn('[WebRTC Receiver] ICE disconnected. Re-requesting stream...');
          this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
        }
      };

      // Set Remote Description (the Offer from Deaf page)
      await this.receiverPC.setRemoteDescription(new RTCSessionDescription(sdp));
      this.hasRemoteDescription = true;

      // Drain any queued candidates that arrived before the offer was set
      while (this.queuedCandidates.length > 0) {
        const cand = this.queuedCandidates.shift();
        try {
          await this.receiverPC.addIceCandidate(new RTCIceCandidate(cand));
        } catch (e) {
          console.warn('[WebRTC Receiver] Queued candidate error:', e);
        }
      }

      // Create and send SDP Answer back to Deaf page
      const answer = await this.receiverPC.createAnswer();
      await this.receiverPC.setLocalDescription(answer);

      this.sendSignaling({
        type: 'RTC_ANSWER',
        sdp: answer
      });
    } catch (err) {
      console.error('[WebRTC Receiver] Error handling remote offer:', err);
    }
  }

  attachStreamToVideo(stream) {
    if (!this.videoElement || !stream) return;
    console.log('[WebRTC Receiver] Attaching stream to Admin video element:', stream.id);

    this.videoElement.srcObject = stream;
    this.videoElement.autoplay = true;
    this.videoElement.playsInline = true;
    this.videoElement.muted = true;
    this.videoElement.style.transform = 'none'; // Un-mirrored for Admin view
    this.videoElement.style.objectFit = 'cover';

    const playPromise = this.videoElement.play();
    if (playPromise !== undefined) {
      playPromise
        .then(() => {
          console.log('[WebRTC Receiver] Admin video playback started successfully!');
          if (this.onStatusChange) this.onStatusChange(true, stream);
        })
        .catch((err) => {
          console.warn('[WebRTC Receiver] Auto-play was prevented, retrying muted...', err);
          this.videoElement.muted = true;
          this.videoElement.play().catch(() => {});
        });
    }
  }

  unsubscribeStream() {
    console.log('[WebRTC Receiver] Unsubscribing Admin stream.');
    if (this.requestRetryTimer) {
      clearInterval(this.requestRetryTimer);
      this.requestRetryTimer = null;
    }

    if (this.receiverPC) {
      try { this.receiverPC.close(); } catch (e) {}
      this.receiverPC = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
      this.videoElement = null;
    }

    this.remoteStream = null;
    this.onStatusChange = null;
    this.role = null;
  }

  // =========================================================================
  // 3. CANDIDATE QUEUE DISPATCHER
  // =========================================================================
  async handleIncomingIceCandidate(data) {
    if (!data.candidate) return;

    const targetPC = data.origin === 'broadcaster' ? this.receiverPC : this.senderPC;
    const isTargetReady = targetPC && targetPC.remoteDescription && targetPC.remoteDescription.type;

    if (isTargetReady) {
      try {
        await targetPC.addIceCandidate(new RTCIceCandidate(data.candidate));
      } catch (err) {
        console.warn('[WebRTC] addIceCandidate error:', err);
      }
    } else {
      // Queue candidate until setRemoteDescription finishes
      this.queuedCandidates.push(data.candidate);
    }
  }

  getRemoteStream() {
    return this.remoteStream;
  }
}

export const webrtcService = new WebRTCService();
