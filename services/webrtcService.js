// services/webrtcService.js - Robust WebRTC Peer-to-Peer Video Stream Relay
// Connects Deaf Person camera stream to Admin "CITIZEN LIVE FEED" across independent devices/laptops
// Features: Dual-channel signaling, state guards against renegotiation storms, Unified Plan track recovery,
// autoplay-safe video attachment, and comprehensive connection diagnostics.

import { communicationService } from './communicationService.js';

const ICE_SERVERS = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  { urls: 'stun:stun2.l.google.com:19302' },
  { urls: 'stun:stun3.l.google.com:19302' },
  { urls: 'stun:stun4.l.google.com:19302' }
];

class WebRTCService {
  constructor() {
    this.role = null; // 'BROADCASTER' | 'RECEIVER' | null
    this.stream = null;
    this.remoteStream = null;
    this.videoElement = null;
    this.onStatusChange = null;

    this.senderPC = null;
    this.receiverPC = null;

    this.broadcasterQueuedCandidates = [];
    this.receiverQueuedCandidates = [];
    this.seenMessageIds = new Set();
    this.heartbeatTimer = null;
    this.requestRetryTimer = null;
    this.isNegotiating = false;

    this.initSignaling();
  }

  // Dual-transport signaling: WebSocket (cross-device) + BroadcastChannel & localStorage (same-machine fallback)
  initSignaling() {
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

    // Same-machine fallback: BroadcastChannel
    try {
      this.channel = new BroadcastChannel('signbridge_webrtc_channel_v3');
      this.channel.onmessage = (event) => {
        if (event.data) {
          this.handleSignalingMessage(event.data);
        }
      };
    } catch (e) {
      console.warn('[WebRTC] BroadcastChannel unavailable:', e);
    }

    // Same-machine fallback: storage event
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

    if (this.seenMessageIds.size > 300) {
      const first = this.seenMessageIds.values().next().value;
      this.seenMessageIds.delete(first);
    }

    switch (data.type) {
      // Opposite peer connected to room -> trigger renegotiation if ready
      case 'PEER_CONNECTED':
        if (this.role === 'BROADCASTER' && this.stream && this.stream.active) {
          console.log('[WebRTC Broadcaster] Opposite peer connected. Checking connection state...');
          if (this.isBroadcasterConnectionActiveOrConnecting()) {
            const ice = this.senderPC?.iceConnectionState;
            const conn = this.senderPC?.connectionState;
            const sig = this.senderPC?.signalingState;
            console.log(`[WebRTC Broadcaster] Connection active or establishing (ice=${ice}, conn=${conn}, sig=${sig}). Preserving senderPC, announcing stream presence.`);
            this.sendSignaling({ type: 'RTC_STREAM_READY' });
            return;
          }
          await this.startBroadcasterOffer();
        } else if (this.role === 'RECEIVER' && (!this.remoteStream || !this.remoteStream.active)) {
          console.log('[WebRTC Receiver] Opposite peer connected. Requesting stream...');
          this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
        }
        break;

      // Admin requests stream from Deaf Person broadcaster
      case 'RTC_REQUEST_STREAM':
      case 'REQUEST_STREAM':
        if (this.role === 'BROADCASTER' && this.stream && this.stream.active) {
          console.log('[WebRTC Broadcaster] Received REQUEST_STREAM from Admin.');
          if (this.isBroadcasterConnectionActiveOrConnecting()) {
            const ice = this.senderPC?.iceConnectionState;
            const conn = this.senderPC?.connectionState;
            const sig = this.senderPC?.signalingState;
            console.log(`[WebRTC Broadcaster] Connection active or in-progress (ice=${ice}, conn=${conn}, sig=${sig}). Preserving senderPC, skipping redundant offer.`);
            this.sendSignaling({ type: 'RTC_STREAM_READY' });
            return;
          }
          await this.startBroadcasterOffer();
        }
        break;

      // Deaf broadcaster announces stream presence
      case 'RTC_STREAM_READY':
      case 'BROADCASTER_ANNOUNCE':
        if (this.role === 'RECEIVER') {
          if (this.receiverPC) {
            const ice = this.receiverPC.iceConnectionState;
            const conn = this.receiverPC.connectionState;
            if ((ice === 'connected' || conn === 'connected') && this.remoteStream && this.remoteStream.active) {
              return; // Already streaming cleanly
            }
          }
          console.log('[WebRTC Receiver] Detected active broadcaster. Requesting stream...');
          this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
        }
        break;

      // Receiver receives Offer from Broadcaster
      case 'RTC_OFFER':
        if (this.role === 'RECEIVER') {
          console.log('[WebRTC Receiver] Received RTC_OFFER.');
          await this.handleRemoteOffer(data.sdp);
        }
        break;

      // Broadcaster receives Answer from Receiver
      case 'RTC_ANSWER':
        if (this.role === 'BROADCASTER' && this.senderPC) {
          console.log('[WebRTC Broadcaster] Received RTC_ANSWER.');
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
          // Guard: Only react if message originated from active broadcaster / deaf role
          if ((data.origin && data.origin !== 'broadcaster') || (data.senderRole && data.senderRole !== 'deaf' && data.senderRole !== 'system')) {
            console.log('[WebRTC Receiver] Ignoring non-broadcaster stream stopped event.');
            break;
          }
          console.log('[WebRTC Receiver] Broadcaster stream stopped.');
          this.remoteStream = null;
          let el = this.videoElement || (typeof document !== 'undefined' ? document.getElementById('admin-camera-video') : null);
          if (el) {
            el.srcObject = null;
          }
          if (this.receiverPC) {
            try { this.receiverPC.close(); } catch (e) {}
            this.receiverPC = null;
          }
          if (this.onStatusChange) {
            this.onStatusChange(false, null);
          }
        }
        break;
    }
  }

  // Check if existing broadcaster senderPC is active or in-progress connecting
  isBroadcasterConnectionActiveOrConnecting() {
    if (!this.senderPC) return false;
    const ice = this.senderPC.iceConnectionState;
    const conn = this.senderPC.connectionState;
    const sig = this.senderPC.signalingState;

    // Active or in-progress states that must be preserved
    if (ice === 'connected' || ice === 'completed' || conn === 'connected') {
      return true;
    }
    if (sig === 'have-local-offer') {
      return true;
    }
    if (ice === 'checking' || conn === 'connecting') {
      return true;
    }
    return false;
  }

  // =========================================================================
  // 1. BROADCASTER ROLE (DEAF PERSON PAGE)
  // =========================================================================
  publishStream(stream) {
    if (!stream) return;
    const liveTracks = stream.getVideoTracks().filter((t) => t.readyState === 'live');
    console.log('[WebRTC Broadcaster] Publishing stream:', stream.id, 'Live video tracks:', liveTracks.length);

    this.role = 'BROADCASTER';
    this.stream = stream;

    // Periodic announcement heartbeat so Admin knows stream is alive
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = setInterval(() => {
      if (this.stream && this.stream.active) {
        this.sendSignaling({ type: 'RTC_STREAM_READY' });
      }
    }, 4000);

    // Announce immediately and initiate offer
    this.sendSignaling({ type: 'RTC_STREAM_READY' });
    return this.startBroadcasterOffer();
  }

  async startBroadcasterOffer() {
    if (!this.stream || !this.stream.active) return;
    if (this.isNegotiating) {
      console.log('[WebRTC Broadcaster] Negotiation already in flight, skipping.');
      return;
    }

    try {
      this.isNegotiating = true;

      // Guard: do not destroy an already active, connected, or in-progress PeerConnection
      if (this.senderPC) {
        if (this.isBroadcasterConnectionActiveOrConnecting()) {
          const ice = this.senderPC.iceConnectionState;
          const conn = this.senderPC.connectionState;
          const sig = this.senderPC.signalingState;
          console.log(`[WebRTC Broadcaster] Connection active or establishing (ice=${ice}, conn=${conn}, sig=${sig}). Preserving.`);
          this.sendSignaling({ type: 'RTC_STREAM_READY' });
          this.isNegotiating = false;
          return;
        }

        // Previous connection failed or disconnected -> close cleanly before recreating
        try { this.senderPC.close(); } catch (e) {}
        this.senderPC = null;
      }

      console.log('[WebRTC Broadcaster] Creating new RTCPeerConnection...');
      this.broadcasterQueuedCandidates = [];

      this.senderPC = new RTCPeerConnection({
        iceServers: ICE_SERVERS,
        iceCandidatePoolSize: 2
      });

      // Diagnostics & state listeners
      this.senderPC.onsignalingstatechange = () => {
        console.log('[WebRTC Broadcaster] signalingState:', this.senderPC?.signalingState);
      };

      this.senderPC.oniceconnectionstatechange = () => {
        console.log('[WebRTC Broadcaster] iceConnectionState:', this.senderPC?.iceConnectionState);
      };

      this.senderPC.onconnectionstatechange = () => {
        console.log('[WebRTC Broadcaster] connectionState:', this.senderPC?.connectionState);
      };

      this.senderPC.onicegatheringstatechange = () => {
        console.log('[WebRTC Broadcaster] iceGatheringState:', this.senderPC?.iceGatheringState);
      };

      // Add camera video tracks to connection
      const videoTracks = this.stream.getVideoTracks().filter((t) => t.readyState === 'live');
      if (videoTracks.length === 0) {
        console.warn('[WebRTC Broadcaster] Warning: No live video tracks found in stream!');
      }

      videoTracks.forEach((track) => {
        const sender = this.senderPC.addTrack(track, this.stream);
        console.log('[WebRTC Broadcaster] Added track to senderPC:', track.kind, track.id, 'Sender confirmed:', Boolean(sender));
      });

      console.log(
        '[WebRTC Broadcaster] Active senders count:',
        this.senderPC.getSenders().length,
        this.senderPC.getSenders().map((s) => ({
          kind: s.track?.kind,
          id: s.track?.id,
          readyState: s.track?.readyState
        }))
      );

      // Send local ICE candidates to Admin receiver
      this.senderPC.onicecandidate = (event) => {
        if (event.candidate) {
          const candidateData = event.candidate.toJSON ? event.candidate.toJSON() : event.candidate;
          console.log('[WebRTC Broadcaster] Gathered local ICE candidate:', candidateData.candidate?.substring(0, 45) + '...');
          this.sendSignaling({
            type: 'ICE_CANDIDATE',
            origin: 'broadcaster',
            candidate: candidateData
          });
        } else {
          console.log('[WebRTC Broadcaster] ICE candidate gathering complete (candidate=null).');
        }
      };

      // Create and send SDP Offer
      const offer = await this.senderPC.createOffer({
        offerToReceiveAudio: false,
        offerToReceiveVideo: false
      });

      console.log('[WebRTC Broadcaster] Offer created. Contains m=video:', offer.sdp?.includes('m=video'));
      await this.senderPC.setLocalDescription(offer);
      console.log('[WebRTC Broadcaster] Local description set (offer). signalingState:', this.senderPC.signalingState);

      this.sendSignaling({
        type: 'RTC_OFFER',
        sdp: offer
      });
    } catch (err) {
      console.error('[WebRTC Broadcaster] Error creating offer:', err);
    } finally {
      this.isNegotiating = false;
    }
  }

  async handleRemoteAnswer(sdp) {
    if (!this.senderPC) {
      console.warn('[WebRTC Broadcaster] Cannot set remote answer: senderPC is null');
      return;
    }

    try {
      if (this.senderPC.signalingState !== 'have-local-offer') {
        console.warn(`[WebRTC Broadcaster] Unexpected signalingState for answer: ${this.senderPC.signalingState}. Ignoring.`);
        return;
      }

      await this.senderPC.setRemoteDescription(new RTCSessionDescription(sdp));
      console.log('[WebRTC Broadcaster] Remote answer applied successfully! signalingState:', this.senderPC.signalingState);

      // Drain queued candidates that arrived before the answer was set
      while (this.broadcasterQueuedCandidates.length > 0) {
        const cand = this.broadcasterQueuedCandidates.shift();
        try {
          await this.senderPC.addIceCandidate(new RTCIceCandidate(cand));
          console.log('[WebRTC Broadcaster] Drained queued ICE candidate successfully.');
        } catch (e) {
          console.warn('[WebRTC Broadcaster] Error applying queued candidate:', e);
        }
      }
    } catch (err) {
      console.error('[WebRTC Broadcaster] Error setting remote answer:', err);
    }
  }

  unpublishStream() {
    // FIX 2 Guard: Only broadcast RTC_STREAM_STOPPED if this instance was actually an active BROADCASTER
    const wasActiveBroadcaster = Boolean(this.role === 'BROADCASTER' && this.stream && this.stream.active !== false);

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (wasActiveBroadcaster) {
      console.log('[WebRTC Broadcaster] Unpublishing stream and notifying peers.');
      this.sendSignaling({
        type: 'RTC_STREAM_STOPPED',
        origin: 'broadcaster',
        senderRole: 'deaf'
      });
    } else {
      console.log('[WebRTC Broadcaster] unpublishStream called without active broadcast role/stream; skipping RTC_STREAM_STOPPED broadcast.');
    }

    if (this.senderPC) {
      try { this.senderPC.close(); } catch (e) {}
      this.senderPC = null;
    }
    this.stream = null;
    this.role = null;
    this.isNegotiating = false;
    this.broadcasterQueuedCandidates = [];
  }

  // =========================================================================
  // 2. RECEIVER ROLE (ADMIN / CITIZEN LIVE FEED)
  // =========================================================================
  subscribeStream(videoElement, onStatusChange) {
    console.log('[WebRTC Receiver] Subscribing Admin video element to stream...');
    this.role = 'RECEIVER';
    this.videoElement = videoElement;
    this.onStatusChange = onStatusChange;

    // Check if stream already arrived or is active in same runtime (e.g. single tab SPA test)
    if (this.remoteStream && this.remoteStream.active) {
      this.attachStreamToVideo(this.remoteStream);
      if (this.onStatusChange) this.onStatusChange(true, this.remoteStream);
      return;
    }

    // Request stream from broadcaster immediately
    this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });

    // Retry requesting stream periodically until stream arrives
    if (this.requestRetryTimer) clearInterval(this.requestRetryTimer);
    this.requestRetryTimer = setInterval(() => {
      if (!this.remoteStream || !this.remoteStream.active) {
        console.log('[WebRTC Receiver] Stream not active yet, sending RTC_REQUEST_STREAM retry...');
        this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
      } else {
        clearInterval(this.requestRetryTimer);
        this.requestRetryTimer = null;
      }
    }, 2500);
  }

  async handleRemoteOffer(sdp) {
    if (this.role !== 'RECEIVER') return;

    try {
      // Guard: If we already have a connected receiverPC and live stream, do not tear down
      if (this.receiverPC) {
        const ice = this.receiverPC.iceConnectionState;
        const conn = this.receiverPC.connectionState;
        if ((ice === 'connected' || conn === 'connected') && this.remoteStream && this.remoteStream.active) {
          console.log('[WebRTC Receiver] Already actively connected and streaming. Ignoring duplicate offer.');
          return;
        }

        try { this.receiverPC.close(); } catch (e) {}
        this.receiverPC = null;
      }

      console.log('[WebRTC Receiver] Creating new receiver RTCPeerConnection...');
      this.receiverPC = new RTCPeerConnection({
        iceServers: ICE_SERVERS,
        iceCandidatePoolSize: 2
      });

      // Diagnostics & state listeners
      this.receiverPC.onsignalingstatechange = () => {
        console.log('[WebRTC Receiver] signalingState:', this.receiverPC?.signalingState);
      };

      this.receiverPC.oniceconnectionstatechange = () => {
        console.log('[WebRTC Receiver] iceConnectionState:', this.receiverPC?.iceConnectionState);
        if (this.receiverPC?.iceConnectionState === 'connected' || this.receiverPC?.iceConnectionState === 'completed') {
          if (this.onStatusChange) this.onStatusChange(true, this.remoteStream);
        } else if (this.receiverPC?.iceConnectionState === 'disconnected' || this.receiverPC?.iceConnectionState === 'failed') {
          console.warn('[WebRTC Receiver] ICE disconnected/failed. Re-requesting stream...');
          this.sendSignaling({ type: 'RTC_REQUEST_STREAM' });
        }
      };

      this.receiverPC.onconnectionstatechange = () => {
        console.log('[WebRTC Receiver] connectionState:', this.receiverPC?.connectionState);
      };

      this.receiverPC.onicegatheringstatechange = () => {
        console.log('[WebRTC Receiver] iceGatheringState:', this.receiverPC?.iceGatheringState);
      };

      // Unified Plan ontrack: handle both stream-based and track-only payloads
      this.receiverPC.ontrack = (event) => {
        console.log('[WebRTC Receiver] Incoming video track received!', {
          kind: event.track?.kind,
          id: event.track?.id,
          readyState: event.track?.readyState,
          streamsCount: event.streams ? event.streams.length : 0
        });

        let stream = event.streams && event.streams[0] ? event.streams[0] : null;
        if (!stream) {
          console.log('[WebRTC Receiver] event.streams[0] not present; assembling MediaStream from track');
          if (!this.remoteStream) {
            this.remoteStream = new MediaStream();
          }
          this.remoteStream.addTrack(event.track);
          stream = this.remoteStream;
        } else {
          this.remoteStream = stream;
        }

        const videoTracks = stream.getVideoTracks();
        console.log(
          '[WebRTC Receiver] Remote stream video tracks count:',
          videoTracks.length,
          videoTracks.map((t) => ({ id: t.id, readyState: t.readyState, enabled: t.enabled }))
        );

        if (this.requestRetryTimer) {
          clearInterval(this.requestRetryTimer);
          this.requestRetryTimer = null;
        }

        this.attachStreamToVideo(stream);

        if (this.onStatusChange) {
          this.onStatusChange(true, stream);
        }
      };

      // Send local ICE candidates to Broadcaster
      this.receiverPC.onicecandidate = (event) => {
        if (event.candidate) {
          const candidateData = event.candidate.toJSON ? event.candidate.toJSON() : event.candidate;
          console.log('[WebRTC Receiver] Gathered local ICE candidate:', candidateData.candidate?.substring(0, 45) + '...');
          this.sendSignaling({
            type: 'ICE_CANDIDATE',
            origin: 'receiver',
            candidate: candidateData
          });
        } else {
          console.log('[WebRTC Receiver] ICE candidate gathering complete (candidate=null).');
        }
      };

      // Set Remote Description (the Offer from Deaf page)
      await this.receiverPC.setRemoteDescription(new RTCSessionDescription(sdp));
      console.log('[WebRTC Receiver] Remote description set (offer). signalingState:', this.receiverPC.signalingState);

      // Drain queued candidates that arrived before the offer was set
      while (this.receiverQueuedCandidates.length > 0) {
        const cand = this.receiverQueuedCandidates.shift();
        try {
          await this.receiverPC.addIceCandidate(new RTCIceCandidate(cand));
          console.log('[WebRTC Receiver] Drained queued candidate successfully.');
        } catch (e) {
          console.warn('[WebRTC Receiver] Error draining queued candidate:', e);
        }
      }

      // Create and send SDP Answer back to Deaf page
      const answer = await this.receiverPC.createAnswer();
      await this.receiverPC.setLocalDescription(answer);
      console.log('[WebRTC Receiver] Local answer created and set. signalingState:', this.receiverPC.signalingState);

      this.sendSignaling({
        type: 'RTC_ANSWER',
        sdp: answer
      });
    } catch (err) {
      console.error('[WebRTC Receiver] Error handling remote offer:', err);
    }
  }

  attachStreamToVideo(stream) {
    if (!stream) return;

    // Look up live element if cached reference is detached or null
    let el = this.videoElement;
    if (!el || !el.isConnected) {
      if (typeof document !== 'undefined') {
        el = document.getElementById('admin-camera-video');
        if (el) this.videoElement = el;
      }
    }

    if (!el) {
      console.warn('[WebRTC Receiver] No DOM video element found to attach stream');
      return;
    }

    console.log('[WebRTC Receiver] Attaching stream to Admin video element:', stream.id);

    // Guaranteed autoplay configuration
    el.autoplay = true;
    el.playsInline = true;
    el.muted = true;
    el.defaultMuted = true;
    el.setAttribute('autoplay', '');
    el.setAttribute('playsinline', '');
    el.setAttribute('muted', '');
    el.style.transform = 'none'; // Un-mirrored for Admin view
    el.style.objectFit = 'cover';

    if (el.srcObject !== stream) {
      el.srcObject = stream;
      console.log('[WebRTC Receiver] Assigned srcObject to video element successfully');
    }

    const playPromise = el.play();
    if (playPromise !== undefined) {
      playPromise
        .then(() => {
          console.log('[WebRTC Receiver] Admin video playback started successfully! Dimensions:', el.videoWidth, 'x', el.videoHeight);
          if (this.onStatusChange) this.onStatusChange(true, stream);
        })
        .catch((err) => {
          console.warn('[WebRTC Receiver] Auto-play was prevented, retrying muted...', err);
          el.muted = true;
          el.defaultMuted = true;
          el.play().catch(() => {});
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
    this.receiverQueuedCandidates = [];
  }

  // =========================================================================
  // 3. CANDIDATE QUEUE DISPATCHER
  // =========================================================================
  async handleIncomingIceCandidate(data) {
    if (!data || !data.candidate) return;

    // Deterministic target selection based on this peer's role
    const targetPC = this.role === 'RECEIVER' ? this.receiverPC : this.senderPC;
    const queue = this.role === 'RECEIVER' ? this.receiverQueuedCandidates : this.broadcasterQueuedCandidates;

    if (!targetPC) {
      console.log(`[WebRTC ${this.role || 'PEER'}] Target PC not ready yet; queuing ICE candidate`);
      queue.push(data.candidate);
      return;
    }

    const hasRemote = targetPC.remoteDescription && targetPC.remoteDescription.type;
    if (hasRemote) {
      try {
        await targetPC.addIceCandidate(new RTCIceCandidate(data.candidate));
        console.log(`[WebRTC ${this.role}] Added ICE candidate successfully`);
      } catch (err) {
        console.warn(`[WebRTC ${this.role}] addIceCandidate error:`, err);
      }
    } else {
      console.log(`[WebRTC ${this.role}] Remote description not set yet; queuing ICE candidate`);
      queue.push(data.candidate);
    }
  }

  getRemoteStream() {
    return this.remoteStream;
  }
}

export const webrtcService = new WebRTCService();
