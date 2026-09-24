"""
SignBridge AI - Cross-Device WebSocket Relay Integration Tests
Validates connection lifecycle, role segregation, message relaying, WebRTC signaling relay,
duplicate connection handling, and graceful disconnect cleanup.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app, manager


def test_invalid_role_rejected():
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/relay/desk_04/invalid_role") as ws:
            pass


def test_deaf_and_admin_connect_and_receive_peer_notifications():
    client = TestClient(app)
    room_id = "test_room_01"

    # Deaf connects first
    with client.websocket_connect(f"/ws/relay/{room_id}/deaf") as ws_deaf:
        # Admin connects second -> Deaf should receive PEER_CONNECTED notification
        with client.websocket_connect(f"/ws/relay/{room_id}/admin") as ws_admin:
            # Deaf receives system message that admin connected
            msg = ws_deaf.receive_json()
            assert msg["type"] == "PEER_CONNECTED"
            assert msg["senderRole"] == "system"
            assert msg["targetRole"] == "deaf"
            assert msg["payload"]["connectedRole"] == "admin"


def test_deaf_message_relayed_to_admin():
    client = TestClient(app)
    room_id = "test_room_02"

    with client.websocket_connect(f"/ws/relay/{room_id}/deaf") as ws_deaf:
        with client.websocket_connect(f"/ws/relay/{room_id}/admin") as ws_admin:
            # Drain PEER_CONNECTED notification
            _ = ws_deaf.receive_json()

            # Deaf sends DEAF_MESSAGE_SENT
            deaf_payload = {
                "text": "Hello, I need help.",
                "rawSign": "HELLO -> HELP",
                "rawSequence": ["HELLO", "HELP"],
                "confidence": 0.94,
                "timestamp": 123456789
            }
            ws_deaf.send_json({
                "type": "DEAF_MESSAGE_SENT",
                "payload": deaf_payload
            })

            # Admin receives the relayed envelope
            envelope = ws_admin.receive_json()
            assert envelope["roomId"] == room_id
            assert envelope["senderRole"] == "deaf"
            assert envelope["targetRole"] == "admin"
            assert envelope["type"] == "DEAF_MESSAGE_SENT"
            assert envelope["payload"]["text"] == "Hello, I need help."
            assert envelope["payload"]["rawSign"] == "HELLO -> HELP"


def test_admin_response_relayed_to_deaf():
    client = TestClient(app)
    room_id = "test_room_03"

    with client.websocket_connect(f"/ws/relay/{room_id}/deaf") as ws_deaf:
        with client.websocket_connect(f"/ws/relay/{room_id}/admin") as ws_admin:
            _ = ws_deaf.receive_json()

            # Admin sends ADMIN_SIGN_RESPONSE
            admin_payload = {
                "text": "Please take a seat, the doctor will see you shortly.",
                "mode": "FINGERSPELLING",
                "signSequence": ["PLEASE", "TAKE", "SEAT"],
                "messageId": "msg_admin_123",
                "timestamp": 987654321
            }
            ws_admin.send_json({
                "type": "ADMIN_SIGN_RESPONSE",
                "payload": admin_payload
            })

            # Deaf receives the relayed envelope
            envelope = ws_deaf.receive_json()
            assert envelope["roomId"] == room_id
            assert envelope["senderRole"] == "admin"
            assert envelope["targetRole"] == "deaf"
            assert envelope["type"] == "ADMIN_SIGN_RESPONSE"
            assert envelope["payload"]["text"] == "Please take a seat, the doctor will see you shortly."
            assert envelope["payload"]["mode"] == "FINGERSPELLING"


def test_webrtc_signaling_relay_offer_answer_candidates():
    client = TestClient(app)
    room_id = "test_room_04"

    with client.websocket_connect(f"/ws/relay/{room_id}/deaf") as ws_deaf:
        with client.websocket_connect(f"/ws/relay/{room_id}/admin") as ws_admin:
            _ = ws_deaf.receive_json()

            # 1. Admin sends RTC_REQUEST_STREAM
            ws_admin.send_json({
                "type": "RTC_REQUEST_STREAM",
                "payload": {}
            })
            req_envelope = ws_deaf.receive_json()
            assert req_envelope["type"] == "RTC_REQUEST_STREAM"
            assert req_envelope["senderRole"] == "admin"

            # 2. Deaf sends RTC_OFFER
            fake_offer = {"type": "offer", "sdp": "v=0\r\no=deaf 12345..."}
            ws_deaf.send_json({
                "type": "RTC_OFFER",
                "payload": {"sdp": fake_offer}
            })
            offer_envelope = ws_admin.receive_json()
            assert offer_envelope["type"] == "RTC_OFFER"
            assert offer_envelope["payload"]["sdp"] == fake_offer

            # 3. Admin sends RTC_ANSWER
            fake_answer = {"type": "answer", "sdp": "v=0\r\no=admin 67890..."}
            ws_admin.send_json({
                "type": "RTC_ANSWER",
                "payload": {"sdp": fake_answer}
            })
            answer_envelope = ws_deaf.receive_json()
            assert answer_envelope["type"] == "RTC_ANSWER"
            assert answer_envelope["payload"]["sdp"] == fake_answer

            # 4. Deaf sends ICE candidate
            fake_candidate = {"candidate": "candidate:1 1 UDP 2130706431 ...", "sdpMid": "0"}
            ws_deaf.send_json({
                "type": "ICE_CANDIDATE",
                "payload": {"candidate": fake_candidate, "origin": "broadcaster"}
            })
            ice_envelope = ws_admin.receive_json()
            assert ice_envelope["type"] == "ICE_CANDIDATE"
            assert ice_envelope["payload"]["candidate"] == fake_candidate


def test_disconnect_and_peer_notification():
    client = TestClient(app)
    room_id = "test_room_05"

    with client.websocket_connect(f"/ws/relay/{room_id}/admin") as ws_admin:
        with client.websocket_connect(f"/ws/relay/{room_id}/deaf") as ws_deaf:
            # Admin receives PEER_CONNECTED
            conn_msg = ws_admin.receive_json()
            assert conn_msg["type"] == "PEER_CONNECTED"
            assert conn_msg["payload"]["connectedRole"] == "deaf"

        # Deaf closes connection -> Admin should receive PEER_DISCONNECTED
        disc_msg = ws_admin.receive_json()
        assert disc_msg["type"] == "PEER_DISCONNECTED"
        assert disc_msg["payload"]["disconnectedRole"] == "deaf"


def test_duplicate_same_role_connection_handled_gracefully():
    client = TestClient(app)
    room_id = "test_room_06"

    # Deaf 1 connects
    ws_deaf_1 = client.websocket_connect(f"/ws/relay/{room_id}/deaf")
    ws_deaf_1.__enter__()

    # Deaf 2 connects with same role -> manager replaces Deaf 1 without crashing
    with client.websocket_connect(f"/ws/relay/{room_id}/deaf") as ws_deaf_2:
        with client.websocket_connect(f"/ws/relay/{room_id}/admin") as ws_admin:
            # Drain PEER_CONNECTED notification
            _ = ws_deaf_2.receive_json()

            # Message from Deaf 2 reaches Admin
            ws_deaf_2.send_json({
                "type": "DEAF_MESSAGE_SENT",
                "payload": {"text": "From replacement client"}
            })
            msg = ws_admin.receive_json()
            assert msg["payload"]["text"] == "From replacement client"

    try:
        ws_deaf_1.__exit__(None, None, None)
    except Exception:
        pass
