"""Server connection behaviour that does not need a running server."""

import json
import logging
import socket
import time

from services.network_service import NetworkService


def unused_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_service() -> NetworkService:
    service = NetworkService(connection_retry_delay=0.1)
    service.server_host = "127.0.0.1"
    service.server_port = unused_port()  # nothing listens here
    return service


def test_monitor_detections_are_queued_with_their_frame_in_the_server_format():
    service = make_service()
    service.send_stream_detections(
        [{"class": "whitefly", "confidence": 0.8, "bbox": (1, 2, 3, 4)}], (640, 480), b"jpeg-bytes"
    )
    message, frame = service._stream_detections[0]
    assert frame == b"jpeg-bytes"
    assert json.loads(message) == {
        "type": "detections",
        "image_size": [640, 480],
        "detections": [{"class": "whitefly", "confidence": 0.8, "bbox": [1.0, 2.0, 3.0, 4.0]}],
    }


def test_queued_detections_are_bounded_so_a_slow_link_cannot_fill_memory():
    service = make_service()
    for i in range(20):
        service.send_stream_detections([], (10, 10), f"frame-{i}".encode())
    assert len(service._stream_detections) == 8
    assert service._stream_detections[-1][1] == b"frame-19"  # newest kept, oldest dropped


def test_detections_without_a_frame_are_ignored():
    service = make_service()
    service.send_stream_detections([{"class": "x", "confidence": 1, "bbox": (0, 0, 1, 1)}], (10, 10), b"")
    assert len(service._stream_detections) == 0


def test_unreachable_server_does_not_block_startup_and_keeps_retrying():
    service = make_service()
    started = time.monotonic()
    assert service.connect_to_server(wait_timeout=0.5) is False
    assert time.monotonic() - started < 2
    # Still retrying in the background, so a server that comes up later is found.
    assert service._control_thread.is_alive()
    service.disconnect()
    assert not service._control_thread or not service._control_thread.is_alive()


def test_stream_channel_starts_without_waiting_and_switches_mode():
    service = make_service()
    assert service.start_stream_channel(mode="monitor") is False  # not connected (yet)
    assert service.get_stream_mode() == "monitor"
    service.start_stream_channel(mode="server")
    assert service.get_stream_mode() == "server"
    service.stop_stream_channel()
    assert service.get_stream_mode() is None


def test_rejected_robot_key_gets_a_clear_error(caplog):
    with caplog.at_level(logging.ERROR):
        NetworkService._log_session_error("Control", Exception("server rejected WebSocket connection: HTTP 403"))
    assert "ROBOT_KEY" in caplog.text
