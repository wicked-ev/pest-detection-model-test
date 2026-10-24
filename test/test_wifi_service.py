"""Wi-Fi commands are built correctly, without running nmcli."""

import json
import subprocess

import pytest

import services.wifi_service as wifi_service
from services.wifi_service import WiFiCredentials, WiFiManager


class FakeRun:
    """Records commands instead of running them."""

    def __init__(self, returncode=0, stdout="", missing=()):
        self.commands = []
        self.returncode = returncode
        self.stdout = stdout
        self.missing = set(missing)

    def __call__(self, cmd, timeout):
        self.commands.append(cmd)
        if cmd[0] in self.missing:
            return None
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.stdout, stderr="")


@pytest.fixture
def manager(tmp_path):
    return WiFiManager(credential_path=tmp_path / "wifi.json")


def test_connect_uses_the_password_keyword(manager, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(wifi_service, "_run", fake)
    assert manager.connect([WiFiCredentials("Farm", "secret123")])
    cmd = fake.commands[0]
    # nmcli requires "connect <ssid> password <password>"; the bare password was rejected.
    assert cmd[cmd.index("connect") + 1:cmd.index("connect") + 4] == ["Farm", "password", "secret123"]


def test_open_network_has_no_password_argument(manager, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(wifi_service, "_run", fake)
    manager.connect([WiFiCredentials("Open", "")])
    assert "password" not in fake.commands[0]


def test_failed_connection_tries_the_next_network(manager, monkeypatch):
    fake = FakeRun(returncode=10)
    monkeypatch.setattr(wifi_service, "_run", fake)
    assert not manager.connect([WiFiCredentials("A", "secret123"), WiFiCredentials("B", "secret123")])
    assert len(fake.commands) == 2


def test_saved_credentials_accept_a_single_object(manager):
    # The file in the repository held one object, which used to be rejected.
    manager.credentials_path.write_text(json.dumps({"ssid": "Farm", "password": "secret123"}))
    assert manager.load_saved_credentials() == [WiFiCredentials("Farm", "secret123")]


def test_saved_credentials_accept_a_list(manager):
    manager.save_credentials(WiFiCredentials("Farm", "secret123"))
    assert manager.load_saved_credentials() == [WiFiCredentials("Farm", "secret123")]


def test_stopping_the_hotspot_does_not_disconnect_wifi(manager, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(wifi_service, "_run", fake)
    manager.stop_hotspot()
    assert fake.commands[0][:3] == ["nmcli", "connection", "down"]
    assert "disconnect" not in fake.commands[0]


def test_hotspot_uses_configured_name_and_rejects_short_password(manager, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(wifi_service, "_run", fake)
    monkeypatch.setattr(wifi_service.configs, "HOTSPOT_SSID", "pest-robot-setup")
    monkeypatch.setattr(wifi_service.configs, "HOTSPOT_PASSWORD", "short")
    assert manager.start_hotspot() is False
    assert fake.commands == []

    monkeypatch.setattr(wifi_service.configs, "HOTSPOT_PASSWORD", "long-enough")
    assert manager.start_hotspot() is True
    assert "pest-robot-setup" in fake.commands[0]


def test_hotspot_failure_is_reported(manager, monkeypatch):
    monkeypatch.setattr(wifi_service, "_run", FakeRun(returncode=4))
    monkeypatch.setattr(wifi_service.configs, "HOTSPOT_PASSWORD", "long-enough")
    assert manager.start_hotspot() is False


def test_missing_network_tools_mean_not_connected_instead_of_a_crash(manager, monkeypatch):
    monkeypatch.setattr(wifi_service, "_run", FakeRun(missing={"iwgetid", "nmcli"}))
    assert manager.is_wifi_connected() is False


def test_connected_state_falls_back_to_nmcli(manager, monkeypatch):
    fake = FakeRun(stdout="GENERAL.STATE:100 (connected)\n", missing={"iwgetid"})
    monkeypatch.setattr(wifi_service, "_run", fake)
    assert manager.is_wifi_connected() is True
