# tests for the daemon's fleet-discovery route installation (issue #9). the
# daemon is bare-metal and off the docker network, so it installs its own
# traefik file-provider route. these tests pin: correct watched directory,
# valid route content, idempotency, and best-effort (never-raises) behaviour.
import pytest

from paling import discovery


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PALING_TRAEFIK_DYNAMIC_DIR", raising=False)
    monkeypatch.delenv("PALING_VAR", raising=False)
    return tmp_path


def test_dynamic_dir_is_traefik_watched_path(fake_home):
    d = discovery._dynamic_dir(prefix=fake_home)
    # the fleet traefik watches ${PREFIX}/var/traefik/dynamic; the route lands there.
    assert d == fake_home / "var" / "traefik" / "dynamic"


def test_dynamic_dir_defaults_to_home(fake_home):
    # with no prefix arg and no env, PREFIX defaults to ${HOME}.
    assert discovery._dynamic_dir() == fake_home / "var" / "traefik" / "dynamic"


def test_dynamic_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PALING_TRAEFIK_DYNAMIC_DIR", str(tmp_path / "custom"))
    assert discovery._dynamic_dir() == tmp_path / "custom"


def test_dynamic_dir_paling_var_prefix(monkeypatch, tmp_path):
    # PALING_VAR sets the PREFIX; var/traefik/dynamic hangs off it. it wins over
    # the prefix arg so an operator can relocate "var" without code changes.
    monkeypatch.delenv("PALING_TRAEFIK_DYNAMIC_DIR", raising=False)
    monkeypatch.setenv("PALING_VAR", str(tmp_path / "elsewhere"))
    assert discovery._dynamic_dir(prefix=tmp_path / "ignored") == (
        tmp_path / "elsewhere" / "var" / "traefik" / "dynamic"
    )


def test_install_writes_route_file(fake_home):
    result = discovery.install_daemon_route(prefix=fake_home)
    target = fake_home / "var" / "traefik" / "dynamic" / "paling.yml"
    assert result.installed is True
    assert result.path == str(target)
    body = target.read_text()
    # the route must point traefik at the bare-metal daemon and match the host.
    assert "paling-daemon.local" in body
    assert "host.docker.internal:8090" in body
    assert "loadBalancer" in body


def test_install_is_idempotent(fake_home):
    first = discovery.install_daemon_route(prefix=fake_home)
    second = discovery.install_daemon_route(prefix=fake_home)
    assert first.installed and second.installed
    target = fake_home / "var" / "traefik" / "dynamic" / "paling.yml"
    # re-running overwrites in place; content is stable.
    assert target.read_text() == discovery._render_traefik_dynamic_config(
        discovery._DEFAULT_DAEMON_HOST_RULE, discovery._DEFAULT_DAEMON_UPSTREAM
    )


def test_install_custom_rule_and_upstream(fake_home):
    result = discovery.install_daemon_route(
        prefix=fake_home, upstream="http://10.0.0.5:9999", rule="paling-dev.local"
    )
    body = (fake_home / "var" / "traefik" / "dynamic" / "paling.yml").read_text()
    assert result.rule == "paling-dev.local"
    assert "paling-dev.local" in body
    assert "10.0.0.5:9999" in body


def test_install_is_best_effort_on_oserror(fake_home, monkeypatch):
    # a write failure must be swallowed: discovery never blocks the daemon.
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(discovery.Path, "mkdir", boom)
    result = discovery.install_daemon_route(prefix=fake_home)
    assert result.installed is False
    assert "read-only" in (result.message or "")
