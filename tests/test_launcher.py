import io
import json

from nwm.__main__ import _local_app_ready, _open_app_when_ready, main


def test_app_launcher_reuses_running_server_and_opens_browser(tmp_path, monkeypatch):
    settings = tmp_path / "app.json"
    settings.write_text(json.dumps({"root": str(tmp_path / "storage")}))
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs:
                        io.BytesIO(b'{"status":"ok","worker":{"state":"idle"}}'))
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Duplicate server")))
    main(["serve", "--config", str(settings), "--open-browser"])
    assert opened == ["http://127.0.0.1:8000"]
    assert not (tmp_path / "storage").exists()


def test_app_opens_only_after_the_health_check_passes(monkeypatch):
    attempts = iter([False, True])
    monkeypatch.setattr("nwm.__main__._local_app_ready", lambda url: next(attempts))
    monkeypatch.setattr("time.sleep", lambda _: None)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    _open_app_when_ready("http://127.0.0.1:8000")
    assert opened == ["http://127.0.0.1:8000"]
