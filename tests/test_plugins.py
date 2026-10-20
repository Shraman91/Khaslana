"""Tests for Phase 9: Plugin system, dynamic discovery, and built-in tools."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from khaslana.cli.main import app
from khaslana.plugins.builtin.api_tester import APITesterInput, APITesterPlugin, APITesterTool
from khaslana.plugins.builtin.db_query import DBQueryInput, DBQueryPlugin, DBQueryTool
from khaslana.plugins.builtin.docker_tool import DockerPlugin, DockerTool, DockerToolInput
from khaslana.plugins.loader import PluginLoader
from khaslana.tools.base import ToolRegistry

# ============================================================================
# 1. Built-in Tool Tests: db-query
# ============================================================================


def test_db_query_select_success(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, role TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 'admin')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 'dev')")
    conn.commit()
    conn.close()

    tool = DBQueryTool()
    res = tool.execute(
        DBQueryInput(
            connection_string=str(db_file),
            query="SELECT id, name, role FROM users ORDER BY id ASC",
        )
    )

    assert res.ok is True
    assert "Alice" in res.output
    assert "Bob" in res.output
    assert res.metadata["row_count"] == 2
    assert "name" in res.metadata["columns"]


def test_db_query_sqlite_uri_prefix(tmp_path: Path) -> None:
    db_file = tmp_path / "test_uri.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE items (id INT, val TEXT)")
    conn.execute("INSERT INTO items VALUES (10, 'Widget')")
    conn.commit()
    conn.close()

    tool = DBQueryTool()
    res = tool.execute(
        DBQueryInput(
            connection_string=f"sqlite:///{db_file.as_posix()}",
            query="SELECT * FROM items",
        )
    )
    assert res.ok is True
    assert "Widget" in res.output


def test_db_query_blocks_mutating_statements(tmp_path: Path) -> None:
    db_file = tmp_path / "safe.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    tool = DBQueryTool()

    # INSERT
    res_insert = tool.execute(
        DBQueryInput(
            connection_string=str(db_file),
            query="INSERT INTO users VALUES (1, 'Attacker')",
        )
    )
    assert res_insert.ok is False
    assert "Mutating SQL command rejected" in (res_insert.error or "")

    # DROP
    res_drop = tool.execute(
        DBQueryInput(
            connection_string=str(db_file),
            query="DROP TABLE users",
        )
    )
    assert res_drop.ok is False
    assert "Mutating SQL command rejected" in (res_drop.error or "")

    # DELETE
    res_delete = tool.execute(
        DBQueryInput(
            connection_string=str(db_file),
            query="DELETE FROM users WHERE id = 1",
        )
    )
    assert res_delete.ok is False
    assert "Mutating SQL command rejected" in (res_delete.error or "")


def test_db_query_missing_db_file(tmp_path: Path) -> None:
    tool = DBQueryTool()
    res = tool.execute(
        DBQueryInput(
            connection_string=str(tmp_path / "nonexistent.db"),
            query="SELECT 1",
        )
    )
    assert res.ok is False
    assert "Database file not found" in (res.error or "")


def test_db_query_postgres_notice() -> None:
    tool = DBQueryTool()
    res = tool.execute(
        DBQueryInput(
            connection_string="postgresql://user:pass@localhost:5432/mydb",
            query="SELECT 1",
        )
    )
    assert res.ok is False
    assert "PostgreSQL query execution requires 'psycopg2'" in (res.error or "")


def test_db_query_plugin_registration() -> None:
    plugin = DBQueryPlugin()
    registry = ToolRegistry()
    plugin.register(registry)
    tool = registry.get("db_query")
    assert tool is not None
    assert isinstance(tool, DBQueryTool)


# ============================================================================
# 2. Built-in Tool Tests: docker
# ============================================================================


def test_docker_tool_graceful_failure_no_daemon() -> None:
    tool = DockerTool()
    with patch("docker.from_env", side_effect=Exception("Connection refused")):
        res = tool.execute(DockerToolInput(action="list"))
        assert res.ok is False
        assert "Docker is unavailable" in (res.error or "")


def test_docker_tool_list_mocked() -> None:
    tool = DockerTool()
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_c1 = MagicMock()
    mock_c1.short_id = "abc1234"
    mock_c1.image.tags = ["my-app:v1"]
    mock_c1.status = "running"
    mock_c1.name = "web_server"

    mock_client.containers.list.return_value = [mock_c1]

    with patch("docker.from_env", return_value=mock_client):
        res = tool.execute(DockerToolInput(action="list"))
        assert res.ok is True
        assert "abc1234" in res.output
        assert "web_server" in res.output
        assert res.metadata["container_count"] == 1


def test_docker_tool_logs_and_inspect_mocked() -> None:
    tool = DockerTool()
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_container.id = "full_id_123"
    mock_container.name = "db_postgres"
    mock_container.status = "running"
    mock_container.image = "postgres:16"
    mock_container.attrs = {"Created": "2026-10-01", "State": {"Running": True}}
    mock_container.logs.return_value = (
        b"PostgreSQL database server initialized.\nReady to accept connections.\n"
    )

    mock_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_client):
        # Test logs
        res_logs = tool.execute(DockerToolInput(action="logs", container="db_postgres", tail=10))
        assert res_logs.ok is True
        assert "database server initialized" in res_logs.output

        # Test inspect
        res_inspect = tool.execute(DockerToolInput(action="inspect", container="db_postgres"))
        assert res_inspect.ok is True
        assert "full_id_123" in res_inspect.output
        assert "postgres:16" in res_inspect.output

        # Missing container arg
        res_err = tool.execute(DockerToolInput(action="logs", container=None))
        assert res_err.ok is False
        assert "Parameter 'container' is required" in (res_err.error or "")


def test_docker_plugin_registration() -> None:
    plugin = DockerPlugin()
    registry = ToolRegistry()
    plugin.register(registry)
    tool = registry.get("docker_inspect")
    assert tool is not None
    assert isinstance(tool, DockerTool)


# ============================================================================
# 3. Built-in Tool Tests: api-tester
# ============================================================================


def test_api_tester_success() -> None:
    tool = APITesterTool()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"status": "healthy", "uptime": 3600}'
    mock_response.json.return_value = {"status": "healthy", "uptime": 3600}
    mock_response.headers = {"content-type": "application/json"}

    with patch("httpx.Client.request", return_value=mock_response):
        res = tool.execute(
            APITesterInput(
                url="https://api.example.com/health",
                method="GET",
                expected_status=200,
            )
        )
        assert res.ok is True
        assert "Status: 200 (Expected: 200)" in res.output
        assert "healthy" in res.output
        assert res.metadata["status_code"] == 200


def test_api_tester_status_mismatch() -> None:
    tool = APITesterTool()

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_response.json.side_effect = Exception("Not json")
    mock_response.headers = {"content-type": "text/plain"}

    with patch("httpx.Client.request", return_value=mock_response):
        res = tool.execute(
            APITesterInput(
                url="https://api.example.com/missing",
                method="GET",
                expected_status=200,
            )
        )
        assert res.ok is False
        assert "Assertion failed: expected HTTP 200, got HTTP 404" in (res.error or "")


def test_api_tester_request_exception() -> None:
    tool = APITesterTool()

    with patch("httpx.Client.request", side_effect=Exception("Connection timed out")):
        res = tool.execute(
            APITesterInput(
                url="https://api.example.com/timeout",
                method="GET",
            )
        )
        assert res.ok is False
        assert "Connection timed out" in (res.error or "")


def test_api_tester_plugin_registration() -> None:
    plugin = APITesterPlugin()
    registry = ToolRegistry()
    plugin.register(registry)
    tool = registry.get("api_test")
    assert tool is not None
    assert isinstance(tool, APITesterTool)


# ============================================================================
# 4. PluginLoader Tests (Discovery, Validation, Isolation, Toggles)
# ============================================================================


def test_plugin_loader_discovers_builtins(tmp_path: Path) -> None:
    config_file = tmp_path / "plugins.json"
    loader = PluginLoader(
        user_plugins_dir=tmp_path / "user_plugins",
        local_plugins_dir=tmp_path / "local_plugins",
        config_file=config_file,
    )

    plugins = loader.discover_all()
    names = {p.metadata.name for p in plugins}
    assert "db-query" in names
    assert "docker" in names
    assert "api-tester" in names

    meta_list = loader.get_metadata_list()
    assert len(meta_list) >= 3


def test_plugin_loader_enable_disable_toggle(tmp_path: Path) -> None:
    config_file = tmp_path / "plugins.json"
    loader = PluginLoader(
        user_plugins_dir=tmp_path / "user_plugins",
        local_plugins_dir=tmp_path / "local_plugins",
        config_file=config_file,
    )

    assert loader.is_enabled("db-query") is True

    # Disable db-query
    loader.disable("db-query")
    assert loader.is_enabled("db-query") is False

    # Check that metadata reflects disabled status
    meta = next(m for m in loader.get_metadata_list() if m.name == "db-query")
    assert meta.enabled is False

    # Re-enable
    loader.enable("db-query")
    assert loader.is_enabled("db-query") is True


def test_plugin_loader_custom_directory_plugin_class(tmp_path: Path) -> None:
    local_plugins = tmp_path / "plugins"
    local_plugins.mkdir()

    # Create a custom plugin file defining a Plugin subclass
    custom_py = local_plugins / "custom_calc.py"
    custom_py.write_text(
        """
from khaslana.plugins.models import Plugin, PluginMetadata
from khaslana.tools.base import Tool, ToolResult
from pydantic import BaseModel

class DummyCalcTool(Tool):
    name = "dummy_calc"
    description = "A dummy calculator tool"
    input_schema = BaseModel

    def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output="42")

class CustomCalcPlugin(Plugin):
    metadata = PluginMetadata(
        name="custom-calc",
        version="1.0.0",
        description="Math utilities",
        tools=["dummy_calc"]
    )

    def register(self, registry):
        registry.register(DummyCalcTool())
""",
        encoding="utf-8",
    )

    loader = PluginLoader(
        user_plugins_dir=tmp_path / "empty_user",
        local_plugins_dir=local_plugins,
        config_file=tmp_path / "plugins.json",
    )

    plugin = loader.get_plugin("custom-calc")
    assert plugin is not None
    assert plugin.metadata.name == "custom-calc"
    assert plugin.metadata.version == "1.0.0"

    registry = ToolRegistry()
    loader.register_enabled_tools(registry)
    assert registry.get("dummy_calc") is not None


def test_plugin_loader_custom_register_function(tmp_path: Path) -> None:
    local_plugins = tmp_path / "plugins"
    local_plugins.mkdir()

    # Create a custom plugin file with standalone register() function
    custom_py = local_plugins / "fn_plugin.py"
    custom_py.write_text(
        """
from khaslana.tools.base import Tool, ToolResult
from pydantic import BaseModel

class EchoTool(Tool):
    name = "echo_tool"
    description = "Echo tool"
    input_schema = BaseModel

    def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output="echo")

def register(registry):
    '''Simple register function plugin.'''
    registry.register(EchoTool())
""",
        encoding="utf-8",
    )

    loader = PluginLoader(
        user_plugins_dir=tmp_path / "empty_user",
        local_plugins_dir=local_plugins,
        config_file=tmp_path / "plugins.json",
    )

    plugin = loader.get_plugin("fn_plugin")
    assert plugin is not None
    assert plugin.metadata.name == "fn_plugin"

    registry = ToolRegistry()
    loader.register_enabled_tools(registry)
    assert registry.get("echo_tool") is not None


def test_plugin_loader_error_isolation_on_broken_file(tmp_path: Path) -> None:
    local_plugins = tmp_path / "plugins"
    local_plugins.mkdir()

    # Create a broken plugin file that has syntax/import errors
    broken_py = local_plugins / "broken_plugin.py"
    broken_py.write_text("this is completely invalid python syntax :::", encoding="utf-8")

    loader = PluginLoader(
        user_plugins_dir=tmp_path / "empty_user",
        local_plugins_dir=local_plugins,
        config_file=tmp_path / "plugins.json",
    )

    # Loader should NOT raise exception
    plugins = loader.discover_all()
    broken = next((p for p in plugins if p.metadata.name == "broken_plugin"), None)
    assert broken is not None
    assert broken.metadata.error is not None
    assert "Import failed" in broken.metadata.error

    # Other plugins still work fine
    assert any(p.metadata.name == "db-query" for p in plugins)


# ============================================================================
# 5. CLI Commands Tests: khaslana plugins list|info|enable|disable
# ============================================================================


def test_cli_plugins_list() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    assert "db-query" in result.stdout
    assert "docker" in result.stdout
    assert "api-tester" in result.stdout


def test_cli_plugins_info() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plugins", "info", "db-query"])
    assert result.exit_code == 0
    assert "Plugin: db-query" in result.stdout
    assert "db_query" in result.stdout
    assert "connection_string" in result.stdout
    assert "query" in result.stdout


def test_cli_plugins_info_not_found() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plugins", "info", "nonexistent-plugin"])
    assert result.exit_code != 0
    assert "not found" in result.stdout


def test_cli_plugins_disable_and_enable(tmp_path: Path) -> None:
    runner = CliRunner()

    # Test disable
    result_disable = runner.invoke(app, ["plugins", "disable", "db-query"])
    assert result_disable.exit_code == 0
    assert "disabled" in result_disable.stdout.lower()

    # Test enable
    result_enable = runner.invoke(app, ["plugins", "enable", "db-query"])
    assert result_enable.exit_code == 0
    assert "enabled successfully" in result_enable.stdout.lower()
