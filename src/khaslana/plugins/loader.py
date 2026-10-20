"""Dynamic plugin discovery, validation, and error-isolated loading."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import json
import logging
from pathlib import Path
from typing import Any

from khaslana.plugins.builtin.api_tester import APITesterPlugin
from khaslana.plugins.builtin.db_query import DBQueryPlugin
from khaslana.plugins.builtin.docker_tool import DockerPlugin
from khaslana.plugins.models import Plugin, PluginMetadata
from khaslana.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

USER_PLUGINS_DIR = Path.home() / ".config" / "khaslana" / "plugins"
LOCAL_PLUGINS_DIR = Path(".khaslana") / "plugins"
PLUGINS_CONFIG_FILE = Path.home() / ".config" / "khaslana" / "plugins.json"


class PluginLoader:
    """Discovers, validates, and loads Khaslana plugins with full error isolation."""

    def __init__(
        self,
        user_plugins_dir: Path | None = None,
        local_plugins_dir: Path | None = None,
        config_file: Path | None = None,
    ) -> None:
        self.user_plugins_dir = user_plugins_dir or USER_PLUGINS_DIR
        self.local_plugins_dir = local_plugins_dir or LOCAL_PLUGINS_DIR
        self.config_file = config_file or PLUGINS_CONFIG_FILE

    def _load_disabled_list(self) -> set[str]:
        """Read set of disabled plugin names from configuration."""
        if not self.config_file.is_file():
            return set()
        try:
            data = json.loads(self.config_file.read_text(encoding="utf-8"))
            return set(data.get("disabled", []))
        except Exception:
            return set()

    def _save_disabled_list(self, disabled: set[str]) -> None:
        """Persist set of disabled plugin names to disk."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"disabled": sorted(disabled)}
        self.config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def is_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is currently enabled."""
        return plugin_name not in self._load_disabled_list()

    def enable(self, plugin_name: str) -> bool:
        """Enable a plugin by name."""
        disabled = self._load_disabled_list()
        if plugin_name in disabled:
            disabled.remove(plugin_name)
            self._save_disabled_list(disabled)
            return True
        return False

    def disable(self, plugin_name: str) -> bool:
        """Disable a plugin by name."""
        disabled = self._load_disabled_list()
        if plugin_name not in disabled:
            disabled.add(plugin_name)
            self._save_disabled_list(disabled)
            return True
        return False

    def _discover_builtin(self) -> list[Plugin]:
        """Instantiate standard built-in plugins."""
        return [
            DBQueryPlugin(),
            DockerPlugin(),
            APITesterPlugin(),
        ]

    def _discover_entrypoints(self) -> list[Plugin]:
        """Discover plugins registered via Python entry points ('khaslana.plugins')."""
        plugins: list[Plugin] = []
        try:
            selected_eps = importlib.metadata.entry_points(group="khaslana.plugins")

            for ep in selected_eps:
                try:
                    loaded = ep.load()
                    plugin_instance = self._wrap_loaded_plugin(
                        loaded, name=ep.name, source="entrypoint"
                    )
                    if plugin_instance:
                        plugins.append(plugin_instance)
                except Exception as e:
                    logger.warning("Failed to load entry point plugin '%s': %s", ep.name, e)
                    meta = PluginMetadata(
                        name=ep.name,
                        source="entrypoint",
                        error=f"Entry point load failed: {e}",
                    )
                    plugins.append(_ErrorPlugin(meta))
        except Exception as e:
            logger.debug("Entry points discovery encountered error: %s", e)

        return plugins

    def _discover_directory(self, directory: Path) -> list[Plugin]:
        """Discover Python files or folders inside a plugins directory."""
        if not directory.is_dir():
            return []

        plugins: list[Plugin] = []
        for path in sorted(directory.iterdir()):
            target_py: Path | None = None
            if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
                target_py = path
            elif path.is_dir() and not path.name.startswith((".", "_")):
                init_py = path / "__init__.py"
                plugin_py = path / "plugin.py"
                if init_py.is_file():
                    target_py = init_py
                elif plugin_py.is_file():
                    target_py = plugin_py

            if not target_py:
                continue

            plugin_name = path.stem if path.is_file() else path.name
            try:
                mod_name = f"khaslana_dynamic_plugin_{plugin_name}"
                spec = importlib.util.spec_from_file_location(mod_name, target_py)
                if not spec or not spec.loader:
                    raise ImportError(f"Cannot create spec for {target_py}")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                plugin_instance = self._wrap_loaded_plugin(
                    module, name=plugin_name, source="directory", path=str(target_py)
                )
                if plugin_instance:
                    plugins.append(plugin_instance)
                else:
                    meta = PluginMetadata(
                        name=plugin_name,
                        source="directory",
                        path=str(target_py),
                        error=(
                            "Module does not define a 'register(registry)' "
                            "function or 'Plugin' class."
                        ),
                    )
                    plugins.append(_ErrorPlugin(meta))
            except Exception as e:
                logger.warning("Error loading directory plugin '%s': %s", plugin_name, e)
                meta = PluginMetadata(
                    name=plugin_name,
                    source="directory",
                    path=str(target_py),
                    error=f"Import failed: {e}",
                )
                plugins.append(_ErrorPlugin(meta))

        return plugins

    def _wrap_loaded_plugin(
        self,
        obj: Any,
        name: str,
        source: Any = "directory",
        path: str | None = None,
    ) -> Plugin | None:
        """Convert module or class into a validated Plugin instance."""
        # 1. Direct Plugin instance
        if isinstance(obj, Plugin):
            obj.metadata.source = source
            obj.metadata.path = path
            return obj

        # 2. Plugin class
        if inspect.isclass(obj) and issubclass(obj, Plugin):
            inst = obj()
            inst.metadata.source = source
            inst.metadata.path = path
            return inst

        # 3. Look for Plugin subclass or 'plugin' attribute inside a module
        if inspect.ismodule(obj):
            if hasattr(obj, "plugin") and isinstance(obj.plugin, Plugin):
                obj.plugin.metadata.source = source
                obj.plugin.metadata.path = path
                return obj.plugin

            for _, val in inspect.getmembers(obj):
                if inspect.isclass(val) and issubclass(val, Plugin) and val is not Plugin:
                    inst = val()
                    inst.metadata.source = source
                    inst.metadata.path = path
                    return inst

            # 4. Check for standalone 'register(registry)' function
            if hasattr(obj, "register") and callable(obj.register):
                doc = inspect.getdoc(obj) or inspect.getdoc(obj.register) or ""
                meta = PluginMetadata(
                    name=name,
                    description=doc.split("\n")[0] if doc else "Custom directory plugin",
                    source=source,
                    path=path,
                )
                return _FunctionPlugin(meta, obj.register)

        return None

    def discover_all(self) -> list[Plugin]:
        """Discover all plugins across builtin, entrypoints, and configured directories."""
        disabled_set = self._load_disabled_list()
        plugins_map: dict[str, Plugin] = {}

        # 1. Builtin plugins
        for p in self._discover_builtin():
            p.metadata.enabled = p.metadata.name not in disabled_set
            plugins_map[p.metadata.name] = p

        # 2. Entrypoint plugins
        for p in self._discover_entrypoints():
            p.metadata.enabled = p.metadata.name not in disabled_set
            plugins_map[p.metadata.name] = p

        # 3. User config directory plugins (~/.config/khaslana/plugins)
        for p in self._discover_directory(self.user_plugins_dir):
            p.metadata.enabled = p.metadata.name not in disabled_set
            plugins_map[p.metadata.name] = p

        # 4. Local workspace plugins (.khaslana/plugins)
        for p in self._discover_directory(self.local_plugins_dir):
            p.metadata.enabled = p.metadata.name not in disabled_set
            plugins_map[p.metadata.name] = p

        return list(plugins_map.values())

    def get_metadata_list(self) -> list[PluginMetadata]:
        """Retrieve metadata summary for all discovered plugins."""
        plugins = self.discover_all()
        # Pre-populate registered tools list by dry-running register in a dummy registry
        for p in plugins:
            if not p.metadata.tools and not p.metadata.error:
                dummy = ToolRegistry()
                try:
                    p.register(dummy)
                    p.metadata.tools = [t.name for t in dummy.list_tools()]
                except Exception as e:
                    p.metadata.error = f"Registration error: {e}"
        return [p.metadata for p in plugins]

    def get_plugin(self, name: str) -> Plugin | None:
        """Find a single plugin by name."""
        for p in self.discover_all():
            if p.metadata.name == name:
                return p
        return None

    def register_enabled_tools(self, registry: ToolRegistry) -> None:
        """Register all active, enabled plugins into the provided ToolRegistry."""
        for p in self.discover_all():
            if p.metadata.enabled and not p.metadata.error:
                try:
                    p.register(registry)
                except Exception as e:
                    logger.warning(
                        "Error registering tools for plugin '%s': %s", p.metadata.name, e
                    )


class _FunctionPlugin(Plugin):
    """Wrapper adapting a standalone 'register(registry)' function as a Plugin."""

    def __init__(self, metadata: PluginMetadata, register_fn: Any) -> None:
        self.metadata = metadata
        self.register_fn = register_fn

    def register(self, registry: ToolRegistry) -> None:
        self.register_fn(registry)


class _ErrorPlugin(Plugin):
    """Placeholder plugin recording a failure during discovery."""

    def __init__(self, metadata: PluginMetadata) -> None:
        self.metadata = metadata

    def register(self, registry: ToolRegistry) -> None:
        pass
