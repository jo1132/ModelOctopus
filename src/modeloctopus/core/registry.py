"""Discovery stores entry points without importing their backend modules."""

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from pydantic import BaseModel

from .schema import CapabilityReport, ModelOctopusError, Schema


class PluginDescriptor(Schema):
    id: str
    api_version: int = 1
    config_schema: type[BaseModel]
    capabilities: CapabilityReport


class Registry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[PluginDescriptor, Callable[[], Any]]] = {}
        self._loaded: dict[str, Any] = {}

    def register(self, descriptor: PluginDescriptor, loader: Callable[[], Any]) -> None:
        if descriptor.api_version != 1:
            raise ModelOctopusError("PLUGIN_API_MISMATCH", descriptor.id)
        if descriptor.id in self._entries:
            raise ModelOctopusError("DUPLICATE_PLUGIN", descriptor.id)
        self._entries[descriptor.id] = descriptor, loader

    def discover(self, group: str) -> None:
        # Entry points expose a lightweight descriptor factory, never a backend factory.
        for entry in entry_points(group=f"modeloctopus.{group}"):
            descriptor, loader = entry.load()()
            self.register(descriptor, loader)

    def describe(self, plugin_id: str) -> PluginDescriptor:
        if plugin_id not in self._entries:
            raise ModelOctopusError("UNKNOWN_PLUGIN", plugin_id)
        return self._entries[plugin_id][0]

    def load(self, plugin_id: str, config: dict[str, Any]) -> Any:
        descriptor = self.describe(plugin_id)
        descriptor.config_schema.model_validate(config)
        if plugin_id not in self._loaded:
            try:
                self._loaded[plugin_id] = self._entries[plugin_id][1]()
            except ImportError as exc:
                raise ModelOctopusError("DEPENDENCY_NOT_INSTALLED", str(exc)) from exc
        return self._loaded[plugin_id]
