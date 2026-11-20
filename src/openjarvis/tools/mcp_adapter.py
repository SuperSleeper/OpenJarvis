"""MCP tool adapter — wraps external MCP server tools as native BaseTool instances."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Iterable, List, Mapping

from openjarvis.core.types import ToolResult
from openjarvis.mcp.client import MCPClient
from openjarvis.tools._stubs import BaseTool, ToolSpec


class MCPToolAdapter(BaseTool):
    """Wraps a single MCP-hosted tool as a native BaseTool.

    This adapter enables tools discovered from external MCP servers to
    be used seamlessly within OpenJarvis agents via the ``ToolExecutor``.

    Parameters
    ----------
    client:
        The ``MCPClient`` connected to the external MCP server.
    tool_spec:
        The ``ToolSpec`` describing this tool (from ``MCPClient.list_tools()``).
    """

    tool_id = "mcp_adapter"

    def __init__(
        self,
        client: MCPClient,
        tool_spec: ToolSpec,
        hidden_params: Iterable[str] = (),
    ) -> None:
        self._client = client
        self._hidden = frozenset(hidden_params)
        self._spec = (
            _without_params(tool_spec, self._hidden) if self._hidden else tool_spec
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, **params: Any) -> ToolResult:
        """Execute the remote MCP tool and return a ToolResult."""
        if self._hidden:
            # Models sometimes send a hidden parameter anyway (it may still
            # appear in the tool description); never forward it.
            params = {k: v for k, v in params.items() if k not in self._hidden}
        try:
            result = self._client.call_tool(self._spec.name, params)
            content_parts = result.get("content", [])
            text = "\n".join(
                p.get("text", "") for p in content_parts if isinstance(p, dict)
            )
            return ToolResult(
                tool_name=self._spec.name,
                content=text,
                success=not result.get("isError", False),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self._spec.name,
                content=f"MCP tool error: {exc}",
                success=False,
            )


class MCPToolProvider:
    """Discovers tools from an MCP server and returns BaseTool adapters.

    Parameters
    ----------
    client:
        The ``MCPClient`` connected to the MCP server.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    def discover(self, hide_params: Mapping[str, Any] | None = None) -> List[BaseTool]:
        """Discover available tools and return them as BaseTool adapters.

        *hide_params* is a server's ``hide_params`` config: a mapping of tool
        name to the parameter names to hide from the model, or ``"*"`` for
        all of them. Use it for optional filter parameters a model misuses.
        """
        hide = hide_params or {}
        return [
            MCPToolAdapter(self._client, s, _hidden_params_for(s, hide))
            for s in self._client.list_tools()
        ]


def _hidden_params_for(spec: ToolSpec, hide: Mapping[str, Any]) -> frozenset[str]:
    """Resolve a ``hide_params`` entry for *spec*.

    ``hide_params`` maps a tool name to a list of parameter names to remove
    from the schema the model sees, or to ``"*"`` to remove all of them.
    """
    entry = hide.get(spec.name)
    if not entry:
        return frozenset()
    if entry == "*" or entry == ["*"]:
        return frozenset(spec.parameters.get("properties", {}))
    return frozenset(entry)


def _without_params(spec: ToolSpec, hidden: frozenset[str]) -> ToolSpec:
    params = copy.deepcopy(spec.parameters)
    props = params.get("properties")
    if isinstance(props, dict):
        for name in hidden:
            props.pop(name, None)
    if isinstance(params.get("required"), list):
        params["required"] = [r for r in params["required"] if r not in hidden]
    return replace(spec, parameters=params)


__all__ = ["MCPToolAdapter", "MCPToolProvider"]
