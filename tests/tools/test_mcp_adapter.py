"""Tests for the MCP tool adapter — round-trip through MCPServer + MCPClient."""

from __future__ import annotations

import pytest

from openjarvis.mcp.client import MCPClient
from openjarvis.mcp.server import MCPServer
from openjarvis.mcp.transport import InProcessTransport
from openjarvis.tools._stubs import ToolSpec
from openjarvis.tools.calculator import CalculatorTool
from openjarvis.tools.mcp_adapter import MCPToolAdapter, MCPToolProvider
from openjarvis.tools.think import ThinkTool


@pytest.fixture
def server():
    """Create an MCP server with calculator and think tools."""
    return MCPServer([CalculatorTool(), ThinkTool()])


@pytest.fixture
def client(server):
    """Create an MCP client connected to the server via in-process transport."""
    transport = InProcessTransport(server)
    c = MCPClient(transport)
    c.initialize()
    return c


class TestMCPToolAdapter:
    def test_adapter_spec(self, client):
        spec = ToolSpec(
            name="calculator",
            description="Evaluate a math expression",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
        )
        adapter = MCPToolAdapter(client, spec)
        assert adapter.spec.name == "calculator"
        assert adapter.spec.description == "Evaluate a math expression"

    def test_adapter_execute_success(self, client):
        spec = ToolSpec(
            name="calculator",
            description="Calculator",
            parameters={},
        )
        adapter = MCPToolAdapter(client, spec)
        result = adapter.execute(expression="2+2")
        assert result.success is True
        assert "4" in result.content
        assert result.tool_name == "calculator"

    def test_adapter_execute_think(self, client):
        spec = ToolSpec(
            name="think",
            description="Think tool",
            parameters={},
        )
        adapter = MCPToolAdapter(client, spec)
        result = adapter.execute(thought="Let me analyze this.")
        assert result.success is True
        assert "analyze" in result.content

    def test_adapter_execute_error(self, client):
        spec = ToolSpec(
            name="calculator",
            description="Calculator",
            parameters={},
        )
        adapter = MCPToolAdapter(client, spec)
        result = adapter.execute(expression="1/0")
        assert result.success is True
        assert result.content == "inf"

    def test_adapter_unknown_tool(self, client):
        spec = ToolSpec(
            name="nonexistent",
            description="Does not exist",
            parameters={},
        )
        adapter = MCPToolAdapter(client, spec)
        result = adapter.execute(x="y")
        assert result.success is False
        assert "error" in result.content.lower() or "MCP" in result.content

    def test_adapter_tool_id(self, client):
        spec = ToolSpec(name="calculator", description="Calc", parameters={})
        adapter = MCPToolAdapter(client, spec)
        assert adapter.tool_id == "mcp_adapter"


class TestMCPToolProvider:
    def test_discover_returns_adapters(self, client):
        provider = MCPToolProvider(client)
        tools = provider.discover()
        assert len(tools) == 2
        names = {t.spec.name for t in tools}
        assert "calculator" in names
        assert "think" in names

    def test_discovered_tools_are_base_tool(self, client):
        from openjarvis.tools._stubs import BaseTool

        provider = MCPToolProvider(client)
        tools = provider.discover()
        for tool in tools:
            assert isinstance(tool, BaseTool)

    def test_discovered_tool_execution(self, client):
        provider = MCPToolProvider(client)
        tools = provider.discover()
        calc = next(t for t in tools if t.spec.name == "calculator")
        result = calc.execute(expression="3*7")
        assert result.success is True
        assert "21" in result.content


class TestMCPAdapterRoundTrip:
    """End-to-end: Server → InProcessTransport → Client → Adapter → execute."""

    def test_full_round_trip(self):
        server = MCPServer([CalculatorTool(), ThinkTool()])
        transport = InProcessTransport(server)
        client = MCPClient(transport)
        client.initialize()
        provider = MCPToolProvider(client)
        tools = provider.discover()

        calc = next(t for t in tools if t.spec.name == "calculator")
        result = calc.execute(expression="10+20")
        assert result.success is True
        assert "30" in result.content

    def test_round_trip_error_handling(self):
        server = MCPServer([CalculatorTool()])
        transport = InProcessTransport(server)
        client = MCPClient(transport)
        client.initialize()
        provider = MCPToolProvider(client)
        tools = provider.discover()

        calc = tools[0]
        result = calc.execute(expression="1/0")
        assert result.success is True
        assert result.content == "inf"

    def test_empty_server_discover(self):
        server = MCPServer([])
        transport = InProcessTransport(server)
        client = MCPClient(transport)
        client.initialize()
        provider = MCPToolProvider(client)
        tools = provider.discover()
        assert tools == []


class TestHideParams:
    """``hide_params`` removes filter parameters a model keeps misusing."""

    def _spec(self):
        return ToolSpec(
            name="calculator",
            description="Calculator",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "precision": {"type": "integer"},
                },
                "required": ["expression", "precision"],
            },
        )

    def test_hidden_param_removed_from_schema(self, client):
        adapter = MCPToolAdapter(client, self._spec(), hidden_params=["precision"])
        assert set(adapter.spec.parameters["properties"]) == {"expression"}
        assert adapter.spec.parameters["required"] == ["expression"]

    def test_original_spec_not_mutated(self, client):
        spec = self._spec()
        MCPToolAdapter(client, spec, hidden_params=["precision"])
        assert "precision" in spec.parameters["properties"]

    def test_hidden_param_stripped_from_call(self, client):
        sent = {}
        client.call_tool = lambda name, params: (
            sent.update(params) or {"content": [{"type": "text", "text": "ok"}]}
        )
        adapter = MCPToolAdapter(client, self._spec(), hidden_params=["precision"])
        assert adapter.execute(expression="2+2", precision=3).success
        assert sent == {"expression": "2+2"}

    def test_provider_applies_hide_params_wildcard(self, client):
        tools = MCPToolProvider(client).discover(hide_params={"calculator": "*"})
        calc = next(t for t in tools if t.spec.name == "calculator")
        think = next(t for t in tools if t.spec.name == "think")
        assert calc.spec.parameters.get("properties") == {}
        assert think.spec.parameters.get("properties")

    def test_no_config_is_unchanged(self, client):
        assert len(MCPToolProvider(client).discover()) == len(client.list_tools())
