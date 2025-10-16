"""Protocol definition for MCP server interface."""

from typing import Any, Callable, Protocol


class MCPServer(Protocol):
    """Protocol for MCP server tool registration.

    This protocol defines the interface that MCP servers must implement
    for tool registration. It allows for type-safe tool definitions while
    remaining decoupled from specific MCP implementations.
    """

    def tool(
        self,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        annotations: Any = None,
        icons: Any = None,
        structured_output: bool | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a tool with the MCP server.

        Args:
        ----
            name: Optional tool name.
            title: Optional tool title.
            description: Optional description of the tool's functionality.
            annotations: Optional tool annotations.
            icons: Optional tool icons.
            structured_output: Optional structured output flag.

        Returns:
        -------
            Callable: Decorator function that registers the tool.

        """
        ...
