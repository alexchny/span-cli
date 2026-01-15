from typing import Any

from span.models.tools import ApplyPatchResult, ToolResult
from span.tools.base import Tool


def test_tool_result_success() -> None:
    result = ToolResult(success=True, output="File read successfully")

    assert result.success is True
    assert result.output == "File read successfully"
    assert result.error is None


def test_tool_result_failure() -> None:
    result = ToolResult(success=False, output="", error="File not found")

    assert result.success is False
    assert result.error == "File not found"


def test_tool_result_to_content_success() -> None:
    result = ToolResult(success=True, output="Hello world")
    content = result.to_content()

    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Hello world"


def test_tool_result_to_content_failure() -> None:
    result = ToolResult(success=False, output="Details here", error="Something failed")
    content = result.to_content()

    assert len(content) == 1
    assert "Error: Something failed" in content[0]["text"]
    assert "Details here" in content[0]["text"]


def test_apply_patch_result() -> None:
    result = ApplyPatchResult(
        success=True,
        output="Patch applied",
        file_path="test.py",
        reverse_diff="--- a/test.py\n+++ b/test.py",
    )

    assert result.success is True
    assert result.file_path == "test.py"
    assert result.reverse_diff is not None


class MockTool(Tool):
    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "input": {
                "type": "string",
                "description": "Input string",
                "required": True,
            },
            "optional": {
                "type": "boolean",
                "description": "Optional flag",
                "required": False,
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="Executed")


def test_tool_to_anthropic_schema() -> None:
    tool = MockTool()
    schema = tool.to_anthropic_tool()

    assert schema["name"] == "mock_tool"
    assert schema["description"] == "A mock tool for testing"
    assert schema["input_schema"]["type"] == "object"
    assert "input" in schema["input_schema"]["properties"]
    assert "optional" in schema["input_schema"]["properties"]
    assert "input" in schema["input_schema"]["required"]
    assert "optional" not in schema["input_schema"]["required"]


def test_tool_execute() -> None:
    tool = MockTool()
    result = tool.execute(input="test")

    assert result.success is True
    assert result.output == "Executed"
