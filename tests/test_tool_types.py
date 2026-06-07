"""A6: shared tool primitives + broken agent_tools↔tool_parsing cycle."""


def test_tool_types_is_single_source():
    from src import agent_tools, tool_parsing, tool_types
    # The same object is shared everywhere (not copies) so in-place updates
    # propagate.
    assert agent_tools.TOOL_TAGS is tool_types.TOOL_TAGS
    assert tool_parsing.TOOL_TAGS is tool_types.TOOL_TAGS
    assert agent_tools.ToolBlock is tool_types.ToolBlock


def test_register_extra_tool_tags_is_shared():
    from src import agent_tools, tool_parsing, tool_types
    name = "zz_a6_probe_tag"
    try:
        tool_parsing.register_extra_tool_tags([name])
        # mutation visible through every reference
        assert name in tool_types.TOOL_TAGS
        assert name in agent_tools.TOOL_TAGS
        # and the block regex was rebuilt to know the new tag
        blocks = tool_parsing.parse_tool_blocks(f"```{name}\n{{}}\n```")
        assert any(b.tool_type == name for b in blocks)
    finally:
        tool_types.TOOL_TAGS.discard(name)


def test_tool_parsing_imports_without_agent_tools_module_cycle():
    # tool_parsing must not import agent_tools at module load (the old cycle).
    import ast
    import pathlib
    src = pathlib.Path("src/tool_parsing.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.agent_tools":
            raise AssertionError("tool_parsing still imports src.agent_tools at module level")
