"""Drop-in plugins package.

Any module placed here is auto-imported at startup via
`src.plugin_registry.load_plugins()`. Use the registry decorators
(`register_tool`, `register_action`, `register_route`) to add a feature
without editing central wiring. See `plugins/example_ping.py`.
"""
