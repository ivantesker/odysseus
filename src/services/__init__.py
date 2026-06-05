"""Domain services — pure logic, no FastAPI.

A service holds the "what the app does" independent of "how it's exposed over
HTTP". Functions/classes here take plain arguments (ids, owners, dicts) and
return plain data; they never touch ``Request`` or ``request.state``. Route
handlers stay thin: read inputs (via dependencies like ``RequiredUser``), call a
service, return its result.

This is the seam that lets the same logic back multiple UIs and be unit-tested
without spinning up the web layer. New code should put logic here and keep the
router a few lines; existing god-routes can be migrated incrementally.
"""
