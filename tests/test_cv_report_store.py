"""CV report store — save/load HTML reports, owner scoping, id safety."""
from src.services.cv import report_store as rs


def test_save_and_load_unowned():
    rid = rs.save_report("<html>hi</html>", owner="", meta={"title": "t"})
    assert rid.startswith("cv-")
    # unowned report readable by anyone
    assert rs.load_report(rid, owner=None) == "<html>hi</html>"
    assert rs.load_report(rid, owner="alice") == "<html>hi</html>"


def test_owner_scoping():
    rid = rs.save_report("<html>secret</html>", owner="alice")
    assert rs.load_report(rid, owner="alice") == "<html>secret</html>"
    assert rs.load_report(rid, owner="bob") is None      # wrong owner
    assert rs.load_report(rid, owner=None) is None        # anonymous can't read an owned one


def test_load_unknown_or_bad_id():
    assert rs.load_report("cv-deadbeef0000", owner=None) is None
    assert rs.load_report("../etc/passwd", owner=None) is None      # traversal-safe
    assert rs.load_report("not-an-id", owner=None) is None
