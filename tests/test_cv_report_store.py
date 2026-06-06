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


def test_list_reports_owner_scoped_and_sorted():
    a = rs.save_report("<html>a</html>", owner="lister", meta={"title": "A"})
    b = rs.save_report("<html>b</html>", owner="lister", meta={"title": "B"})
    rs.save_report("<html>x</html>", owner="someone_else", meta={"title": "X"})
    mine = rs.list_reports(owner="lister")
    ids = {r["id"] for r in mine}
    assert a in ids and b in ids
    assert all(r["title"] and r["url"].startswith("/api/cv/report/") for r in mine)
    # other owner's report not listed for this owner
    assert all("someone_else" not in r.get("id", "") for r in mine)


def test_list_reports_pagination():
    for i in range(5):
        rs.save_report(f"<html>{i}</html>", owner="pager", meta={"title": f"P{i}"})
    page1 = rs.list_reports(owner="pager", limit=2, offset=0)
    page2 = rs.list_reports(owner="pager", limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})
