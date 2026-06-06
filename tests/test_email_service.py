"""Email service pure helpers — parsing, formatting, sanitization (no IMAP)."""
from src.services import email_service as es


def test_folder_name_from_list_line():
    assert es.folder_name_from_list_line(b'(\\HasNoChildren) "/" "INBOX"') == "INBOX"
    assert es.folder_name_from_list_line('(\\All) "/" [Gmail]/All') == "[Gmail]/All"


def test_folder_role_from_name():
    assert es.folder_role_from_name("[Gmail]/Trash") == "trash"
    assert es.folder_role_from_name("Deleted Items") == "trash"
    assert es.folder_role_from_name("Spam") == "junk"
    assert es.folder_role_from_name("[Gmail]/All Mail") == "archive"
    assert es.folder_role_from_name("INBOX") == ""


def test_uid_helpers():
    assert es.uid_bytes("42") == b"42"
    assert es.uid_bytes(b"7") == b"7"
    assert es.uid_from_fetch_meta(b"* 1 FETCH (UID 123 FLAGS ())") == "123"
    assert es.uid_from_fetch_meta(b"no uid here") == ""


def test_smtp_ready():
    assert es.smtp_ready({"smtp_host": "h", "smtp_user": "u", "smtp_password": "p"}) is True
    assert es.smtp_ready({"smtp_host": "h", "smtp_user": "u"}) is False
    assert es.smtp_ready({}) is False


def test_envelope_recipients_handles_comma_in_display_name():
    # The canonical Outlook form: a comma inside the quoted display name must
    # not split into two bogus addresses.
    addrs = es.envelope_recipients('"Smith, John" <john@corp.com>, jane@x.com')
    assert addrs == ["john@corp.com", "jane@x.com"]


def test_envelope_recipients_empty():
    assert es.envelope_recipients("", None) == []


def test_md_to_email_html_escapes_then_formats():
    out = es.md_to_email_html("**bold** and <script>alert(1)</script>")
    assert "<strong>bold</strong>" in out
    # The script tag is escaped, never live.
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_md_to_email_html_lists_and_links():
    out = es.md_to_email_html("- one\n- two\n[site](https://example.com)")
    assert "<ul>" in out and "<li>one</li>" in out
    assert '<a href="https://example.com">site</a>' in out


def test_sanitize_email_html_drops_script_keeps_formatting():
    out = es.sanitize_email_html("<p>hi <b>there</b></p><script>steal()</script>")
    assert "<b>there</b>" in out
    assert "script" not in out.lower() or "steal" not in out

def test_sanitize_email_html_link_gets_safe_attrs_and_blocks_js():
    safe = es.sanitize_email_html('<a href="https://ok.com">x</a>')
    assert 'href="https://ok.com"' in safe and 'rel="noopener noreferrer"' in safe
    js = es.sanitize_email_html('<a href="javascript:alert(1)">x</a>')
    assert "javascript:" not in js


def test_sanitize_email_html_empty_returns_none():
    assert es.sanitize_email_html("") is None
    assert es.sanitize_email_html("   ") is None
