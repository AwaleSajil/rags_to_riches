"""Upload filename and size handling.

The client-supplied filename reaches both `os.path.join(temp_dir, ...)` and the
object-storage key, and anything not recognised as an image is handed to the
CSV parser — so both the path and the extension need checking.
"""

import io

import pytest

from backend.routers.files import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, _safe_filename


@pytest.mark.parametrize("raw,expected", [
    ("statement.csv", "statement.csv"),
    ("receipt_1.JPG", "receipt_1.JPG"),          # extension check is case-insensitive
    ("a/b/c/x.png", "x.png"),
    ("../../../etc/passwd.csv", "passwd.csv"),
    (r"..\..\windows\system32\cfg.csv", "cfg.csv"),  # Windows-style separators
    ("....//x.csv", "x.csv"),
    ("  spaced.csv  ", "spaced.csv"),
])
def test_filename_reduced_to_safe_basename(raw, expected):
    assert _safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, ".", "..", "...."])
def test_unusable_filenames_rejected(raw):
    with pytest.raises(ValueError):
        _safe_filename(raw)


@pytest.mark.parametrize("raw", [
    "notes.pdf",
    "photo.heic",          # iPhone default — would otherwise hit the CSV parser
    "sheet.xlsx",
    "script.sh",
    "noextension",
    "evil.csv\x00.exe",    # null-byte truncation trick
])
def test_unsupported_extensions_rejected(raw):
    with pytest.raises(ValueError, match="unsupported type"):
        _safe_filename(raw)


def test_traversal_can_never_escape_the_temp_dir():
    """Whatever comes in, the result must be a bare name with no path parts."""
    for attempt in ["../x.csv", "../../x.csv", "/abs/path/x.csv", r"..\..\x.csv"]:
        result = _safe_filename(attempt)
        assert "/" not in result and "\\" not in result
        assert not result.startswith(".")


def test_allowlist_matches_what_the_pipeline_understands():
    # is_image in file_service keys off exactly these image types; everything
    # else in the allowlist must be something the CSV path can read.
    assert ALLOWED_EXTENSIONS == {".csv", ".png", ".jpg", ".jpeg"}


# --- end-to-end through the route -------------------------------------------

def _post(client, name: str, data: bytes):
    return client.post(
        "/api/v1/files/upload",
        files={"files": (name, io.BytesIO(data), "text/csv")},
    )


def test_oversized_upload_rejected(client):
    response = _post(client, "huge.csv", b"x" * (MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 400
    assert "larger than" in response.json()["detail"]


def test_empty_upload_rejected(client):
    response = _post(client, "empty.csv", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_bad_extension_rejected_at_the_route(client):
    response = _post(client, "notes.pdf", b"%PDF-1.4")
    assert response.status_code == 400
    assert "unsupported type" in response.json()["detail"]


def test_valid_file_clears_validation(client):
    """A good CSV must get past validation into the service layer, where the
    fake token then fails — a non-validation error proves it was accepted."""
    response = _post(client, "statement.csv", b"date,amount\n2026-01-01,5\n")
    detail = str(response.json().get("detail", ""))
    assert "unsupported type" not in detail
    assert "larger than" not in detail
    assert "is empty" not in detail


# --- an unexamined photo must not claim to be a receipt ----------------------


def test_bill_uploads_start_unexamined():
    """BillFile.kind defaults to 'receipt' in the schema, but only so rows that
    predate the column backfill correctly. A new upload must say 'unknown'
    until the vision pass has actually looked at it: if ingestion crashes in
    between, an inherited 'receipt' leaves a photo nobody examined asserting it
    is one — and confirming that invents spending that never happened."""
    captured = {}

    class FakeTable:
        def insert(self, record):
            captured.update(record)
            return self

        def execute(self):
            return type("R", (), {"data": [{"id": "file-1"}]})()

    class FakeSupabase:
        def table(self, name):
            return FakeTable()

    from backend.db_client import DatabaseClient

    client = DatabaseClient.__new__(DatabaseClient)
    client.supabase = FakeSupabase()
    client.insert_file_record("BillFile", "user-1", "shelf.jpg", "k")

    assert captured["kind"] == "unknown"


def test_csv_uploads_carry_no_kind():
    """kind is a property of a photo. A CSV has none, and sending the column
    would be rejected by the table."""
    captured = {}

    class FakeTable:
        def insert(self, record):
            captured.update(record)
            return self

        def execute(self):
            return type("R", (), {"data": [{"id": "file-2"}]})()

    class FakeSupabase:
        def table(self, name):
            return FakeTable()

    from backend.db_client import DatabaseClient

    client = DatabaseClient.__new__(DatabaseClient)
    client.supabase = FakeSupabase()
    client.insert_file_record("CSVFile", "user-1", "statement.csv", "k")

    assert "kind" not in captured
