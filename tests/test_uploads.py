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


# --- route ordering -----------------------------------------------------------
#
# GET /files/{file_id} is a catch-all. FastAPI matches in declaration order, so
# if it is registered above a literal path, that path stops existing: the
# request arrives at the by-id handler as a file named "ingestion-status" and
# 404s. Nothing else fails, the upload screen just polls forever.

def test_ingestion_status_is_not_shadowed_by_the_by_id_route(client):
    response = client.get("/api/v1/files/ingestion-status")
    assert response.status_code == 200
    # The status handler's shape, not the by-id handler's "File not found".
    assert "status" in response.json()


def test_the_by_id_route_still_resolves(client, monkeypatch):
    from backend.services import file_service

    async def _one(user, file_id):
        return {"id": file_id, "filename": "r.jpg", "s3_key": "k", "upload_date": "", "type": "bill"}

    monkeypatch.setattr(file_service, "get_file", _one)
    response = client.get("/api/v1/files/abc-123")
    assert response.status_code == 200
    assert response.json()["id"] == "abc-123"


def test_a_missing_file_is_a_404(client, monkeypatch):
    from backend.services import file_service

    async def _none(user, file_id):
        return None

    monkeypatch.setattr(file_service, "get_file", _none)
    assert client.get("/api/v1/files/nope").status_code == 404


# --- persisted photo orientation ----------------------------------------------
#
# Stored as an angle, not by rewriting the image: the photo is the evidence
# behind a financial record, re-encoding it loses quality on every turn, and an
# interrupted re-upload can leave a receipt half-replaced with no other copy.

import pytest


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
@pytest.mark.asyncio
async def test_quarter_turns_are_accepted(degrees, monkeypatch):
    from backend.services import file_service

    seen = {}

    def _fake(access_token, file_id, value):
        seen["value"] = value
        return value

    monkeypatch.setattr(file_service, "_set_file_rotation_sync", _fake)
    result = await file_service.set_file_rotation(
        {"access_token": "t"}, "file-1", degrees
    )
    assert result == degrees
    assert seen["value"] == degrees


@pytest.mark.parametrize("degrees", [45, -90, 360, 1, 91])
@pytest.mark.asyncio
async def test_anything_but_a_quarter_turn_is_rejected(degrees):
    """The column has the same CHECK, so this keeps a bad value from surfacing
    as a raw database error."""
    from backend.services import file_service

    with pytest.raises(ValueError, match="0, 90, 180 or 270"):
        await file_service.set_file_rotation({"access_token": "t"}, "file-1", degrees)


def test_rotation_defaults_to_upright_on_the_schema():
    """Rows written before migration 036 have no rotation; they are not sideways."""
    from backend.schemas.files import FileItem

    item = FileItem(
        id="f1", filename="r.jpg", s3_key="k", upload_date="2026-08-07", type="bill"
    )
    assert item.rotation == 0


# --- the same statement, uploaded twice ----------------------------------------
#
# Row-level dedup cannot catch this. Each row's content_hash is built from the
# CSV's own file id, which is new on every upload — deliberately, so two
# different exports covering one period stay separate. The cost is that
# re-uploading ONE identical file produces entirely new hashes, matches nothing,
# and writes every transaction again: doubled rows, a second pass of LLM
# enrichment, a second set of embeddings, and an agent that sums twice.

def test_identical_bytes_hash_identically(tmp_path):
    from backend.services.upload_utils import file_sha256

    first = tmp_path / "jan.csv"
    second = tmp_path / "jan-copy.csv"
    body = b"date,desc,amount\n2026-01-04,TESCO,12.30\n"
    first.write_bytes(body)
    second.write_bytes(body)
    # The NAME is not part of it — the same statement saved twice by a browser
    # arrives as "statement.csv" and "statement (1).csv".
    assert file_sha256(str(first)) == file_sha256(str(second))


def test_a_changed_byte_is_a_different_file(tmp_path):
    """A later export covering more days must still import."""
    from backend.services.upload_utils import file_sha256

    first = tmp_path / "jan.csv"
    second = tmp_path / "jan-feb.csv"
    first.write_bytes(b"date,desc,amount\n2026-01-04,TESCO,12.30\n")
    second.write_bytes(b"date,desc,amount\n2026-01-04,TESCO,12.30\n2026-02-01,BP,40.00\n")
    assert file_sha256(str(first)) != file_sha256(str(second))


def test_a_hash_never_seen_before_is_not_a_duplicate():
    """csv_file_by_content_hash returns None rather than raising, and an empty
    hash short-circuits — rows predating the column have a NULL one."""
    from backend.db_client import DatabaseClient

    assert DatabaseClient.csv_file_by_content_hash(object(), "user-1", "") is None


def test_the_upload_response_reports_skipped_files():
    """A silent skip is indistinguishable from a silent success."""
    from backend.schemas.files import UploadResponse

    response = UploadResponse(
        message="Nothing new to import — you already have these.",
        file_ids=[],
        already_imported=[
            {"filename": "statement (1).csv", "existing_filename": "statement.csv",
             "uploaded_at": "2026-08-05"}
        ],
    )
    assert response.already_imported[0].uploaded_at == "2026-08-05"


def test_an_ordinary_upload_reports_nothing_skipped():
    from backend.schemas.files import UploadResponse

    assert UploadResponse(message="ok", file_ids=["f1"]).already_imported == []
