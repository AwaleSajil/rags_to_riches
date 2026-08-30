"""Deleting an account has to actually delete the account.

Both stores require this and both mean the same thing: Apple's guideline
5.1.1(v) and Play's deletion policy each say an app that lets you create an
account must let you delete it from inside the app, and that clearing your data
or emailing support does not count.

The thing that makes it a real deletion is removing the `auth.users` row —
"User".id references it ON DELETE CASCADE and every user-scoped table cascades
from "User", so one delete takes all of it. These tests pin the parts that do
NOT cascade and would therefore rot silently: the storage objects, the
in-memory state, and the ordering that decides what a partial failure leaves
behind.
"""

import pytest

from backend.services import account_service
from tests.conftest import OTHER_USER_ID, TEST_USER_ID


class _Bucket:
    def __init__(self, listing=None, fail_remove=False):
        self._listing = listing or {}
        self.removed = None
        self.fail_remove = fail_remove

    def list(self, prefix):
        return self._listing.get(prefix, [])

    def remove(self, keys):
        if self.fail_remove:
            raise RuntimeError("bucket unavailable")
        self.removed = keys


class _Table:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a):
        return self

    def eq(self, *_a):
        return self

    def execute(self):
        return type("R", (), {"data": self._rows})()


class _UserClient:
    """A Supabase client scoped to the caller, as client_for returns."""

    def __init__(self, rows_by_table, bucket):
        self._rows = rows_by_table
        self._bucket = bucket

    def table(self, name):
        return _Table(self._rows.get(name, []))

    @property
    def storage(self):
        return self

    def from_(self, _bucket):
        return self._bucket


class _Admin:
    def __init__(self):
        self.deleted = []
        self.auth = self

    @property
    def admin(self):
        return self

    def delete_user(self, user_id):
        self.deleted.append(user_id)


@pytest.fixture
def user():
    return {"id": TEST_USER_ID, "email": "test@example.com", "access_token": "t"}


@pytest.fixture
def wired(monkeypatch):
    """Patch the two clients and hand back the doubles."""
    def _wire(rows=None, listing=None, fail_remove=False):
        bucket = _Bucket(listing, fail_remove)
        client = _UserClient(rows or {}, bucket)
        admin = _Admin()
        monkeypatch.setattr(account_service, "client_for", lambda _u: client)
        monkeypatch.setattr(account_service, "admin_client", lambda: admin)
        return bucket, admin
    return _wire


@pytest.mark.asyncio
async def test_the_auth_record_is_what_gets_deleted(user, wired):
    """Not the rows — the account. Everything else cascades from this."""
    _, admin = wired()
    await account_service.delete_account(user)
    assert admin.deleted == [TEST_USER_ID]


@pytest.mark.asyncio
async def test_stored_files_go_too(user, wired):
    """Receipt photos and bank exports have no foreign key to anything, so a
    cascade never touches them. They are also the most sensitive thing here."""
    bucket, _ = wired(
        rows={
            "CSVFile": [{"s3_key": f"{TEST_USER_ID}/csvs/abc_statement.csv"}],
            "BillFile": [{"s3_key": f"{TEST_USER_ID}/bills/def_receipt.jpg"}],
        }
    )
    await account_service.delete_account(user)
    assert set(bucket.removed) == {
        f"{TEST_USER_ID}/csvs/abc_statement.csv",
        f"{TEST_USER_ID}/bills/def_receipt.jpg",
    }


@pytest.mark.asyncio
async def test_orphaned_objects_are_found_by_listing_the_bucket(user, wired):
    """An upload that succeeded just before its INSERT did not leaves a file
    no row knows about. A deletion is the wrong moment to trust bookkeeping."""
    bucket, _ = wired(
        rows={"BillFile": []},
        listing={
            f"{TEST_USER_ID}/bills": [{"name": "orphan.jpg", "id": "obj-1"}],
            f"{TEST_USER_ID}/csvs": [],
        },
    )
    await account_service.delete_account(user)
    assert bucket.removed == [f"{TEST_USER_ID}/bills/orphan.jpg"]


@pytest.mark.asyncio
async def test_folder_placeholders_are_not_mistaken_for_files(user, wired):
    """Supabase lists a folder as a nameless-id entry; removing it is a no-op
    that would still count towards 'objects removed' and mislead the log."""
    bucket, _ = wired(
        listing={f"{TEST_USER_ID}/bills": [{"name": ".emptyFolderPlaceholder", "id": None}]}
    )
    result = await account_service.delete_account(user)
    assert bucket.removed is None
    assert result["objects_removed"] == 0


@pytest.mark.asyncio
async def test_a_failed_storage_wipe_leaves_the_account_intact(user, wired):
    """The recoverable order. Deleting the account first and then failing to
    clear the bucket would leave someone's receipts with nothing pointing at
    them and no way for them to try again."""
    _, admin = wired(
        rows={"BillFile": [{"s3_key": f"{TEST_USER_ID}/bills/x.jpg"}]},
        fail_remove=True,
    )
    with pytest.raises(RuntimeError):
        await account_service.delete_account(user)
    assert admin.deleted == []


@pytest.mark.asyncio
async def test_in_flight_state_for_that_user_is_dropped(user, wired, tmp_path):
    """A cached RAG instance owns a temp directory and a pending capture holds
    photo bytes on local disk. Neither is in the database, so neither cascades."""
    from backend.services import capture_service, file_service

    photo = tmp_path / "capture" / "shelf.jpg"
    photo.parent.mkdir()
    photo.write_bytes(b"jpeg")
    capture_service._pending["cap-1"] = {
        "user_id": TEST_USER_ID, "local_path": str(photo), "file_id": None,
        "created_at": 0, "kind": "processing", "draft": {},
    }
    file_service.ingestion_status[TEST_USER_ID] = {"status": "processing"}
    wired()
    try:
        await account_service.delete_account(user)
        assert "cap-1" not in capture_service._pending
        assert TEST_USER_ID not in file_service.ingestion_status
        assert not photo.parent.exists()
    finally:
        capture_service._pending.pop("cap-1", None)
        file_service.ingestion_status.pop(TEST_USER_ID, None)


@pytest.mark.asyncio
async def test_another_users_capture_is_left_alone(user, wired):
    """The sweep is keyed on the account being deleted, not on the whole map."""
    from backend.services import capture_service

    capture_service._pending["theirs"] = {
        "user_id": OTHER_USER_ID, "local_path": "/tmp/nope/x.jpg", "file_id": None,
        "created_at": 0, "kind": "processing", "draft": {},
    }
    wired()
    try:
        await account_service.delete_account(user)
        assert "theirs" in capture_service._pending
    finally:
        capture_service._pending.pop("theirs", None)


def test_the_route_needs_no_body_and_names_no_user(client, monkeypatch):
    """The id comes from the validated token, so there is no parameter here
    that could ever name somebody else's account."""
    seen = {}

    async def _fake(user):
        seen["user_id"] = user["id"]
        return {"deleted": True, "objects_removed": 0}

    monkeypatch.setattr(account_service, "delete_account", _fake)
    response = client.delete("/api/v1/auth/account")
    assert response.status_code == 200
    assert seen["user_id"] == TEST_USER_ID


def test_deleting_an_account_requires_authentication(anon_client):
    assert anon_client.delete("/api/v1/auth/account").status_code == 401


def test_a_deployment_without_the_service_key_says_so(client, monkeypatch):
    """503 with a human answer, not a generic 500. Someone who has decided to
    leave should be told what to do next."""
    async def _unavailable(_user):
        raise RuntimeError("SUPABASE_SERVICE_KEY is not configured")

    monkeypatch.setattr(account_service, "delete_account", _unavailable)
    response = client.delete("/api/v1/auth/account")
    assert response.status_code == 503
    assert "contact support" in response.json()["detail"].lower()


# --- the page Play requires to exist without the app -------------------------

def test_the_deletion_page_is_public(anon_client):
    """Requiring a login to read the deletion instructions would defeat the
    point: the policy exists for people who have already uninstalled."""
    response = anon_client.get("/account-deletion")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_the_deletion_page_says_what_goes_and_how(anon_client):
    body = anon_client.get("/account-deletion").text.lower()
    # The in-app route, which is what both stores actually check for.
    assert "settings" in body and "delete my account" in body
    # And the categories a reviewer looks for.
    for expected in ("receipt", "transaction", "api key"):
        assert expected in body


def test_an_unconfigured_support_address_is_admitted_not_faked(anon_client):
    """A placeholder that looks like an address sends deletion requests into a
    void, which is worse than a page saying it is unconfigured."""
    assert "SUPPORT_EMAIL_PLACEHOLDER" not in anon_client.get("/account-deletion").text
