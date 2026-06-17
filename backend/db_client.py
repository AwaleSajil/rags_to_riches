import logging
from typing import List, Dict, Any, Optional

from backend.config import get_settings
from backend.dependencies import get_supabase

logger = logging.getLogger("moneyrag.db_client")


class DatabaseClient:
    def __init__(self, access_token: str):
        self.settings = get_settings()
        self.access_token = access_token
        
        logger.debug("Initializing Supabase client (token=%s...)", access_token[:20] if access_token else "Service")
        self.supabase = get_supabase(access_token)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --- AccountConfig ---

    def get_account_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        logger.debug("DatabaseClient.get_account_config for user_id=%s", user_id)
        res = self.supabase.table("AccountConfig").select("*").eq("user_id", user_id).execute()
        return res.data[0] if res.data else None

    def upsert_account_config(self, user_id: str, data: dict) -> Dict[str, Any]:
        logger.debug("DatabaseClient.upsert_account_config for user_id=%s", user_id)
        record = {
            "user_id": user_id,
            "llm_provider": data["llm_provider"],
            "api_key": data["api_key"],
            "decode_model": data["decode_model"],
            "embedding_model": data["embedding_model"],
            "deep_enrichment": data.get("deep_enrichment", False),
        }
        
        existing = self.supabase.table("AccountConfig").select("id").eq("user_id", user_id).execute()
        if existing.data:
            logger.debug("Updating existing AccountConfig id=%s", existing.data[0]["id"])
            self.supabase.table("AccountConfig").update(record).eq("id", existing.data[0]["id"]).execute()
        else:
            logger.debug("Inserting new AccountConfig")
            self.supabase.table("AccountConfig").insert(record).execute()
        return record

    # --- Files ---

    def list_files(self, user_id: str) -> tuple[List[Dict], List[Dict]]:
        """Returns tuple of (csv_files, bill_files)"""
        logger.debug("DatabaseClient.list_files for user_id=%s", user_id)
        res_csv = self.supabase.table("CSVFile").select("*").eq("user_id", user_id).execute()
        res_bill = self.supabase.table("BillFile").select("*").eq("user_id", user_id).execute()
        return res_csv.data or [], res_bill.data or []

    def insert_file_record(self, table: str, user_id: str, filename: str, s3_key: str) -> str:
        """Inserts a file record and returns its ID."""
        logger.debug("DatabaseClient.insert_file_record in %s for '%s'", table, filename)
        file_record = self.supabase.table(table).insert({
            "user_id": user_id,
            "filename": filename,
            "s3_key": s3_key,
        }).execute()
        return str(file_record.data[0]["id"])

    def get_file_record(self, table: str, file_id: str) -> Optional[Dict[str, Any]]:
        logger.debug("DatabaseClient.get_file_record from %s id=%s", table, file_id)
        record = self.supabase.table(table).select("*").eq("id", file_id).execute()
        return record.data[0] if record.data else None

    def delete_file_record(self, table: str, file_id: str):
        logger.debug("DatabaseClient.delete_file_record from %s id=%s", table, file_id)
        if table == "CSVFile":
            self.supabase.table("Transaction").delete().eq("source_csv_id", file_id).execute()
        self.supabase.table(table).delete().eq("id", file_id).execute()

def get_db_client(access_token: str) -> DatabaseClient:
    return DatabaseClient(access_token)
