-- The photos a chat turn was about.
--
-- A capture shows in the conversation as a local file URI, which does not
-- survive a reload, and a signed URL would have expired long before anyone
-- scrolled back. Storing the BillFile ids instead lets the server resolve fresh
-- signed URLs when the conversation is reloaded, so the picture the question
-- was about comes back with it.
--
-- jsonb rather than uuid[]: the column is written and read as a JSON list by
-- conversation_service, alongside the other jsonb payloads on this table
-- (charts, images, pending_transactions).
--
-- RECONSTRUCTED. This migration was applied to the database before its file was
-- committed, so this describes the column as it exists rather than the original
-- statement. Written to match the live schema: jsonb, nullable, no default.
ALTER TABLE "Message"
  ADD COLUMN IF NOT EXISTS bill_file_ids jsonb;
