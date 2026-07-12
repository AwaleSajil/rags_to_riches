-- ============================================================
-- Persistent chat sessions: Conversation + Message
-- ============================================================

CREATE TABLE IF NOT EXISTS public."Conversation" (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    title      text NOT NULL DEFAULT 'New chat',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversation_user ON public."Conversation"(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS public."Message" (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id      uuid NOT NULL REFERENCES public."Conversation"(id) ON DELETE CASCADE,
    user_id              uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    role                 text NOT NULL,          -- 'user' | 'assistant'
    content              text,
    charts               jsonb,
    images               jsonb,
    pending_transactions jsonb,
    created_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_message_conversation ON public."Message"(conversation_id, created_at);

-- RLS: a user only sees their own conversations/messages.
ALTER TABLE public."Conversation" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Message"      ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS conversation_own ON public."Conversation";
CREATE POLICY conversation_own ON public."Conversation"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS message_own ON public."Message";
CREATE POLICY message_own ON public."Message"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
