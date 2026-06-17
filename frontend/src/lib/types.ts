export interface User {
  id: string;
  email: string;
}

export interface AccountConfig {
  id?: string;
  user_id: string;
  llm_provider: string;
  api_key: string;
  decode_model: string;
  embedding_model: string;
  deep_enrichment?: boolean;
}

export interface FileItem {
  id: string;
  filename: string;
  s3_key: string;
  upload_date: string;
  type: "csv" | "bill";
}

export interface PendingTransaction {
  description: string;
  amount: number;
  trans_date: string;
  category: string;
  merchant_name: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  charts?: string[];
  images?: string[];
  toolTraces?: ToolEvent[];
  pendingTransactions?: PendingTransaction[];
}

export interface ToolEvent {
  type: "tool_start" | "tool_end";
  name: string;
  input?: string;
  snippet?: string;
}
