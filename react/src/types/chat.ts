export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  timestamp?: Date;
}

export interface QueryRequest {
  query: string;
  top_k?: number;
}

export interface QueryResponse {
  answer: string;
  sources: string[];
}

