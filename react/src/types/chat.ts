export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  timestamp?: Date;
}

export interface QueryRequest {
  query: string;
  top_k?: number;
  messages?: Message[];
}

export interface QueryResponse {
  answer: string;
  sources: string[];
}

