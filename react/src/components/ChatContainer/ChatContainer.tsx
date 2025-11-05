import { useEffect, useRef, useState, useCallback } from "react";
import { sendQuery } from "@/lib/api";
import { Message } from "@/types/chat";
import { ChatMessage } from "../ChatMessage";
import { ChatInput } from "../ChatInput";
import { LoadingIndicator } from "./components/LoadingIndicator";
import { INITIAL_MESSAGE } from "./constants";

export const ChatContainer = () => {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSendMessage = useCallback(
    async (content: string) => {
      const userMessage: Message = {
        role: "user",
        content,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setError(null);

      try {
        const conversationHistory = messages
          .filter((msg) => msg.role === "user" || msg.role === "assistant")
          .map((msg) => ({
            role: msg.role,
            content: msg.content,
          }));
        
        const response = await sendQuery({ 
          query: content,
          messages: conversationHistory.length > 0 ? conversationHistory : undefined,
        });

        const assistantMessage: Message = {
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        console.error("Error sending message:", err);
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);

        const assistantMessage: Message = {
          role: "assistant",
          content:
            "I'm sorry, I'm having trouble connecting to the server. Please try again later.",
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, assistantMessage]);
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return (
    <div className="flex flex-col h-screen">
      <header className="border-b border-border bg-card px-4 sm:px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
            <span className="text-primary-foreground font-bold text-xl">T</span>
          </div>
          <div className="min-w-0">
            <h1 className="text-lg sm:text-xl font-semibold text-foreground truncate">
              Deutsche Telekom
            </h1>
            <p className="text-xs sm:text-sm text-muted-foreground truncate">
              Publications Assistant
            </p>
          </div>
        </div>
      </header>

      {error && (
        <div
          className="bg-destructive/10 border-b border-destructive/20 px-4 sm:px-6 py-2"
          role="alert"
          aria-live="assertive"
        >
          <p className="text-sm text-destructive max-w-4xl mx-auto">{error}</p>
        </div>
      )}

      <main className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
        <div className="max-w-4xl mx-auto">
          {messages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))}
          {isLoading && <LoadingIndicator />}
          <div ref={messagesEndRef} aria-hidden="true" />
        </div>
      </main>

      <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
};
