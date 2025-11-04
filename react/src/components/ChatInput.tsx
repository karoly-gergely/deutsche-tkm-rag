import { useState, useCallback } from "react";
import { Send, Loader2 } from "lucide-react";
import { Button } from "./core/button";
import { Textarea } from "./core/textarea";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput = ({ onSendMessage, disabled }: ChatInputProps) => {
  const [message, setMessage] = useState("");

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const trimmedMessage = message.trim();
      if (trimmedMessage && !disabled) {
        onSendMessage(trimmedMessage);
        setMessage("");
      }
    },
    [message, disabled, onSendMessage]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
      }
    },
    [handleSubmit]
  );

  const isEmpty = !message.trim();

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-border bg-card p-4"
      aria-label="Chat input"
    >
      <div className="flex gap-2 items-end max-w-4xl mx-auto">
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about Deutsche Telekom publications..."
          disabled={disabled}
          className="min-h-[60px] max-h-[200px] resize-none"
          aria-label="Message input"
          aria-describedby="input-help"
        />
        <Button
          type="submit"
          disabled={isEmpty || disabled}
          size="icon"
          className="h-[60px] w-[60px] flex-shrink-0"
          aria-label="Send message"
        >
          {disabled ? (
            <Loader2 className="w-5 h-5 animate-spin" aria-hidden="true" />
          ) : (
            <Send className="w-5 h-5" aria-hidden="true" />
          )}
        </Button>
      </div>
      <p id="input-help" className="sr-only">
        Press Enter to send, Shift+Enter for new line
      </p>
    </form>
  );
};
