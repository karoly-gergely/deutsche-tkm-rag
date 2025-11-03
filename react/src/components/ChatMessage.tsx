import { Bot, User, FileText } from "lucide-react";
import type { Message } from "@/types/chat";
import { Badge } from "./ui/badge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./ui/accordion";

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === "user";
  const hasSources = message.sources && message.sources.length > 0;

  return (
    <div
      className={`flex gap-3 mb-4 ${isUser ? "flex-row-reverse" : "flex-row"}`}
      role="article"
      aria-label={`${message.role} message`}
    >
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? "bg-primary" : "bg-secondary"
        }`}
        aria-hidden="true"
      >
        {isUser ? (
          <User className="w-5 h-5 text-primary-foreground" aria-label="User" />
        ) : (
          <Bot className="w-5 h-5 text-foreground" aria-label="Assistant" />
        )}
      </div>

      <div
        className={`flex-1 max-w-[80%] sm:max-w-[75%] md:max-w-[70%] ${
          isUser ? "flex flex-col items-end" : "flex flex-col items-start"
        }`}
      >
        <div
          className={`p-4 rounded-2xl ${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-secondary text-foreground"
          }`}
        >
          <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
            {message.content}
          </p>
        </div>

        {!isUser && hasSources && (
          <div className="mt-2 w-full">
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="sources" className="border-none">
                <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:no-underline">
                  <div className="flex items-center gap-1">
                    <FileText className="w-3 h-3" />
                    <span>{message.sources.length} source(s)</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {message.sources.map((source, index) => (
                      <Badge
                        key={index}
                        variant="outline"
                        className="text-xs font-mono"
                      >
                        {source}
                      </Badge>
                    ))}
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        )}
      </div>
    </div>
  );
};
