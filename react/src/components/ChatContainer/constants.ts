import type { Message } from "@/types/chat";

export const INITIAL_MESSAGE: Message = {
    role: "assistant",
    content: "Hello! I'm here to help you with questions about Deutsche Telekom publications. What would you like to know?",
    timestamp: new Date(),
};
