import { Loader2 } from "lucide-react";

export const LoadingIndicator = () => (
    <div className="flex gap-3 mb-4" role="status" aria-label="Loading">
    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
      <Loader2
        className="w-5 h-5 animate-spin text-foreground"
        aria-hidden="true"
      />
    </div>
    <div className="flex-1 max-w-[80%] sm:max-w-[75%] md:max-w-[70%] p-4 rounded-2xl bg-secondary">
      <p className="text-sm text-muted-foreground">Thinking...</p>
    </div>
  </div>
);
