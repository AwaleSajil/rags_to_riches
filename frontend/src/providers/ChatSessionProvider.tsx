import React, { createContext, useContext, useState } from "react";

/**
 * Lets the tab header (rendered in the layout) trigger actions that live in the
 * Chat screen — opening the sessions drawer and starting a new chat.
 * The Chat screen registers handlers; the header buttons call them.
 */
interface ChatSessionHandlers {
  onMenu?: () => void;
  onNewChat?: () => void;
}

interface ChatSessionContextValue {
  handlers: ChatSessionHandlers;
  setHandlers: (h: ChatSessionHandlers) => void;
}

const ChatSessionContext = createContext<ChatSessionContextValue>({
  handlers: {},
  setHandlers: () => {},
});

export function ChatSessionProvider({ children }: { children: React.ReactNode }) {
  const [handlers, setHandlers] = useState<ChatSessionHandlers>({});
  return (
    <ChatSessionContext.Provider value={{ handlers, setHandlers }}>
      {children}
    </ChatSessionContext.Provider>
  );
}

export const useChatSession = () => useContext(ChatSessionContext);
