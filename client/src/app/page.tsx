"use client";

import { useState, useMemo } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, ToolUIPart } from "ai";
import { nanoid } from "nanoid";
import { Conversation, ConversationContent, ConversationScrollButton } from "@/components/ai-elements/conversation";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputAttachments,
  PromptInputAttachment,
  PromptInputProvider,
  PromptInputTools,
  PromptInputActionMenu,
  PromptInputActionMenuTrigger,
  PromptInputActionMenuContent,
  PromptInputActionAddAttachments,
  PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { Loader } from "@/components/ai-elements/loader";
import { Tool, ToolHeader, ToolContent, ToolInput, ToolOutput } from "@/components/ai-elements/tool";
import { Suggestions, Suggestion } from "@/components/ai-elements/suggestion";
import Greeting from "@/components/Greeting";
import Header from "@/components/Header";
import { useUser, useAuth } from "@clerk/nextjs";
import { BACKEND_URL } from "@/config/env";
import { toast } from "sonner";

// Type definitions for message parts
type TextPart = {
  type: "text";
  text: string;
};

type ToolPart = {
  type: `tool-${string}`;
  input?: ToolUIPart["input"];
  output?: ToolUIPart["output"];
  error?: string;
  state?: ToolUIPart["state"];
};

type DataPart = {
  type: `data-${string}`;
  data?: unknown;
};

type MessagePart = TextPart | ToolPart | DataPart | { type: string; [key: string]: unknown };

function ChatApp() {
  const [input, setInput] = useState("");
  const [threadId] = useState(() => nanoid());
  const { isSignedIn, isLoaded, getToken } = useAuth();
  const { user } = useUser();

  /* eslint-disable */
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: `${BACKEND_URL}/qa`,
        headers: async () => {
          const token = await getToken({ template: "ikms" });
          return {
            Authorization: `Bearer ${token}`,
          };
        },
        body: () => {
          return {
            thread_id: threadId
          };
        },
      }),
    [threadId]
  );

  const { messages, sendMessage, status } = useChat({
    transport,
    onError: (error) => {
      if (error instanceof Error) {
        const errorData = JSON.parse(error.message);
        toast.error(`${errorData.detail}`,{
          action: {
            label: "Dismiss",
            onClick: () => toast.dismiss(),
          },
        });
        return;
      }
    },
  });

  const handleSubmit = (message: PromptInputMessage) => {
    if (!message.text || !message.text.trim()) {
      return;
    }

    // Send the message - the transport will use the current isInterrupted value
    sendMessage({
      text: message.text.trim(),
      files: message.files,
    });

    setInput("");
  };

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage({ text: suggestion });
  };

  const renderMessagePart = (part: MessagePart, index: number, messageId: string) => {
    // Render text parts
    if (part.type === "text") {
      const textPart = part as TextPart;
      return (
        <Message key={`${messageId}-${index}`} from="assistant" className="my-4">
          <MessageContent>
            <MessageResponse>{textPart.text}</MessageResponse>
          </MessageContent>
        </Message>
      );
    }

    // Render tool calls
    if (part.type.startsWith("tool-")) {
      const toolPart = part as ToolPart;
      const toolName = toolPart.type.slice(5); // Remove 'tool-' prefix
      return (
        <Tool key={`${messageId}-${index}`}>
          <ToolHeader
            title={toolName.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
            type={toolPart.type as `tool-${string}`}
            state={(toolPart.state || "input-available") as ToolUIPart["state"]}
          />
          <ToolContent>
            {toolPart.input ? <ToolInput input={toolPart.input!} /> : null}
            {toolPart.output ? <ToolOutput output={toolPart.output!} errorText={undefined} /> : null}
            {toolPart.error ? <ToolOutput output={undefined} errorText={toolPart.error} /> : null}
          </ToolContent>
        </Tool>
      );
    }

    // Render custom data parts (Vercel protocol compliant)
    if (part.type.startsWith("data-")) {
      const dataPart = part as DataPart;
      const dataType = dataPart.type.slice(5); // Remove 'data-' prefix

      if (dataType === "final_user_requirements" && dataPart.data) {
        return (
          <div key={`${messageId}-${index}`} className="my-4">
            {/* <InterviewRequirements requirements={dataPart.data as unknown as ReqGathringModel} /> */}
          </div>
        );
      }

      if (dataType === "final_interview_evaluation" && dataPart.data) {
        return (
          <div key={`${messageId}-${index}`} className="my-4">
            {/* <InterviewEvaluation evaluation={dataPart.data as unknown as InterviewEvaluationType} /> */}
          </div>
        );
      }
    }

    return null;
  };

  return (
    <div className="mx-auto h-screen px-6 py-4 relative container overflow-hidden">
      <div className="h-full flex flex-col">
        {/* Header */}
        <Header />
        {/* chat body */}
        <div className="flex flex-col flex-1 min-h-0 w-full max-w-4xl mx-auto ">
          {/* Conversation Area */}
          <Conversation className="flex-1 text-lg flow min-h-0 scrollbar-thin">
            <ConversationContent>
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full gap-6">
                  <div className="text-center space-y-2">
                    {isSignedIn && isLoaded ? (
                      <Greeting name={user?.firstName || "User"} className="text-3xl" showIcon={false} />
                    ) : (
                      <p className="text-3xl font-semibold tracking-tight">Organize and manage your knowledge.</p>
                    )}
                    <p className="text-muted-foreground">Your intelligent knowledge management system with AI-powered organization and retrieval.</p>
                  </div>
                </div>
              )}

              {messages.map((message) => (
                <div key={message.id}>
                  {message.role === "user" ? (
                    <Message from="user">
                      <MessageContent>
                        <MessageResponse>{message.parts.find((p) => p.type === "text")?.text || ""}</MessageResponse>
                      </MessageContent>
                    </Message>
                  ) : (
                    message.parts.map((part, i) => renderMessagePart(part, i, message.id))
                  )}
                </div>
              ))}

              {/* Loading State */}
              {status === "submitted" && (
                <Message from="assistant">
                  <MessageContent>
                    <div className="flex items-center gap-2">
                      <Loader />
                      <span className="text-sm text-muted-foreground">Thinking...</span>
                    </div>
                  </MessageContent>
                </Message>
              )}
            </ConversationContent>

            {/* Scroll to Bottom Button */}
            <ConversationScrollButton />
          </Conversation>

          {/* Suggestions (only show when no messages) */}
          {messages.length === 0 && isSignedIn && (
            <Suggestions className="mb-4 shrink-0">
              <Suggestion onClick={() => handleSuggestionClick("How can I organize my project documentation?")} suggestion="Organize project documentation" />
              <Suggestion onClick={() => handleSuggestionClick("Help me categorize my research papers and notes")} suggestion="Categorize research materials" />
              <Suggestion onClick={() => handleSuggestionClick("Create a knowledge base for my team")} suggestion="Create team knowledge base" />
            </Suggestions>
          )}

          {/* Input Area */}
          <PromptInputProvider>
            <PromptInput onSubmit={handleSubmit} className="mt-4 shrink-0">
              <PromptInputAttachments>{(attachment) => <PromptInputAttachment data={attachment} />}</PromptInputAttachments>
              {/* Textarea */}
              <PromptInputBody>
                <PromptInputTextarea onChange={(e) => setInput(e.target.value)} value={input} placeholder="Ask me about organizing, retrieving, or managing your knowledge..." />
              </PromptInputBody>

              {/* Footer with Submit Button */}
              <PromptInputFooter>
                <PromptInputTools>
                  <PromptInputActionMenu>
                    <PromptInputActionMenuTrigger />
                    <PromptInputActionMenuContent>
                      <PromptInputActionAddAttachments />
                    </PromptInputActionMenuContent>
                  </PromptInputActionMenu>
                </PromptInputTools>
                <PromptInputSubmit disabled={!input || !isSignedIn} status={status} size={"sm"} />
              </PromptInputFooter>
            </PromptInput>
          </PromptInputProvider>
        </div>
      </div>
    </div>
  );
}

export default ChatApp;
