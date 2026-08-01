import { RotateCcw } from "lucide-react";
import { ChatComposer } from "./components/ChatComposer";
import { ChatHeader } from "./components/ChatHeader";
import { ChatMessageList } from "./components/ChatMessageList";
import { useAutoMLChat } from "./hooks/useAutoMLChat";

export default function App() {
  const {
    messages,
    runState,
    selectedFile,
    errorMessage,
    isSubmitting,
    isComposerEnabled,
    canAttachFile,
    isAwaitingClarification,
    showStartNewAnalysis,
    selectFile,
    sendMessage,
    resetSession,
    getDownloadUrl,
    setErrorMessage,
  } = useAutoMLChat();

  return (
    <div className="flex h-dvh flex-col bg-neutral-950 text-neutral-100">
      <ChatHeader runState={runState} onClearSession={resetSession} />

      <main className="flex flex-1 flex-col overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col">
          <ChatMessageList messages={messages} getDownloadUrl={getDownloadUrl} />

          {errorMessage && (
            <div
              role="alert"
              className="mx-4 mb-4 rounded-lg border border-red-900/50 bg-red-950/30 px-4 py-2.5 text-sm text-red-300 sm:mx-0"
            >
              {errorMessage}
            </div>
          )}

          {showStartNewAnalysis && (
            <div className="flex justify-center px-4 pb-6 sm:px-0">
              <button
                type="button"
                onClick={resetSession}
                className="flex items-center gap-2 rounded-full bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
              >
                <RotateCcw className="h-4 w-4" aria-hidden="true" />
                Start new analysis
              </button>
            </div>
          )}
        </div>
      </main>

      {!showStartNewAnalysis && (
        <ChatComposer
          isEnabled={isComposerEnabled}
          isSubmitting={isSubmitting}
          canAttachFile={canAttachFile}
          isAwaitingClarification={isAwaitingClarification}
          selectedFile={selectedFile}
          onSelectFile={selectFile}
          onSend={sendMessage}
          onInvalidFile={setErrorMessage}
        />
      )}
    </div>
  );
}
