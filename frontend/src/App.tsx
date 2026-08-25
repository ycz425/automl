import { RotateCcw } from "lucide-react";
import { ChatComposer } from "./components/ChatComposer";
import { ChatHeader } from "./components/ChatHeader";
import { ChatMessageList } from "./components/ChatMessageList";
import { PredictBar } from "./components/PredictBar";
import { useAutoMLChat } from "./hooks/useAutoMLChat";

export default function App() {
  const {
    messages,
    runState,
    selectedFile,
    errorMessage,
    isSubmitting,
    isPredicting,
    isComposerEnabled,
    canAttachFile,
    isPredictMode,
    isAwaitingClarification,
    showStartNewAnalysis,
    selectFile,
    sendMessage,
    runPrediction,
    resetSession,
    getDownloadUrl,
    setErrorMessage,
  } = useAutoMLChat();

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-neutral-950 text-neutral-100">
      <ChatHeader runState={runState} onClearSession={resetSession} />

      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
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
                className="flex items-center gap-1.5 text-xs font-medium text-neutral-500 transition-colors hover:text-neutral-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                Start new analysis
              </button>
            </div>
          )}
        </div>
      </main>

      {isPredictMode ? (
        <PredictBar
          isPredicting={isPredicting}
          onPredict={runPrediction}
          onInvalidFile={setErrorMessage}
        />
      ) : (
        runState !== "failed" && (
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
        )
      )}
    </div>
  );
}
