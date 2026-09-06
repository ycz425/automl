import { RotateCcw } from "lucide-react";
import { useLayoutEffect, useRef } from "react";
import { ChatComposer } from "./components/ChatComposer";
import { ChatHeader } from "./components/ChatHeader";
import { ChatMessageList } from "./components/ChatMessageList";
import { PredictBar } from "./components/PredictBar";
import { RetrainPanel } from "./components/RetrainPanel";
import { useAutoMLChat } from "./hooks/useAutoMLChat";
import { useRetrain } from "./hooks/useRetrain";

export default function App() {
  const {
    messages,
    runState,
    threadId,
    selectedFile,
    errorMessage,
    isSubmitting,
    isPredicting,
    predictionHistory,
    artifacts,
    isLoadingArtifacts,
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
    refreshArtifacts,
    setErrorMessage,
    addMessage,
    updateMessage,
  } = useAutoMLChat();

  const retrain = useRetrain(threadId, addMessage, updateMessage);
  const mainRef = useRef<HTMLElement | null>(null);

  // Owned here rather than inside ChatMessageList: errorMessage and the
  // "Start new analysis" button also render inside this same scroll
  // container, after the message list, so a sentinel scrolled into view
  // from within ChatMessageList alone can land above content that's still
  // below the fold. Scrolling the container to its actual max height
  // whenever anything in it changes guarantees it's always fully scrolled.
  // predictionHistory is included too: the drift monitor dashboard renders
  // in PredictBar, below (outside) this scroll container, so it doesn't
  // touch `messages` at all — but it still shrinks <main>'s available
  // height the first time it appears, which needs the same re-scroll.
  useLayoutEffect(() => {
    const el = mainRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, errorMessage, showStartNewAnalysis, predictionHistory]);

  function handleClearSession() {
    // Discards any unresolved retrain comparison (and the dataset created for
    // it) before resetting the main session — resetSession only knows about
    // the originally uploaded dataset, not one created mid-session for
    // retraining.
    retrain.discardAndClose(messages);
    resetSession();
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-neutral-950 text-neutral-100">
      <ChatHeader
        runState={runState}
        onClearSession={handleClearSession}
        isArtifactsAvailable={runState === "completed"}
        artifacts={artifacts}
        isLoadingArtifacts={isLoadingArtifacts}
        getDownloadUrl={getDownloadUrl}
        onOpenArtifacts={refreshArtifacts}
      />

      <main ref={mainRef} className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col">
          <ChatMessageList
            messages={messages}
            isRetrainResolving={retrain.isResolving}
            onPromoteRetrain={retrain.promote}
            onRejectRetrain={retrain.reject}
          />

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
                onClick={handleClearSession}
                className="flex items-center gap-1.5 text-xs font-medium text-neutral-500 transition-colors hover:text-neutral-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-400"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                Start new analysis
              </button>
            </div>
          )}
        </div>
      </main>

      <RetrainPanel
        stage={retrain.stage}
        dataRecord={retrain.dataRecord}
        manualLabels={retrain.manualLabels}
        allManualLabelsFilled={retrain.allManualLabelsFilled}
        isLoadingDataRecord={retrain.isLoadingDataRecord}
        isSubmittingLabels={retrain.isSubmittingLabels}
        error={retrain.error}
        onSetLabel={retrain.setLabel}
        onSubmitManualLabels={retrain.submitManualLabels}
        onSubmitLabelsFile={retrain.submitLabelsFile}
        onClose={retrain.close}
      />

      {isPredictMode ? (
        <PredictBar
          isPredicting={isPredicting}
          predictionHistory={predictionHistory}
          onPredict={runPrediction}
          onInvalidFile={setErrorMessage}
          onOpenRetrain={retrain.open}
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
