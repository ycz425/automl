import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  deleteArtifacts,
  deleteDataset,
  deleteStatus,
  getArtifactDownloadUrl,
  getArtifacts,
  getPredictionMetrics,
  predict,
  resumeAutoML,
  startAutoML,
  uploadDataset,
} from "../api/automlApi";
import type { Artifact, AutoMLNode, AutoMLResponse, PredictionLogEntry } from "../types/automl";
import type { ChatMessage, RunState } from "../types/chat";
import { toFriendlyMessage } from "../utils/errors";
import { generateId } from "../utils/files";
import { useAutoMLStream } from "./useAutoMLStream";

const STORAGE_KEY = "automl-agent-session-v1";

type PersistedSession = {
  threadId: string | null;
  datasetId: string | null;
  messages: ChatMessage[];
  runState: RunState;
};

const VALID_RUN_STATES: RunState[] = [
  "idle",
  "uploading",
  "starting",
  "running",
  "awaiting_clarification",
  "resuming",
  "completed",
  "failed",
];

function isPersistedSession(value: unknown): value is PersistedSession {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    (candidate.threadId === null || typeof candidate.threadId === "string") &&
    (candidate.datasetId === null || typeof candidate.datasetId === "string") &&
    Array.isArray(candidate.messages) &&
    typeof candidate.runState === "string" &&
    VALID_RUN_STATES.includes(candidate.runState as RunState)
  );
}

function loadPersistedSession(): PersistedSession | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!isPersistedSession(parsed)) return null;
    return parsed;
  } catch {
    return null;
  }
}

function clearPersistedSession() {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // localStorage may be unavailable (private browsing); ignore.
  }
}

function createWelcomeMessage(): ChatMessage {
  return {
    id: generateId(),
    role: "assistant",
    kind: "text",
    createdAt: new Date().toISOString(),
    content:
      "Welcome! I can help you build a machine learning pipeline from a dataset.\n\n" +
      "To get started:\n\n" +
      "1. Attach a CSV file using the paperclip button below.\n" +
      "2. Describe the target column and the ML task — binary classification, multiclass classification, or regression.\n" +
      "3. Say how you'd like the model evaluated (e.g. an 80/20 train/validation split, or 5-fold cross-validation) and which metric matters most (e.g. accuracy, F1, RMSE).\n" +
      "4. Optionally, mention any grouping that must stay together across splits (e.g. patient or subject ID), features to include or exclude, or other constraints.\n\n" +
      "Once you send your request, I'll analyze the data, design an approach, and run the experiments.",
  };
}

export function useAutoMLChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [runState, setRunState] = useState<RunState>("idle");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState<PredictionLogEntry[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [isLoadingArtifacts, setIsLoadingArtifacts] = useState(false);

  const { connect, close } = useAutoMLStream();
  const progressMessageIdRef = useRef<string | null>(null);
  // The node that most recently asked for clarification, so resumeRun can
  // seed the "Resuming..." bubble with it. Only meaningful right after a
  // need_clarification event, so it's only written there.
  const clarificationNodeRef = useRef<AutoMLNode | undefined>(undefined);
  // plan_agent and experiment_agent loop back and forth a variable number of
  // times. iterationRef counts how many planning rounds have started, so the
  // progress UI can show "round N" instead of the pipeline stepper appearing
  // to un-complete a stage each time it loops back to plan_agent.
  const iterationRef = useRef(0);
  const previousRunningNodeRef = useRef<AutoMLNode | undefined>(undefined);
  const abortControllerRef = useRef<AbortController | null>(null);
  const hasInitializedRef = useRef(false);

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const removeMessage = useCallback((id: string) => {
    setMessages((prev) => prev.filter((m) => m.id !== id));
  }, []);

  const updateMessage = useCallback(
    (id: string, patch: Partial<ChatMessage> | ((prev: ChatMessage) => Partial<ChatMessage>)) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m
        )
      );
    },
    []
  );

  const connectStream = useCallback(
    (activeThreadId: string) => {
      connect(activeThreadId, {
        onStatus: (data: AutoMLResponse) => handleStreamStatus(data),
        onConnectionError: (message: string) => {
          // Without this, whichever stage was active keeps spinning forever
          // even though the run just failed and an error bubble is about to
          // appear right below it.
          if (progressMessageIdRef.current) {
            updateMessage(progressMessageIdRef.current, {
              progressStatus: "failed",
              content: "This stage did not complete.",
            });
            progressMessageIdRef.current = null;
          }
          addMessage({
            id: generateId(),
            role: "assistant",
            kind: "error",
            createdAt: new Date().toISOString(),
            content: message,
          });
          setRunState("failed");
        },
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [connect, addMessage]
  );

  function handleStreamStatus(data: AutoMLResponse) {
    if (data.status === "running") {
      setRunState("running");

      // Entering plan_agent (whether for the first time or looping back
      // after experiment_agent) starts a new round.
      if (data.node === "plan_agent" && previousRunningNodeRef.current !== "plan_agent") {
        iterationRef.current += 1;
      }

      const isNewStage = data.node !== previousRunningNodeRef.current;
      previousRunningNodeRef.current = data.node;

      // "running" updates don't reliably carry a message — the node label
      // already conveys progress, so an empty detail line is fine
      // (ChatMessageBubble hides it when blank).
      const runningContent = data.message ?? "";

      if (isNewStage) {
        // A new stage starts its own message, and the previous stage's
        // message (if any) is marked done rather than reused or removed —
        // the chat transcript reads as a persistent record of every stage
        // the run went through, not a single bubble that mutates in place.
        // Reaching a genuinely new stage means the previous one passed, so
        // "done" (a checkmark) is accurate here specifically.
        if (progressMessageIdRef.current) {
          updateMessage(progressMessageIdRef.current, { progressStatus: "done" });
        }
        const id = generateId();
        progressMessageIdRef.current = id;
        addMessage({
          id,
          role: "assistant",
          kind: "progress",
          createdAt: new Date().toISOString(),
          content: runningContent,
          node: data.node,
          iteration: iterationRef.current,
          progressStatus: "active",
        });
      } else if (progressMessageIdRef.current) {
        updateMessage(progressMessageIdRef.current, { content: runningContent });
      }
      return;
    }

    if (data.status === "need_clarification") {
      clarificationNodeRef.current = data.node;
      if (progressMessageIdRef.current) {
        // This stage didn't pass — it paused pending input — so it gets a
        // distinct "clarifying" marker, not the same checkmark as a stage
        // the run moved past on its own. Its content also needs replacing:
        // otherwise it's left showing whatever transient status text
        // ("Parsing request...") was there right before the pause, which
        // reads oddly next to the "clarifying" icon.
        updateMessage(progressMessageIdRef.current, {
          progressStatus: "clarifying",
          content: "Waiting for your input to continue.",
        });
        progressMessageIdRef.current = null;
      }
      const clarificationContent =
        data.message ?? "The agent needs more information to continue.";
      // The stream stays open across a clarification round-trip, so a
      // reconnect (e.g. after a page reload) immediately replays the current
      // status. Skip re-adding the question if it's already the last message.
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.kind === "clarification" && last.content === clarificationContent) {
          return prev;
        }
        return [
          ...prev,
          {
            id: generateId(),
            role: "assistant",
            kind: "clarification",
            createdAt: new Date().toISOString(),
            content: clarificationContent,
            node: data.node,
          },
        ];
      });
      setRunState("awaiting_clarification");
      return;
    }

    if (data.status === "completed") {
      if (progressMessageIdRef.current) {
        updateMessage(progressMessageIdRef.current, { progressStatus: "done" });
        progressMessageIdRef.current = null;
      }
      addMessage({
        id: generateId(),
        role: "assistant",
        kind: "result",
        createdAt: new Date().toISOString(),
        // message is the summary of the experimentation process.
        content: data.message ?? "Your AutoML run has completed.",
        node: data.node,
      });
      setRunState("completed");
      return;
    }

    if (data.status === "failed") {
      if (progressMessageIdRef.current) {
        // The stage in progress when the run failed didn't pass either —
        // it gets the same non-checkmark treatment as a clarification pause,
        // and its stale running-status text is replaced for the same reason.
        // The actual failure reason goes in the separate error bubble below.
        updateMessage(progressMessageIdRef.current, {
          progressStatus: "failed",
          content: "This stage did not complete.",
        });
        progressMessageIdRef.current = null;
      }
      addMessage({
        id: generateId(),
        role: "assistant",
        kind: "error",
        createdAt: new Date().toISOString(),
        content: data.message ?? "The AutoML run failed. Please start a new analysis.",
        node: data.node,
      });
      setRunState("failed");
    }
  }

  const startRun = useCallback(
    async (message: string, file: File) => {
      if (isSubmitting) return;

      addMessage({
        id: generateId(),
        role: "user",
        kind: "csv",
        createdAt: new Date().toISOString(),
        content: "",
        attachment: { name: file.name, size: file.size, file, caption: "Uploaded dataset" },
      });
      addMessage({
        id: generateId(),
        role: "user",
        kind: "text",
        createdAt: new Date().toISOString(),
        content: message,
      });

      setSelectedFile(null);
      setErrorMessage(null);
      setIsSubmitting(true);
      setRunState("uploading");

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const uploadResult = await uploadDataset(file, controller.signal);
        setDatasetId(uploadResult.dataset_id);
        setRunState("starting");
        const startResult = await startAutoML(
          message,
          uploadResult.dataset_id,
          controller.signal
        );
        setThreadId(startResult.thread_id);
        setRunState("running");

        // prompt_agent always runs first, so seed its bubble immediately
        // instead of showing nothing until the first "running" event
        // arrives. previousRunningNodeRef must be set to match — otherwise
        // handleStreamStatus sees that first real "running" event as a node
        // change from its initial undefined value and creates a duplicate
        // bubble instead of updating this one in place.
        const progressId = generateId();
        progressMessageIdRef.current = progressId;
        previousRunningNodeRef.current = "prompt_agent";
        addMessage({
          id: progressId,
          role: "assistant",
          kind: "progress",
          createdAt: new Date().toISOString(),
          content: "Getting started...",
          node: "prompt_agent",
          progressStatus: "active",
        });

        connectStream(startResult.thread_id);
      } catch (error) {
        if (controller.signal.aborted) return;
        const friendly = toFriendlyMessage(
          error,
          "Something went wrong while starting the run."
        );
        addMessage({
          id: generateId(),
          role: "assistant",
          kind: "error",
          createdAt: new Date().toISOString(),
          content: friendly,
        });
        setRunState("failed");
      } finally {
        if (!controller.signal.aborted) setIsSubmitting(false);
      }
    },
    [isSubmitting, addMessage, connectStream]
  );

  const resumeRun = useCallback(
    async (message: string) => {
      if (isSubmitting || !threadId) return;

      addMessage({
        id: generateId(),
        role: "user",
        kind: "text",
        createdAt: new Date().toISOString(),
        content: message,
      });

      setErrorMessage(null);
      setIsSubmitting(true);
      setRunState("resuming");

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        await resumeAutoML(threadId, message, controller.signal);
        setRunState("running");

        // The stream connection opened for the original run stays alive
        // across clarification (only "completed"/"failed" close it), so
        // there's no need to reconnect here — just prime a progress bubble
        // for the next event that arrives on it. Seed it with the node that
        // asked for clarification so the pipeline indicator shows the right
        // stage instead of nothing highlighted.
        const progressId = generateId();
        progressMessageIdRef.current = progressId;
        addMessage({
          id: progressId,
          role: "assistant",
          kind: "progress",
          createdAt: new Date().toISOString(),
          content: "Resuming...",
          node: clarificationNodeRef.current,
          progressStatus: "active",
        });
      } catch (error) {
        if (controller.signal.aborted) return;
        const friendly = toFriendlyMessage(
          error,
          "Something went wrong while resuming the run."
        );
        addMessage({
          id: generateId(),
          role: "assistant",
          kind: "error",
          createdAt: new Date().toISOString(),
          content: friendly,
        });
        setRunState("failed");
      } finally {
        if (!controller.signal.aborted) setIsSubmitting(false);
      }
    },
    [isSubmitting, threadId, addMessage]
  );

  const refreshPredictionMetrics = useCallback(async (activeThreadId: string) => {
    try {
      const response = await getPredictionMetrics(activeThreadId);
      setPredictionHistory(response.history);
    } catch {
      // Monitoring is a bonus view on top of a successful prediction —
      // failing to refresh it shouldn't surface as a user-facing error.
    }
  }, []);

  // Fetched on demand (not cached from the run's "completed" event) since
  // the artifact set changes after retraining/promotion — the download menu
  // should always reflect what's on disk right now, not what existed when
  // the run first finished.
  const refreshArtifacts = useCallback(async () => {
    if (!threadId) return;
    setIsLoadingArtifacts(true);
    try {
      const response = await getArtifacts(threadId);
      setArtifacts(response.artifacts.map((filename) => ({ filename })));
    } catch {
      setArtifacts([]);
    } finally {
      setIsLoadingArtifacts(false);
    }
  }, [threadId]);

  const runPrediction = useCallback(
    async (file: File) => {
      if (isPredicting || !threadId) return;

      addMessage({
        id: generateId(),
        role: "user",
        kind: "csv",
        createdAt: new Date().toISOString(),
        content: "",
        attachment: { name: file.name, size: file.size, file, caption: "Prediction input" },
      });

      const progressId = generateId();
      addMessage({
        id: progressId,
        role: "assistant",
        kind: "progress",
        createdAt: new Date().toISOString(),
        content: "Generating predictions — this may take a moment.",
      });

      setSelectedFile(null);
      setIsPredicting(true);

      try {
        const csvText = await predict(threadId, file);
        const outputName = file.name.replace(/\.csv$/i, "") + "_predictions.csv";
        removeMessage(progressId);
        addMessage({
          id: generateId(),
          role: "assistant",
          kind: "csv",
          createdAt: new Date().toISOString(),
          content: "",
          attachment: { name: outputName, csvText, caption: "Predictions" },
        });
        await refreshPredictionMetrics(threadId);
      } catch (error) {
        removeMessage(progressId);
        const friendly = toFriendlyMessage(
          error,
          "Something went wrong while generating predictions."
        );
        addMessage({
          id: generateId(),
          role: "assistant",
          kind: "error",
          createdAt: new Date().toISOString(),
          content: friendly,
        });
      } finally {
        setIsPredicting(false);
      }
    },
    [isPredicting, threadId, addMessage, removeMessage, refreshPredictionMetrics]
  );

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      if (runState === "idle" && selectedFile) {
        startRun(trimmed, selectedFile);
      } else if (runState === "awaiting_clarification") {
        resumeRun(trimmed);
      }
    },
    [runState, selectedFile, startRun, resumeRun]
  );

  const resetSession = useCallback(() => {
    close();
    abortControllerRef.current?.abort();
    progressMessageIdRef.current = null;
    clarificationNodeRef.current = undefined;
    iterationRef.current = 0;
    previousRunningNodeRef.current = undefined;

    // Best-effort server-side cleanup — don't block the UI reset on it, and
    // don't surface failures since the session is being discarded regardless.
    if (threadId) {
      deleteArtifacts(threadId).catch(() => {});
      deleteStatus(threadId).catch(() => {});
    }
    if (datasetId) {
      deleteDataset(datasetId).catch(() => {});
    }

    clearPersistedSession();
    setMessages([createWelcomeMessage()]);
    setThreadId(null);
    setDatasetId(null);
    setSelectedFile(null);
    setErrorMessage(null);
    setIsSubmitting(false);
    setPredictionHistory([]);
    setArtifacts([]);
    setRunState("idle");
  }, [close, threadId, datasetId]);

  const selectFile = useCallback((file: File | null) => {
    setErrorMessage(null);
    setSelectedFile(file);
  }, []);

  const getDownloadUrl = useCallback(
    (filename: string) => {
      if (!threadId) return "";
      return getArtifactDownloadUrl(threadId, filename);
    },
    [threadId]
  );

  useLayoutEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    const persisted = loadPersistedSession();

    if (!persisted) {
      setMessages([createWelcomeMessage()]);
      return;
    }

    const restoredMessages = persisted.messages.length
      ? persisted.messages
      : [createWelcomeMessage()];
    setMessages(restoredMessages);
    setThreadId(persisted.threadId);
    setDatasetId(persisted.datasetId);

    if (
      persisted.runState === "running" ||
      persisted.runState === "resuming" ||
      persisted.runState === "awaiting_clarification"
    ) {
      if (persisted.threadId) {
        const lastProgress = [...restoredMessages]
          .reverse()
          .find((m) => m.kind === "progress");
        progressMessageIdRef.current = lastProgress ? lastProgress.id : null;
        iterationRef.current = lastProgress?.iteration ?? 0;
        previousRunningNodeRef.current = lastProgress?.node;
        // No need to seed clarificationNodeRef here: reconnecting replays the
        // current status, and if it's still need_clarification that handler
        // sets the ref itself.
        setRunState(persisted.runState);
        connectStream(persisted.threadId);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            role: "assistant",
            kind: "error",
            createdAt: new Date().toISOString(),
            content:
              "Your previous session could not be resumed. Please start a new analysis.",
          },
        ]);
        setRunState("failed");
      }
    } else if (
      persisted.runState === "uploading" ||
      persisted.runState === "starting"
    ) {
      setMessages((prev) => [
        ...prev,
        {
          id: generateId(),
          role: "assistant",
          kind: "error",
          createdAt: new Date().toISOString(),
          content:
            "Your previous request was interrupted. Please start a new analysis.",
        },
      ]);
      setRunState("failed");
    } else {
      setRunState(persisted.runState);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useLayoutEffect(() => {
    if (!hasInitializedRef.current) return;
    try {
      const payload: PersistedSession = { threadId, datasetId, messages, runState };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Storage may be full or unavailable; session persistence is best-effort.
    }
  }, [threadId, datasetId, messages, runState]);

  useLayoutEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  // Restores the monitoring dashboard after a page reload lands back in
  // predict mode — the history lives server-side in the run directory, so
  // there's nothing to persist client-side, just a re-fetch on mount.
  useEffect(() => {
    if (runState === "completed" && threadId) {
      refreshPredictionMetrics(threadId);
    }
  }, [runState, threadId, refreshPredictionMetrics]);

  const isComposerEnabled =
    (runState === "idle" || runState === "awaiting_clarification") &&
    !isSubmitting;
  const canAttachFile = runState === "idle";
  const isPredictMode = runState === "completed";
  const isAwaitingClarification = runState === "awaiting_clarification";
  const showStartNewAnalysis = runState === "completed" || runState === "failed";
  const isBusy =
    runState === "uploading" ||
    runState === "starting" ||
    runState === "running" ||
    runState === "resuming";

  return {
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
    isBusy,
    selectFile,
    sendMessage,
    runPrediction,
    resetSession,
    getDownloadUrl,
    refreshArtifacts,
    setErrorMessage,
    addMessage,
    updateMessage,
  };
}
