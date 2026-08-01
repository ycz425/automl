import { useCallback, useEffect, useRef } from "react";
import { getStreamUrl } from "../api/automlApi";
import type { AutoMLNode, AutoMLResponse, AutoMLStatus } from "../types/automl";
import { PIPELINE_NODES } from "../types/automl";

const STATUSES: AutoMLStatus[] = [
  "running",
  "need_clarification",
  "completed",
  "failed",
];

const MAX_RETRIES = 3;
const BASE_RETRY_DELAY_MS = 1000;
const MAX_RETRY_DELAY_MS = 8000;

function isAutoMLResponse(value: unknown): value is AutoMLResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.status === "string" &&
    STATUSES.includes(candidate.status as AutoMLStatus) &&
    typeof candidate.node === "string" &&
    PIPELINE_NODES.includes(candidate.node as AutoMLNode) &&
    (candidate.message === null || typeof candidate.message === "string")
  );
}

export type AutoMLStreamCallbacks = {
  onStatus: (data: AutoMLResponse) => void;
  onConnectionError: (message: string) => void;
};

// The backend keeps the SSE connection open across a clarification request —
// only "completed" and "failed" end the run and close the stream. Closing on
// "need_clarification" would force a reconnect that can race the backend's
// (async, 202-accepted) resume handler and read back stale status.
const TERMINAL_STATUSES: AutoMLStatus[] = ["completed", "failed"];

/**
 * Owns the lifecycle of a single EventSource connection to the AutoML
 * status stream. Guards against StrictMode double-invocation and stale
 * closures by keeping all mutable connection state in refs.
 */
export function useAutoMLStream() {
  const eventSourceRef = useRef<EventSource | null>(null);
  const activeThreadIdRef = useRef<string | null>(null);
  const callbacksRef = useRef<AutoMLStreamCallbacks | null>(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<number | null>(null);
  const manuallyClosedRef = useRef(true);

  const clearRetryTimer = useCallback(() => {
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  const close = useCallback(() => {
    manuallyClosedRef.current = true;
    clearRetryTimer();
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    activeThreadIdRef.current = null;
  }, [clearRetryTimer]);

  const openConnection = useCallback((threadId: string) => {
    const eventSource = new EventSource(getStreamUrl(threadId));
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        callbacksRef.current?.onConnectionError(
          "Received an unreadable status update from the server."
        );
        return;
      }

      if (!isAutoMLResponse(parsed)) {
        callbacksRef.current?.onConnectionError(
          "Received an incomplete status update from the server."
        );
        return;
      }

      retryCountRef.current = 0;
      callbacksRef.current?.onStatus(parsed);

      if (TERMINAL_STATUSES.includes(parsed.status)) {
        close();
      }
    };

    eventSource.onerror = () => {
      if (manuallyClosedRef.current) return;

      eventSource.close();

      if (retryCountRef.current >= MAX_RETRIES) {
        manuallyClosedRef.current = true;
        callbacksRef.current?.onConnectionError(
          "Lost connection to the status stream. Please start a new analysis."
        );
        return;
      }

      const delay = Math.min(
        BASE_RETRY_DELAY_MS * 2 ** retryCountRef.current,
        MAX_RETRY_DELAY_MS
      );
      retryCountRef.current += 1;
      retryTimerRef.current = window.setTimeout(() => {
        if (!manuallyClosedRef.current && activeThreadIdRef.current === threadId) {
          openConnection(threadId);
        }
      }, delay);
    };
  }, [close]);

  const connect = useCallback(
    (threadId: string, callbacks: AutoMLStreamCallbacks) => {
      close();
      manuallyClosedRef.current = false;
      callbacksRef.current = callbacks;
      activeThreadIdRef.current = threadId;
      retryCountRef.current = 0;
      openConnection(threadId);
    },
    [close, openConnection]
  );

  useEffect(() => {
    return () => {
      close();
    };
  }, [close]);

  return { connect, close };
}
