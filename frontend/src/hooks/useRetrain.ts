import Papa from "papaparse";
import { useCallback, useState } from "react";
import {
  clearRetrain,
  deleteDataset,
  evaluateRetrain,
  getRunFileText,
  labelRetrainData,
  promoteChallenger,
} from "../api/automlApi";
import type { ChatMessage } from "../types/chat";
import { errorMessage, progressMessage, retrainResultMessage } from "../utils/chatMessages";
import { parseCsv, type ParsedCsv } from "../utils/csv";
import { toFriendlyMessage } from "../utils/errors";

export type RetrainStage = "closed" | "labeling";

type AddMessage = (message: ChatMessage) => void;
type UpdateMessage = (
  id: string,
  patch: Partial<ChatMessage> | ((prev: ChatMessage) => Partial<ChatMessage>)
) => void;

// Manages the "retrain from drift" side-flow. The labeling step (fetch the
// unlabeled rolling-window rows, let the user supply targets) is a modal,
// but training/evaluating and promoting are long-running (k-fold retraining
// can take minutes), so once labels are submitted the modal closes and the
// rest plays out as regular chat messages — a progress bubble for the run,
// then a result message with the comparison and Promote/Discard baked in.
export function useRetrain(threadId: string | null, addMessage: AddMessage, updateMessage: UpdateMessage) {
  const [stage, setStage] = useState<RetrainStage>("closed");
  const [dataRecord, setDataRecord] = useState<ParsedCsv | null>(null);
  const [manualLabels, setManualLabels] = useState<Record<number, string>>({});
  const [isLoadingDataRecord, setIsLoadingDataRecord] = useState(false);
  const [isSubmittingLabels, setIsSubmittingLabels] = useState(false);
  const [isResolving, setIsResolving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const close = useCallback(() => {
    setStage("closed");
    setDataRecord(null);
    setManualLabels({});
    setError(null);
  }, []);

  const open = useCallback(async () => {
    if (!threadId) return;
    setStage("labeling");
    setError(null);
    setManualLabels({});
    setIsLoadingDataRecord(true);

    try {
      const text = await getRunFileText(threadId, "data_record.csv");
      const parsed = await parseCsv(text);
      setDataRecord(parsed);
    } catch (err) {
      setError(toFriendlyMessage(err, "Failed to load the data waiting to be labeled."));
    } finally {
      setIsLoadingDataRecord(false);
    }
  }, [threadId]);

  const setLabel = useCallback((rowIndex: number, value: string) => {
    setManualLabels((prev) => ({ ...prev, [rowIndex]: value }));
  }, []);

  const submitLabelsFile = useCallback(
    async (file: File) => {
      if (!threadId) return;
      setIsSubmittingLabels(true);
      close();

      const progressMsg = progressMessage({
        content: "Training and evaluating a challenger model on the labeled data — this can take a few minutes.",
        label: "Retraining",
      });
      const progressId = progressMsg.id;
      addMessage(progressMsg);

      try {
        const labelResponse = await labelRetrainData(threadId, file);
        const result = await evaluateRetrain(threadId, labelResponse.dataset_id);
        updateMessage(progressId, { progressStatus: "done", content: "" });
        addMessage(
          retrainResultMessage({
            datasetId: labelResponse.dataset_id,
            champion: result.champion,
            challenger: result.challenger,
            challengerThreshold: result.challenger_threshold,
          })
        );
      } catch (err) {
        updateMessage(progressId, { progressStatus: "failed", content: "Retraining failed." });
        addMessage(errorMessage(toFriendlyMessage(err, "Failed to train and evaluate the new model.")));
      } finally {
        setIsSubmittingLabels(false);
      }
    },
    [threadId, addMessage, updateMessage, close]
  );

  // Manual entry only ever fills in a single target column — the header
  // text is irrelevant since the backend renames it to the real target
  // column name, so any placeholder works.
  const submitManualLabels = useCallback(() => {
    if (!dataRecord) return;
    const csvText = Papa.unparse({
      fields: ["label"],
      data: dataRecord.rows.map((_, rowIndex) => [manualLabels[rowIndex] ?? ""]),
    });
    const file = new File([csvText], "labels.csv", { type: "text/csv" });
    return submitLabelsFile(file);
  }, [dataRecord, manualLabels, submitLabelsFile]);

  const promote = useCallback(
    async (messageId: string, datasetId: string) => {
      if (!threadId) return;
      setIsResolving(true);

      const progressMsg = progressMessage({
        content:
          "Retraining the final model on the full labeled dataset and deploying it — this can take a few minutes.",
        label: "Promoting new model",
      });
      const progressId = progressMsg.id;
      addMessage(progressMsg);

      try {
        await promoteChallenger(threadId, datasetId);
        updateMessage(progressId, { progressStatus: "done", content: "" });
        updateMessage(messageId, (prev) =>
          prev.retrainResult
            ? { retrainResult: { ...prev.retrainResult, resolution: "promoted" } }
            : {}
        );
      } catch (err) {
        updateMessage(progressId, { progressStatus: "failed", content: "Promotion failed." });
        addMessage(errorMessage(toFriendlyMessage(err, "Failed to promote the new model.")));
      } finally {
        setIsResolving(false);
      }
    },
    [threadId, addMessage, updateMessage]
  );

  const reject = useCallback(
    async (messageId: string) => {
      if (!threadId) return;
      setIsResolving(true);
      try {
        await clearRetrain(threadId);
        updateMessage(messageId, (prev) =>
          prev.retrainResult
            ? { retrainResult: { ...prev.retrainResult, resolution: "rejected" } }
            : {}
        );
      } catch (err) {
        addMessage(errorMessage(toFriendlyMessage(err, "Failed to discard the comparison.")));
      } finally {
        setIsResolving(false);
      }
    },
    [threadId, addMessage, updateMessage]
  );

  // Used when the whole session is being cleared. Unlike close() (which only
  // resets the labeling modal), this also cleans up datasets created for any
  // retrain comparison that's still awaiting a promote/reject decision —
  // resolved ones already had their fate handled, but an abandoned one would
  // otherwise leak in storage.
  const discardAndClose = useCallback(
    (messages: ChatMessage[]) => {
      for (const message of messages) {
        if (message.kind === "retrain_result" && message.retrainResult && !message.retrainResult.resolution) {
          deleteDataset(message.retrainResult.datasetId).catch(() => {});
        }
      }
      close();
    },
    [close]
  );

  const allManualLabelsFilled =
    dataRecord !== null &&
    dataRecord.rows.length > 0 &&
    dataRecord.rows.every((_, rowIndex) => (manualLabels[rowIndex] ?? "").trim() !== "");

  return {
    stage,
    dataRecord,
    manualLabels,
    allManualLabelsFilled,
    isLoadingDataRecord,
    isSubmittingLabels,
    isResolving,
    error,
    open,
    close,
    discardAndClose,
    setLabel,
    submitManualLabels,
    submitLabelsFile,
    promote,
    reject,
  };
}
