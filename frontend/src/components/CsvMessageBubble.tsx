import { ChevronDown, ChevronRight, Download, Table2 } from "lucide-react";
import { useState } from "react";
import { formatFileSize } from "../utils/files";
import { parseCsv, type ParsedCsv } from "../utils/csv";
import { CsvPreviewTable } from "./CsvPreviewTable";

type CsvMessageBubbleProps = {
  name: string;
  size?: number;
  file?: File;
  csvText?: string;
  caption?: string;
  isUser: boolean;
};

export function CsvMessageBubble({
  name,
  size,
  file,
  csvText,
  caption,
  isUser,
}: CsvMessageBubbleProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [parsed, setParsed] = useState<ParsedCsv | null>(null);
  const [isParsing, setIsParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);

  // `File` objects don't survive a JSON round-trip (localStorage persistence
  // serializes messages between reloads) — a deserialized `file` field comes
  // back as a plain, unusable `{}` rather than `undefined`. Only trust it
  // when it's still a real File.
  const liveFile = file instanceof File ? file : undefined;
  const isUnavailable = !liveFile && !csvText;

  const displaySize = size ?? (csvText ? new Blob([csvText]).size : undefined);

  async function handleToggle() {
    const next = !isExpanded;
    setIsExpanded(next);
    if (next && !parsed && !isParsing && !isUnavailable) {
      setIsParsing(true);
      setParseError(null);
      try {
        const source = liveFile ?? csvText ?? "";
        const result = await parseCsv(source);
        setParsed(result);
      } catch {
        setParseError("Could not read this file as CSV.");
      } finally {
        setIsParsing(false);
      }
    }
  }

  function handleDownload(event: React.MouseEvent) {
    event.stopPropagation();
    if (isUnavailable) return;
    const blob = liveFile ?? new Blob([csvText ?? ""], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div
      className={[
        "w-full overflow-hidden rounded-lg",
        isUser ? "bg-indigo-600" : "border border-neutral-800 bg-neutral-900",
      ].join(" ")}
    >
      <div
        className={[
          "flex w-full items-center gap-3 px-4 py-3 transition-colors",
          isUser ? "hover:bg-indigo-500" : "hover:bg-neutral-800",
        ].join(" ")}
      >
        <button
          type="button"
          onClick={handleToggle}
          aria-expanded={isExpanded}
          className="flex min-w-0 flex-1 items-center gap-3 text-left"
        >
          <div
            className={[
              "flex h-9 w-9 shrink-0 items-center justify-center rounded-md",
              isUser ? "bg-white/15 text-white" : "bg-indigo-500/10 text-indigo-400",
            ].join(" ")}
          >
            <Table2 className="h-4.5 w-4.5" aria-hidden="true" />
          </div>
          <div className="min-w-0 flex-1">
            {caption && (
              <p
                className={[
                  "truncate text-[11px] font-medium uppercase tracking-wide",
                  isUser ? "text-indigo-100/80" : "text-neutral-500",
                ].join(" ")}
              >
                {caption}
              </p>
            )}
            <p
              className={[
                "truncate text-sm font-medium",
                isUser ? "text-white" : "text-neutral-100",
              ].join(" ")}
            >
              {name}
            </p>
            {typeof displaySize === "number" && (
              <p className={["text-xs", isUser ? "text-indigo-100/70" : "text-neutral-500"].join(" ")}>
                {formatFileSize(displaySize)}
              </p>
            )}
          </div>
        </button>
        {!isUnavailable && (
          <button
            type="button"
            onClick={handleDownload}
            aria-label={`Download ${name}`}
            title="Download"
            className={[
              "flex h-8 w-8 shrink-0 items-center justify-center rounded-md transition-colors",
              isUser
                ? "text-white/80 hover:bg-white/15 hover:text-white"
                : "text-neutral-400 hover:bg-neutral-700 hover:text-neutral-100",
            ].join(" ")}
          >
            <Download className="h-4 w-4" aria-hidden="true" />
          </button>
        )}
        {isExpanded ? (
          <ChevronDown
            className={["h-4 w-4 shrink-0", isUser ? "text-white/70" : "text-neutral-500"].join(" ")}
            aria-hidden="true"
          />
        ) : (
          <ChevronRight
            className={["h-4 w-4 shrink-0", isUser ? "text-white/70" : "text-neutral-500"].join(" ")}
            aria-hidden="true"
          />
        )}
      </div>

      {isExpanded && (
        <div className="border-t border-neutral-800 bg-neutral-900 px-4 py-3">
          {isUnavailable && (
            <p className="text-sm text-neutral-500">
              This file's contents aren't available anymore — it was from an earlier session.
            </p>
          )}
          {isParsing && <p className="text-sm text-neutral-500">Reading file…</p>}
          {parseError && <p className="text-sm text-red-400">{parseError}</p>}
          {parsed && (
            <CsvPreviewTable
              columns={parsed.columns}
              rows={parsed.rows}
              totalRowCount={parsed.totalRowCount}
            />
          )}
        </div>
      )}
    </div>
  );
}
