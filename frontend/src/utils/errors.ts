// Central place for turning failures (HTTP, network, parsing) into
// short, user-facing strings. Never surface raw stack traces here.

export class AppError extends Error {
  constructor(message: string, public cause?: unknown) {
    super(message);
    this.name = "AppError";
  }
}

type FastAPIErrorBody = {
  detail?: string | { msg?: string }[] | string[];
};

function extractDetail(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const detail = (body as FastAPIErrorBody).detail;
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const first = detail[0];
    if (typeof first === "string") return first;
    if (first && typeof first === "object" && "msg" in first) {
      return String((first as { msg?: string }).msg ?? "");
    }
  }
  return null;
}

/** Parses a non-ok fetch Response into a friendly error message. */
export async function parseResponseError(
  response: Response,
  fallback: string
): Promise<string> {
  try {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const body = await response.json();
      const detail = extractDetail(body);
      if (detail) return detail;
    } else {
      const text = await response.text();
      if (text) return text.slice(0, 300);
    }
  } catch {
    // fall through to generic status-based message
  }

  if (response.status === 404) return "The requested resource was not found.";
  if (response.status >= 500) return "The server is currently unavailable. Please try again.";
  return fallback;
}

/** Normalizes any thrown value (fetch abort, network error, etc.) into a message. */
export function toFriendlyMessage(error: unknown, fallback: string): string {
  if (error instanceof AppError) return error.message;
  if (error instanceof DOMException && error.name === "AbortError") {
    return "The request was cancelled.";
  }
  if (error instanceof TypeError) {
    return "Could not reach the server. Check that the backend is running and reachable.";
  }
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
