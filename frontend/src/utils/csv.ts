import Papa from "papaparse";

export type ParsedCsv = {
  columns: string[];
  rows: Record<string, string>[];
  totalRowCount: number;
};

/**
 * Parses a CSV file or raw CSV text into columns + rows, entirely
 * client-side. Used for the chat's "clickable file preview" bubbles —
 * nothing here is sent to the backend.
 */
export function parseCsv(input: File | string): Promise<ParsedCsv> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(input, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        resolve({
          columns: results.meta.fields ?? [],
          rows: results.data,
          totalRowCount: results.data.length,
        });
      },
      error: (error: Error) => reject(error),
    });
  });
}
