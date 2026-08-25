const PREVIEW_ROW_LIMIT = 100;

type CsvPreviewTableProps = {
  columns: string[];
  rows: Record<string, string>[];
  totalRowCount: number;
};

export function CsvPreviewTable({ columns, rows, totalRowCount }: CsvPreviewTableProps) {
  const visibleRows = rows.slice(0, PREVIEW_ROW_LIMIT);
  const isTruncated = totalRowCount > PREVIEW_ROW_LIMIT;

  if (columns.length === 0) {
    return <p className="text-sm text-neutral-500">This file has no readable rows.</p>;
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="max-h-80 overflow-auto rounded-lg border border-neutral-700">
        <table className="w-full min-w-max border-collapse text-left text-xs">
          <thead className="sticky top-0 bg-neutral-800 text-neutral-300">
            <tr>
              {columns.map((column) => (
                <th key={column} className="whitespace-nowrap px-3 py-2 font-medium">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {visibleRows.map((row, rowIndex) => (
              <tr key={rowIndex} className="odd:bg-neutral-900/40 even:bg-transparent">
                {columns.map((column) => (
                  <td key={column} className="whitespace-nowrap px-3 py-1.5 text-neutral-300">
                    {row[column]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-neutral-500">
        {isTruncated
          ? `Showing first ${PREVIEW_ROW_LIMIT} of ${totalRowCount} rows.`
          : `${totalRowCount} row${totalRowCount === 1 ? "" : "s"}.`}
      </p>
    </div>
  );
}
