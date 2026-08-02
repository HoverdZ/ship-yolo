import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) {
    throw new Error(`Missing required argument ${name}`);
  }
  return process.argv[index + 1];
}

const inputPath = argument("--input");
const outputPath = argument("--output");
const previewPath = argument("--preview");

// The source CSV deliberately uses UTF-8 BOM for direct Excel compatibility.
// Strip that marker before importing so the first header is exactly "创新点".
const csvText = (await fs.readFile(inputPath, "utf8")).replace(/^\uFEFF/, "");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "材料索引" });
const sheet = workbook.worksheets.getItem("材料索引");
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);

const used = sheet.getUsedRange(true);
used.format = {
  font: { name: "Microsoft YaHei", size: 10, color: "#222222" },
  verticalAlignment: "center",
  wrapText: true,
};

const header = sheet.getRange("A1:L1");
header.format = {
  fill: "#1F4E79",
  font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#17365D" },
};
header.format.rowHeight = 34;

const widths = {
  A: 13,
  B: 26,
  C: 38,
  D: 38,
  E: 38,
  F: 35,
  G: 12,
  H: 12,
  I: 17,
  J: 17,
  K: 22,
  L: 48,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}:${column}`).format.columnWidth = width;
}

const rowCount = used.values.length;
if (rowCount > 1) {
  const body = sheet.getRange(`A2:L${rowCount}`);
  body.format.borders = {
    insideHorizontal: { style: "thin", color: "#E7E6E6" },
    bottom: { style: "thin", color: "#D9E2F3" },
  };
  body.format.rowHeight = 42;
  sheet.getRange(`G2:I${rowCount}`).format.horizontalAlignment = "center";
  sheet.tables.add(`A1:L${rowCount}`, true, "PaperMaterialIndex").style = "TableStyleMedium2";
}

const inspection = await workbook.inspect({
  kind: "table",
  range: `材料索引!A1:L${Math.min(rowCount, 25)}`,
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 12,
  maxChars: 5000,
});
console.log(inspection.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(path.dirname(previewPath), { recursive: true });
const preview = await workbook.render({
  sheetName: "材料索引",
  range: `A1:L${Math.min(rowCount, 30)}`,
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
// `inspect()` may emit a diagnostic sidecar next to the requested workbook.
// It is useful during construction but is not a paper artifact.
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
console.log(JSON.stringify({ outputPath, previewPath, rowCount }));
// artifact-tool may keep renderer worker handles alive on Windows. All files
// have been awaited and flushed at this point, so terminate deterministically.
process.exit(0);
