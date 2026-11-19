import {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";

const RED = "FF1737";
const INK = "15191E";
const MUTED = "64716E";
const BORDER = "DCE4E1";
const LIGHT = "F4F7F6";
const PALE_RED = "FFF0F2";
const FONT = "Aptos";

const borders = {
  top: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  left: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  right: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
  insideVertical: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
};

function clean(value, fallback = "—") {
  if (value == null || value === "") return fallback;
  return String(value).replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/g, "").replace(/\s+/g, " ").trim();
}

function paragraph(text, options = {}) {
  return new Paragraph({
    spacing: { after: 100, line: 280 },
    children: [new TextRun({ text: clean(text, ""), font: FONT, size: 21, color: INK })],
    ...options,
  });
}

function sectionTitle(text) {
  return new Paragraph({
    spacing: { before: 260, after: 110 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: RED, space: 5 } },
    children: [new TextRun({ text, font: FONT, size: 30, bold: true, color: INK })],
  });
}

function cell(text, width, { header = false, fill = null, bold = false, color = INK } = {}) {
  const content = Array.isArray(text) ? text : [text];
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 100, bottom: 100, left: 130, right: 130 },
    borders,
    ...(fill ? { shading: { type: ShadingType.CLEAR, color: "FFFFFF", fill } } : {}),
    children: content.map((item) => typeof item === "string"
      ? paragraph(item, { spacing: { after: 0, line: 260 }, children: [new TextRun({ text: clean(item), font: FONT, size: header ? 18 : 19, bold: header || bold, color: header ? "FFFFFF" : color })] })
      : item),
  });
}

function table(headers, rows, widths) {
  const makeRow = (values, header = false, rowIndex = 0) => new TableRow({
    tableHeader: header,
    cantSplit: true,
    children: values.map((value, index) => cell(value, widths[index], {
      header,
      fill: header ? INK : rowIndex % 2 === 1 ? LIGHT : "FFFFFF",
      bold: !header && index === 0,
      color: !header && index === 0 ? INK : MUTED,
    })),
  });
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: widths,
    borders,
    rows: [makeRow(headers, true), ...rows.map((row, index) => makeRow(row, false, index))],
  });
}

function moneylessDate(date) {
  if (!date) return "—";
  const parsed = new Date(`${date}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? date : new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "long", year: "numeric", timeZone: "UTC" }).format(parsed);
}

function addBullets(values) {
  return values.map((value) => new Paragraph({
    bullet: { indent: 360 },
    spacing: { after: 80, line: 280 },
    children: [new TextRun({ text: clean(value), font: FONT, size: 20, color: INK })],
  }));
}

export async function exportSummaryToWord(summary) {
  const report = summary.report || {};
  const categories = Array.isArray(report.categories) ? report.categories : [];
  const priorities = Array.isArray(report.priorities) ? report.priorities : [];
  const lowConfidenceThemes = Array.isArray(report.lowConfidenceThemes) ? report.lowConfidenceThemes : [];
  const children = [
    new Paragraph({ spacing: { before: 90, after: 60 }, children: [new TextRun({ text: "GREAT PLACE TO WORK  /  CHATBOT ANALYTICS", font: FONT, size: 17, bold: true, color: RED, characterSpacing: 12 })] }),
    new Paragraph({ spacing: { before: 260, after: 65 }, children: [new TextRun({ text: "Chat Summary", font: FONT, size: 46, bold: true, color: INK })] }),
    new Paragraph({ spacing: { after: 220 }, children: [new TextRun({ text: "Conversation themes, answer quality signals, and recommended actions", font: FONT, size: 23, color: MUTED })] }),
    table(["REPORT DETAILS", "SELECTION"], [
      ["Date range", `${moneylessDate(summary.startDate)} to ${moneylessDate(summary.endDate)}`],
      ["Source", "Chat Bubble / Iframe"],
      ["Report language", clean(summary.language)],
      ["Conversations analysed", Number(summary.conversationCount || 0).toLocaleString()],
      ["Low-confidence conversations", Number(summary.lowConfidenceCount || 0).toLocaleString()],
      ["Generated", summary.createdAt ? new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short" }).format(new Date(summary.createdAt)) : "—"],
    ], [2800, 6560]),
    sectionTitle("Executive summary"),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [9360],
      borders,
      rows: [new TableRow({ cantSplit: true, children: [new TableCell({
        width: { size: 9360, type: WidthType.DXA },
        margins: { top: 180, bottom: 180, left: 210, right: 210 },
        borders,
        shading: { type: ShadingType.CLEAR, color: "FFFFFF", fill: PALE_RED },
        children: [paragraph(report.executiveSummary || "No executive summary was returned.", { spacing: { after: 0, line: 320 }, children: [new TextRun({ text: clean(report.executiveSummary || "No executive summary was returned."), font: FONT, size: 23, color: INK })] })],
      })] })],
    }),
    sectionTitle("Conversation categories and topics"),
  ];

  if (categories.length) {
    children.push(table(["CATEGORY", "CHATS", "SHARE", "MOST COMMON TOPICS"], categories.map((category) => {
      const topics = Array.isArray(category.topics) && category.topics.length
        ? category.topics.map((topic) => `${clean(topic.name)} (${Number(topic.count || 0).toLocaleString()})`)
        : ["No topic breakdown available"];
      return [clean(category.name), Number(category.count || 0).toLocaleString(), `${Number(category.share || 0)}%`, topics];
    }), [2450, 1050, 1050, 4810]));
  } else children.push(paragraph("No category breakdown is available for this report."));

  children.push(sectionTitle("Recommended priorities"));
  if (priorities.length) {
    children.push(table(["PRIORITY", "FOCUS AREA", "EVIDENCE", "RECOMMENDED ACTION"], priorities.map((item) => [
      clean(item.priority, "Monitor").toUpperCase(), clean(item.title), clean(item.evidence), clean(item.recommendation),
    ]), [1150, 2100, 3050, 3060]));
  } else children.push(paragraph("No improvement priorities were returned."));

  children.push(sectionTitle("Low-confidence answer themes"));
  children.push(paragraph("A low-confidence flag is a review signal based on assistant response scores; it does not by itself mean an answer was incorrect.", { spacing: { after: 100 }, children: [new TextRun({ text: "A low-confidence flag is a review signal based on assistant response scores; it does not by itself mean an answer was incorrect.", font: FONT, size: 19, italics: true, color: MUTED })] }));
  if (lowConfidenceThemes.length) {
    children.push(table(["THEME", "CHATS", "LIKELY GAP", "SUGGESTED IMPROVEMENT"], lowConfidenceThemes.map((item) => [
      clean(item.theme), Number(item.count || 0).toLocaleString(), clean(item.likelyGap), clean(item.improvement),
    ]), [1900, 900, 3280, 3280]));
  } else children.push(paragraph("No distinct low-confidence themes were identified."));

  children.push(sectionTitle("Limitations and interpretation"));
  const limitations = Array.isArray(report.limitations) ? report.limitations.filter(Boolean) : [];
  if (limitations.length) children.push(...addBullets(limitations));
  else children.push(paragraph("Topics and recommendations are model-generated interpretations of the selected chats. Review the underlying conversations before taking action."));
  children.push(new Paragraph({ spacing: { before: 230, after: 0 }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: BORDER, space: 6 } }, children: [new TextRun({ text: "Prepared from the selected Chat Bubble / Iframe conversations. This report contains analysis only; source chat text is not included.", font: FONT, size: 16, color: MUTED })] }));

  const doc = new Document({
    creator: "Chatbot Analytics",
    title: `Chat Summary ${summary.startDate} to ${summary.endDate}`,
    subject: "Chatbot conversation analysis",
    description: "Conversation themes, answer quality signals, and recommended actions.",
    styles: {
      default: { document: { run: { font: FONT, size: 21, color: INK }, paragraph: { spacing: { after: 100, line: 280 } } } },
    },
    sections: [{
      properties: { page: { margin: { top: 1050, right: 1220, bottom: 1050, left: 1220 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "CHATBOT ANALYTICS  •  CONFIDENTIAL", font: FONT, size: 15, bold: true, color: MUTED })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Great Place to Work - Chat Summary", font: FONT, size: 15, color: MUTED })] })] }) },
      children,
    }],
  });

  const blob = await Packer.toBlob(doc);
  const filePart = [summary.startDate, summary.endDate, summary.language]
    .map((part) => clean(part, "report").replace(/[^\p{L}\p{N}-]+/gu, "-").replace(/^-|-$/g, ""))
    .join("_");
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `GPTW_Chat_Summary_${filePart}.docx`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
