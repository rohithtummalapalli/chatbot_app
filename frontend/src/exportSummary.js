import { AlignmentType, Document, Footer, Header, Packer, Paragraph, TextRun } from "docx";

const RED = "FF1737";
const INK = "15191E";
const MUTED = "64716E";
const FONT = "Aptos";

function clean(value, fallback = "—") {
  if (value == null || value === "") return fallback;
  return String(value).replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/g, "").replace(/\s+/g, " ").trim() || fallback;
}

function text(value, options = {}) {
  return new TextRun({ text: clean(value, ""), font: FONT, size: 21, color: INK, ...options });
}

function paragraph(value, options = {}) {
  return new Paragraph({
    spacing: { after: 100, line: 280 },
    children: [text(value)],
    ...options,
  });
}

function sectionHeading(value) {
  return new Paragraph({
    spacing: { before: 300, after: 130 },
    keepNext: true,
    children: [text(value.toUpperCase(), { size: 25, bold: true, color: RED, characterSpacing: 8 })],
  });
}

function subheading(value) {
  return new Paragraph({
    spacing: { before: 150, after: 50 },
    keepNext: true,
    children: [text(value, { size: 22, bold: true, color: INK })],
  });
}

function detail(label, value) {
  return new Paragraph({
    spacing: { after: 55, line: 260 },
    children: [text(`${label}: `, { bold: true }), text(value, { color: MUTED })],
  });
}

function bullet(value) {
  return new Paragraph({
    indent: { left: 300 },
    spacing: { after: 65, line: 270 },
    children: [text(`•  ${value}`)],
  });
}

function formattedDate(value) {
  if (!value) return "—";
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.getTime()) ? clean(value) : new Intl.DateTimeFormat("en-GB", {
    day: "2-digit", month: "long", year: "numeric", timeZone: "UTC",
  }).format(date);
}

export async function exportSummaryToWord(summary) {
  const report = summary.report || {};
  const categories = Array.isArray(report.categories) ? report.categories : [];
  const priorities = Array.isArray(report.priorities) ? report.priorities : [];
  const lowConfidenceThemes = Array.isArray(report.lowConfidenceThemes) ? report.lowConfidenceThemes : [];
  const children = [
    new Paragraph({
      spacing: { after: 90 },
      children: [text("GREAT PLACE TO WORK  /  CHATBOT ANALYTICS", { size: 16, bold: true, color: RED, characterSpacing: 10 })],
    }),
    new Paragraph({
      spacing: { before: 90, after: 60 },
      children: [text("Chat Summary", { size: 42, bold: true, color: INK })],
    }),
    new Paragraph({
      spacing: { after: 220 },
      children: [text("Conversation themes, answer quality signals, and recommended actions", { size: 22, color: MUTED })],
    }),
    sectionHeading("Report details"),
    detail("Date range", `${formattedDate(summary.startDate)} to ${formattedDate(summary.endDate)}`),
    detail("Source", "Chat Bubble / Iframe"),
    detail("Report language", summary.language),
    detail("Conversations analysed", Number(summary.conversationCount || 0).toLocaleString()),
    detail("Low-confidence conversations", Number(summary.lowConfidenceCount || 0).toLocaleString()),
    detail("Generated", summary.createdAt ? new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short" }).format(new Date(summary.createdAt)) : "—"),
    sectionHeading("Executive summary"),
    paragraph(report.executiveSummary || "No executive summary was returned.", { spacing: { after: 130, line: 320 } }),
    sectionHeading("Conversation categories and topics"),
  ];

  if (categories.length) {
    for (const category of categories) {
      children.push(subheading(`${clean(category.name)}  ·  ${Number(category.count || 0).toLocaleString()} conversations (${Number(category.share || 0)}%)`));
      const topics = Array.isArray(category.topics) ? category.topics : [];
      if (topics.length) {
        for (const topic of topics) children.push(bullet(`${clean(topic.name)}  (${Number(topic.count || 0).toLocaleString()})`));
      } else children.push(paragraph("No topic breakdown available.", { indent: { left: 300 }, spacing: { after: 65 } }));
    }
  } else children.push(paragraph("No category breakdown is available for this report."));

  children.push(sectionHeading("Recommended priorities"));
  if (priorities.length) {
    for (const item of priorities) {
      children.push(subheading(`${clean(item.priority, "Monitor").toUpperCase()}  ·  ${clean(item.title)}`));
      children.push(detail("Evidence", item.evidence));
      children.push(detail("Recommended action", item.recommendation));
    }
  } else children.push(paragraph("No improvement priorities were returned."));

  children.push(sectionHeading("Low-confidence answer themes"));
  children.push(paragraph("A low-confidence flag is a review signal based on assistant response scores; it does not by itself mean an answer was incorrect.", {
    spacing: { after: 110 },
    children: [text("A low-confidence flag is a review signal based on assistant response scores; it does not by itself mean an answer was incorrect.", { size: 19, italics: true, color: MUTED })],
  }));
  if (lowConfidenceThemes.length) {
    for (const item of lowConfidenceThemes) {
      children.push(subheading(`${clean(item.theme)}  ·  ${Number(item.count || 0).toLocaleString()} conversations`));
      children.push(detail("Likely gap", item.likelyGap));
      children.push(detail("Suggested improvement", item.improvement));
    }
  } else children.push(paragraph("No distinct low-confidence themes were identified."));

  children.push(sectionHeading("Limitations and interpretation"));
  const limitations = Array.isArray(report.limitations) ? report.limitations.filter(Boolean) : [];
  if (limitations.length) children.push(...limitations.map(bullet));
  else children.push(paragraph("Topics and recommendations are model-generated interpretations of the selected chats. Review the underlying conversations before taking action."));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 260, after: 0 },
    children: [text("Prepared from the selected Chat Bubble / Iframe conversations. Source chat text is not included.", { size: 16, color: MUTED })],
  }));

  const doc = new Document({
    creator: "Chatbot Analytics",
    title: `Chat Summary ${summary.startDate} to ${summary.endDate}`,
    subject: "Chatbot conversation analysis",
    description: "Conversation themes, answer quality signals, and recommended actions.",
    sections: [{
      properties: { page: { margin: { top: 1050, right: 1220, bottom: 1050, left: 1220 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [text("CHATBOT ANALYTICS  ·  CONFIDENTIAL", { size: 15, bold: true, color: MUTED })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [text("Great Place to Work  ·  Chat Summary", { size: 15, color: MUTED })] })] }) },
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
