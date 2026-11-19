import { useEffect, useMemo, useRef, useState } from "react";

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: "overview" },
  { id: "quality", label: "Assistant quality", icon: "quality" },
  { id: "users", label: "Users", icon: "users" },
  { id: "conversations", label: "Conversations", icon: "messages" },
];

const DATE_OPTIONS = ["All dates", "Last 7 days", "Last 30 days", "Last 90 days", "Custom range"];

function Icon({ name, size = 19 }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };
  const paths = {
    overview: <><rect x="3.5" y="3.5" width="7" height="7" rx="1.5" /><rect x="13.5" y="3.5" width="7" height="7" rx="1.5" /><rect x="3.5" y="13.5" width="7" height="7" rx="1.5" /><rect x="13.5" y="13.5" width="7" height="7" rx="1.5" /></>,
    quality: <><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z" /><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z" /></>,
    users: <><circle cx="9" cy="8" r="3.25" /><path d="M3.5 20v-1.4A4.6 4.6 0 0 1 8.1 14h1.8a4.6 4.6 0 0 1 4.6 4.6V20" /><path d="M16 4.9a3.2 3.2 0 0 1 0 6.2M17.2 14.2a4.5 4.5 0 0 1 3.3 4.4V20" /></>,
    messages: <><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-9.5A7.5 7.5 0 0 1 10.5 4H12a8 8 0 0 1 8 7.5Z" /><path d="M8 11h8M8 14.5h5" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.3" /><path d="m16 16 4.3 4.3" /></>,
    filter: <><path d="M4 5h16l-6.2 7.1v5.4l-3.6 1.8v-7.2L4 5Z" /></>,
    calendar: <><rect x="3.5" y="5" width="17" height="15.5" rx="2" /><path d="M7.5 3.5v3M16.5 3.5v3M3.5 9.5h17" /></>,
    chevron: <path d="m8 10 4 4 4-4" />,
    download: <><path d="M12 3.5v11M7.5 10l4.5 4.5 4.5-4.5" /><path d="M4 17v3h16v-3" /></>,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2.5 12h2m15 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
    moon: <path d="M20 15.2A8.4 8.4 0 0 1 8.8 4 8.5 8.5 0 1 0 20 15.2Z" />,
    close: <><path d="m6 6 12 12M18 6 6 18" /></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
  };
  return <svg {...common}>{paths[name] || paths.overview}</svg>;
}

function displayValue(value) {
  if (value == null || value === "") return "Unknown";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function hasSubmission(value) {
  if (value == null || value === false || value === 0) return false;
  if (typeof value === "string") return !["", "false", "no", "0", "null"].includes(value.trim().toLowerCase());
  return Boolean(value);
}

function conversationUserKey(conversation) {
  return conversation.analytics_user_id || conversation.user_id || conversation.anonymous_id || null;
}

function normalizeConversations(raw) {
  return (raw?.conversations || []).map((conversation) => {
    const messages = conversation.messages || [];
    const created = conversation.created_at ? new Date(conversation.created_at) : null;
    const last = conversation.last_message_at ? new Date(conversation.last_message_at) : null;
    const duration = created && last && !Number.isNaN(+created) && !Number.isNaN(+last)
      ? Math.round((+last - +created) / 1000)
      : null;
    const userMessages = messages.filter((message) => message.role === "user").length;
    const assistantMessages = messages.filter((message) => message.role === "assistant").length;
    return {
      ...conversation,
      messages,
      userKey: conversationUserKey(conversation),
      created,
      dateKey: created && !Number.isNaN(+created) ? created.toISOString().slice(0, 10) : null,
      duration,
      userMessages,
      assistantMessages,
      totalMessages: userMessages + assistantMessages,
    };
  });
}

function csvCell(value) {
  const string = value == null ? "" : typeof value === "object" ? JSON.stringify(value) : String(value);
  return `"${string.replaceAll('"', '""')}"`;
}

function downloadCsv(rows, columns, filename) {
  const lines = [columns.map(([, label]) => csvCell(label)).join(",")];
  for (const row of rows) lines.push(columns.map(([key]) => csvCell(row[key])).join(","));
  const blob = new Blob(["\ufeff", lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${Math.round(seconds)} sec`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}

function formatDate(date) {
  if (!date || Number.isNaN(+date)) return "Unknown date";
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(date);
}

function SelectFilter({ label, value, onChange, options }) {
  return (
    <label className="filter-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
      <Icon name="chevron" size={15} />
    </label>
  );
}

function MetricCard({ label, value, icon, detail }) {
  return (
    <article className="metric-card">
      <div className="metric-topline">
        <span>{label}</span>
        <span className="metric-icon"><Icon name={icon} size={17} /></span>
      </div>
      <div className="metric-value">{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </article>
  );
}

function ChartCard({ title, subtitle, children, className = "" }) {
  return (
    <section className={`chart-card ${className}`}>
      <div className="chart-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

function EmptyChart({ text = "No data for these filters" }) {
  return <div className="empty-chart">{text}</div>;
}

function AreaChart({ data, label = "sessions" }) {
  const [hoveredIndex, setHoveredIndex] = useState(null);
  if (!data.length) return <EmptyChart />;
  const width = 760;
  const height = 250;
  const left = 42;
  const right = 12;
  const top = 14;
  const bottom = 34;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const maximum = Math.max(1, ...data.map((item) => item.value));
  const points = data.map((item, index) => ({
    x: left + (data.length === 1 ? plotWidth / 2 : (index / (data.length - 1)) * plotWidth),
    y: top + plotHeight - (item.value / maximum) * plotHeight,
    ...item,
  }));
  const line = points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`).join(" ");
  const area = `${line} L${points.at(-1).x},${top + plotHeight} L${points[0].x},${top + plotHeight} Z`;
  const labels = [points[0], points[Math.floor((points.length - 1) / 2)], points.at(-1)];
  const activePoint = hoveredIndex == null ? null : points[hoveredIndex];
  function setPointFromPointer(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const chartX = ((event.clientX - rect.left) / rect.width) * width;
    const relativeX = Math.max(0, Math.min(plotWidth, chartX - left));
    const nextIndex = data.length === 1 ? 0 : Math.round((relativeX / plotWidth) * (points.length - 1));
    setHoveredIndex(nextIndex);
  }
  return (
    <div className="area-chart-wrap" onPointerLeave={() => setHoveredIndex(null)}>
      <svg className="area-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} over time`} onPointerMove={setPointFromPointer}>
        <defs><linearGradient id="areaFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="var(--accent)" stopOpacity=".22" /><stop offset="100%" stopColor="var(--accent)" stopOpacity="0" /></linearGradient></defs>
        {[0, 0.5, 1].map((fraction) => {
          const y = top + plotHeight - fraction * plotHeight;
          return <g key={fraction}><line className="chart-gridline" x1={left} x2={width - right} y1={y} y2={y} /><text className="axis-label" x={left - 12} y={y + 4} textAnchor="end">{Math.round(maximum * fraction)}</text></g>;
        })}
        <path d={area} fill="url(#areaFill)" />
        <path d={line} className="area-line" />
        {activePoint && <g className="chart-crosshair" aria-hidden="true"><line x1={activePoint.x} x2={activePoint.x} y1={top} y2={top + plotHeight} /><line x1={left} x2={width - right} y1={activePoint.y} y2={activePoint.y} /></g>}
        {points.map((point, index) => <circle key={point.key} className={`area-point ${index === hoveredIndex ? "is-active" : ""}`} cx={point.x} cy={point.y} r={index === hoveredIndex ? "5" : "2.6"} onFocus={() => setHoveredIndex(index)} onBlur={() => setHoveredIndex(null)} tabIndex={0}><title>{point.key}: {point.value} {label}</title></circle>)}
        {labels.map((point, index) => <text key={`${point.key}-${index}`} className="axis-label date-label" x={point.x} y={height - 8} textAnchor={index === 0 ? "start" : index === 2 ? "end" : "middle"}>{point.shortLabel}</text>)}
      </svg>
      {activePoint && <div className={`chart-tooltip ${activePoint.x > width * 0.68 ? "tooltip-left" : ""} ${activePoint.y < 78 ? "tooltip-below" : ""}`} style={{ left: `${(activePoint.x / width) * 100}%`, top: `${(activePoint.y / height) * 100}%` }}><strong>{formatDate(new Date(`${activePoint.key}T00:00:00Z`))}</strong><span>{label}<b>{activePoint.value.toLocaleString()}</b></span></div>}
    </div>
  );
}

function BarList({ data, valueLabel = "sessions", maxItems = 8 }) {
  if (!data.length) return <EmptyChart />;
  const rows = data.slice(0, maxItems);
  const max = Math.max(1, ...rows.map((item) => item.value));
  return (
    <div className="bar-list">
      {rows.map((item) => (
        <div className="bar-row" key={item.label}>
          <div className="bar-label" title={item.label}>{item.label}</div>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${(item.value / max) * 100}%` }} /></div>
          <div className="bar-number">{item.value.toLocaleString()} <span>{valueLabel}</span></div>
        </div>
      ))}
    </div>
  );
}

function DonutChart({ data }) {
  if (!data.length) return <EmptyChart />;
  const total = data.reduce((sum, item) => sum + item.value, 0) || 1;
  const colors = ["var(--accent)", "var(--ink)", "var(--mid)", "var(--soft)", "#b5c4c0", "#d4ddda"];
  const circumference = 2 * Math.PI * 52;
  let offset = 0;
  return (
    <div className="donut-layout">
      <svg className="donut" viewBox="0 0 140 140" role="img" aria-label="Conversation source distribution">
        <circle cx="70" cy="70" r="52" fill="none" stroke="var(--soft)" strokeWidth="16" />
        {data.slice(0, colors.length).map((item, index) => {
          const length = (item.value / total) * circumference;
          const circle = <circle key={item.label} cx="70" cy="70" r="52" fill="none" stroke={colors[index]} strokeWidth="16" strokeDasharray={`${length} ${circumference - length}`} strokeDashoffset={-offset} transform="rotate(-90 70 70)"><title>{item.label}: {item.value}</title></circle>;
          offset += length;
          return circle;
        })}
        <text x="70" y="67" textAnchor="middle" className="donut-total">{total.toLocaleString()}</text>
        <text x="70" y="84" textAnchor="middle" className="donut-caption">sessions</text>
      </svg>
      <div className="legend-list">
        {data.slice(0, colors.length).map((item, index) => <div className="legend-row" key={item.label}><span className="legend-dot" style={{ background: colors[index] }} /><span className="legend-name">{item.label}</span><strong>{Math.round((item.value / total) * 100)}%</strong></div>)}
      </div>
    </div>
  );
}

function ConversationDrawer({ conversation, onClose }) {
  if (!conversation) return null;
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="conversation-drawer" aria-label="Conversation detail">
        <div className="drawer-header"><div><p className="eyebrow">Conversation</p><h2>{conversation.title || conversation.id || "Untitled conversation"}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close conversation"><Icon name="close" /></button></div>
        <div className="drawer-meta"><span>{formatDate(conversation.created)}</span><span>{displayValue(conversation.country)}</span><span>{conversation.userMessages} user messages</span></div>
        <div className="message-list">
          {conversation.messages.map((message, index) => <article className={`message-bubble ${message.role === "assistant" ? "assistant-message" : ""}`} key={message.id || index}><div className="message-role">{displayValue(message.role)}</div><div>{typeof message.content === "string" ? message.content : JSON.stringify(message.content)}</div></article>)}
        </div>
      </aside>
    </div>
  );
}

function App() {
  const [conversations, setConversations] = useState([]);
  const [loadState, setLoadState] = useState("loading");
  const [error, setError] = useState("");
  const [section, setSection] = useState("overview");
  const [searchMode, setSearchMode] = useState("semantic");
  const [textSearchQuery, setTextSearchQuery] = useState("");
  const [semanticResult, setSemanticResult] = useState(null);
  const [semanticStatus, setSemanticStatus] = useState("");
  const [semanticLoading, setSemanticLoading] = useState(false);
  const semanticRequestId = useRef(0);
  const [darkMode, setDarkMode] = useState(() => window.localStorage.getItem("chatbot-theme") === "dark");
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [page, setPage] = useState(0);
  const pageSize = 12;
  const [filters, setFilters] = useState({
    search: "",
    date: "All dates",
    from: "",
    to: "",
    country: "All countries",
    source: "All sources",
    chatbot: "All chatbots",
    sentiment: "All sentiments",
    segment: "All users",
    form: "All submissions",
    minMessages: "Any",
  });

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/conversations.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load conversation data (${response.status}).`);
        return response.json();
      })
      .then((raw) => {
        setConversations(normalizeConversations(raw));
        setLoadState("ready");
      })
      .catch((loadError) => {
        setError(loadError.message);
        setLoadState("error");
      });
  }, []);

  useEffect(() => {
    window.localStorage.setItem("chatbot-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  const choices = useMemo(() => ({
    countries: [...new Set(conversations.map((item) => item.country).filter(Boolean).map(displayValue))].sort(),
    sources: [...new Set(conversations.map((item) => item.source).filter(Boolean).map(displayValue))].sort(),
    chatbots: [...new Set(conversations.map((item) => item.chatbot_id).filter(Boolean).map(displayValue))].sort(),
    sentiments: [...new Set(conversations.map((item) => item.sentiment).filter(Boolean).map(displayValue))].sort(),
  }), [conversations]);

  const maxDate = useMemo(() => conversations.reduce((max, item) => item.dateKey && item.dateKey > max ? item.dateKey : max, ""), [conversations]);
  const minDate = useMemo(() => conversations.reduce((min, item) => item.dateKey && (!min || item.dateKey < min) ? item.dateKey : min, ""), [conversations]);

  async function runSemanticSearch(query = filters.search) {
    const normalizedQuery = query.trim();
    if (normalizedQuery.length < 2) {
      setSemanticStatus("Enter at least two characters to search.");
      return;
    }
    const requestId = ++semanticRequestId.current;
    setSemanticLoading(true);
    setSemanticStatus("Searching conversations across languages…");
    try {
      const documents = conversations.map((item) => {
        const userText = item.messages
          .filter((message) => message.role === "user" && typeof message.content === "string")
          .map((message) => message.content.replace(/\s+/g, " ").trim())
          .filter(Boolean)
          .join(" ");
        let text = [item.title, userText].filter(Boolean).join(". ");
        if (text.length > 400) text = `${text.slice(0, 250)} … ${text.slice(-145)}`;
        return { id: String(item.id), text };
      });
      const response = await fetch("/api/semantic-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: normalizedQuery, documents }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Semantic search failed.");
      if (requestId !== semanticRequestId.current) return;
      setSemanticResult({ query: normalizedQuery, matches: result.matches || [] });
      setSemanticStatus(`Found ${result.matches?.length || 0} matching conversations.`);
    } catch (searchError) {
      if (requestId === semanticRequestId.current) setSemanticStatus(searchError.message || "Semantic search is unavailable. Try text search instead.");
    } finally {
      if (requestId === semanticRequestId.current) setSemanticLoading(false);
    }
  }

  function applyTextSearch(query = filters.search) {
    setTextSearchQuery(query.trim());
  }

  const filtered = useMemo(() => {
    const search = (searchMode === "text" ? textSearchQuery : filters.search).trim().toLowerCase();
    const semanticMatches = searchMode === "semantic" && semanticResult?.query === filters.search.trim()
      ? new Map(semanticResult.matches.map((match, index) => [match.id, { index, score: match.score }]))
      : null;
    let result = conversations.filter((item) => {
      if (filters.date !== "All dates" && filters.date !== "Custom range") {
        const days = Number(filters.date.match(/\d+/)?.[0] || 0);
        const latest = maxDate ? new Date(`${maxDate}T00:00:00Z`) : new Date();
        const start = new Date(latest);
        start.setUTCDate(start.getUTCDate() - days + 1);
        if (!item.dateKey || item.dateKey < start.toISOString().slice(0, 10) || item.dateKey > maxDate) return false;
      }
      if (filters.date === "Custom range") {
        if (filters.from && item.dateKey < filters.from) return false;
        if (filters.to && item.dateKey > filters.to) return false;
      }
      if (filters.country !== "All countries" && displayValue(item.country) !== filters.country) return false;
      if (filters.source !== "All sources" && displayValue(item.source) !== filters.source) return false;
      if (filters.chatbot !== "All chatbots" && displayValue(item.chatbot_id) !== filters.chatbot) return false;
      if (filters.sentiment !== "All sentiments" && displayValue(item.sentiment) !== filters.sentiment) return false;
      if (filters.form === "Submitted" && !hasSubmission(item.form_submission)) return false;
      if (filters.form === "Not submitted" && hasSubmission(item.form_submission)) return false;
      if (filters.minMessages !== "Any" && item.userMessages < Number(filters.minMessages)) return false;
      if (search) {
        if (semanticMatches) return semanticMatches.has(String(item.id));
        const searchable = [item.id, item.title, item.country, item.source, ...item.messages.map((message) => typeof message.content === "string" ? message.content : JSON.stringify(message.content))].join(" ").toLowerCase();
        if (!searchable.includes(search)) return false;
      }
      return true;
    });
    if (filters.segment !== "All users") {
      const counts = new Map();
      for (const item of result) if (item.userKey) counts.set(item.userKey, (counts.get(item.userKey) || 0) + 1);
      result = result.filter((item) => {
        if (!item.userKey) return false;
        const isReturning = (counts.get(item.userKey) || 0) > 1;
        return filters.segment === "Returning users" ? isReturning : !isReturning;
      });
    }
    if (semanticMatches) result.sort((a, b) => semanticMatches.get(String(a.id)).index - semanticMatches.get(String(b.id)).index);
    return result;
  }, [conversations, filters, maxDate, searchMode, semanticResult, textSearchQuery]);

  useEffect(() => setPage(0), [filters]);

  const stats = useMemo(() => {
    const sessions = filtered.length;
    const totalMessages = filtered.reduce((sum, item) => sum + item.totalMessages, 0);
    const durations = filtered.map((item) => item.duration).filter((value) => value != null && Number.isFinite(value) && value >= 0 && value <= 24 * 60 * 60).sort((a, b) => a - b);
    const medianDuration = durations.length ? durations[Math.floor((durations.length - 1) / 2)] : null;
    const p90Duration = durations.length ? durations[Math.floor((durations.length - 1) * 0.9)] : null;
    const users = new Map();
    for (const item of filtered) if (item.userKey) users.set(item.userKey, (users.get(item.userKey) || 0) + 1);
    const oneMessage = filtered.filter((item) => item.userMessages === 1).length;
    const engaged = filtered.filter((item) => item.userMessages >= 2).length;
    return {
      sessions,
      totalMessages,
      averageMessages: sessions ? (totalMessages / sessions).toFixed(2) : "0.00",
      medianDuration,
      p90Duration,
      durationOutliers: filtered.filter((item) => item.duration > 24 * 60 * 60).length,
      bounceRate: sessions ? `${((oneMessage / sessions) * 100).toFixed(1)}%` : "—",
      engagementRate: sessions ? `${((engaged / sessions) * 100).toFixed(1)}%` : "—",
      uniqueUsers: users.size,
      newUsers: [...users.values()].filter((count) => count === 1).length,
      returningUsers: [...users.values()].filter((count) => count > 1).length,
    };
  }, [filtered]);

  const byDay = useMemo(() => {
    const counts = new Map();
    for (const item of filtered) if (item.dateKey) counts.set(item.dateKey, (counts.get(item.dateKey) || 0) + 1);
    const values = [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
    return values.map(([key, value]) => ({ key, value, shortLabel: new Intl.DateTimeFormat("en", { month: "short", day: "numeric", timeZone: "UTC" }).format(new Date(`${key}T00:00:00Z`)) }));
  }, [filtered]);

  const countryCounts = useMemo(() => countBy(filtered, (item) => item.country || "Unknown"), [filtered]);
  const sourceCounts = useMemo(() => countBy(filtered, (item) => item.source || "Unknown"), [filtered]);
  const messageDistribution = useMemo(() => {
    const counts = new Map();
    for (const item of filtered) {
      const label = item.totalMessages >= 10 ? "10+" : String(item.totalMessages);
      counts.set(label, (counts.get(label) || 0) + 1);
    }
    return [...counts.entries()].sort(([a], [b]) => a === "10+" ? 1 : b === "10+" ? -1 : Number(a) - Number(b)).map(([label, value]) => ({ label, value }));
  }, [filtered]);

  const assistantMessages = useMemo(() => filtered.flatMap((item) => item.messages.filter((message) => message.role === "assistant").map((message) => ({ ...message, conversation: item }))), [filtered]);
  const feedbackStats = useMemo(() => {
    const rated = assistantMessages.filter((message) => message.feedback === "up" || message.feedback === "down");
    const positive = rated.filter((message) => message.feedback === "up").length;
    const withSources = assistantMessages.filter((message) => Array.isArray(message.matchedSources) && message.matchedSources.length > 0).length;
    return { rated: rated.length, positive, withSources };
  }, [assistantMessages]);
  const scoreCounts = useMemo(() => {
    const values = assistantMessages.map((item) => Number(item.score)).filter(Number.isFinite);
    const buckets = Array.from({ length: 10 }, (_, index) => ({ label: `${index * 10}–${index * 10 + 10}%`, value: 0 }));
    const normalized = values.map((value) => value <= 1 ? value * 100 : value);
    for (const value of normalized) buckets[Math.min(9, Math.max(0, Math.floor(value / 10)))].value += 1;
    return buckets;
  }, [assistantMessages]);
  const assistantSourceCounts = useMemo(() => countBy(assistantMessages, (item) => item.source || "Unknown"), [assistantMessages]);
  const sessionsPerUser = useMemo(() => {
    const counts = new Map();
    for (const item of filtered) if (item.userKey) counts.set(item.userKey, (counts.get(item.userKey) || 0) + 1);
    const buckets = new Map([["1", 0], ["2", 0], ["3", 0], ["4", 0], ["5+", 0]]);
    for (const value of counts.values()) buckets.set(value >= 5 ? "5+" : String(value), buckets.get(value >= 5 ? "5+" : String(value)) + 1);
    return [...buckets].map(([label, value]) => ({ label: `${label} session${label === "1" ? "" : "s"}`, value }));
  }, [filtered]);
  const byHour = useMemo(() => countBy(filtered, (item) => item.created ? `${String(item.created.getUTCHours()).padStart(2, "0")}:00` : null, 24), [filtered]);
  const byWeekday = useMemo(() => {
    const counts = new Map(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((day) => [day, 0]));
    for (const item of filtered) if (item.created) {
      const day = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][item.created.getUTCDay()];
      counts.set(day, counts.get(day) + 1);
    }
    return [...counts].map(([label, value]) => ({ label, value }));
  }, [filtered]);

  const title = NAV_ITEMS.find((item) => item.id === section)?.label || "Overview";
  const visibleConversations = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const visibleMessages = filtered.flatMap((item) => item.messages.map((message, index) => ({
    ...message,
    conversationId: item.id,
    conversationTitle: item.title,
    messageKey: message.id || `${item.id}-${index}`,
  })));

  function updateFilter(key, value) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  function resetFilters() {
    semanticRequestId.current += 1;
    setSemanticLoading(false);
    setFilters({ search: "", date: "All dates", from: "", to: "", country: "All countries", source: "All sources", chatbot: "All chatbots", sentiment: "All sentiments", segment: "All users", form: "All submissions", minMessages: "Any" });
    setSemanticResult(null);
    setSemanticStatus("");
    setTextSearchQuery("");
  }

  if (loadState === "loading") return <div className="loading-screen"><span className="loading-mark">CA</span><div className="loading-spinner" /><p>Preparing your analytics…</p></div>;
  if (loadState === "error") return <div className="error-screen"><span className="loading-mark">CA</span><h1>We couldn’t load the conversation export</h1><p>{error}</p><p>Make sure the build copied the JSON file into <code>public/data/conversations.json</code>.</p></div>;

  return (
    <div className={`app-shell ${darkMode ? "theme-dark" : ""}`}>
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">CA</div>
          <div className="brand-name">Chatbot Analytics<span>Conversation intelligence</span></div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Dashboard sections">
          {NAV_ITEMS.map((item) => <button key={item.id} className={`nav-item ${section === item.id ? "active" : ""}`} onClick={() => setSection(item.id)} aria-current={section === item.id ? "page" : undefined}><Icon name={item.icon} /><span>{item.label}</span>{section === item.id && <span className="nav-active-dot" />}</button>)}
        </nav>
        <div className="sidebar-footer">
          <div className="sidebar-status"><span className="status-dot" /><span>Live dataset</span></div>
          <p>{conversations.length.toLocaleString()} conversations loaded</p>
          <button className="theme-toggle" onClick={() => setDarkMode((value) => !value)}><Icon name={darkMode ? "sun" : "moon"} size={17} /><span>{darkMode ? "Light appearance" : "Dark appearance"}</span></button>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-context"><span className="topbar-kicker">WORKSPACE</span><span className="topbar-separator">/</span><span>GPTW Demo Test</span></div>
          <div className="topbar-right"><span className="data-live"><span className="status-dot" /> Updated from export</span><button className="mobile-theme-button" onClick={() => setDarkMode((value) => !value)} aria-label="Toggle color theme"><Icon name={darkMode ? "sun" : "moon"} size={18} /></button></div>
        </header>

        <main className="page-content">
          <section className="page-heading">
            <div><p className="eyebrow">ANALYTICS</p><h1>{title}</h1><p className="page-subtitle">Understand engagement, response quality, and how people use your chatbot.</p></div>
            <div className="heading-meta"><span className="source-indicator" /><span>GPTW Demo Test</span></div>
          </section>

          <section className="filter-panel" aria-label="Dashboard filters">
            <div className="filter-toolbar">
              <div className="search-field"><Icon name="search" size={19} /><input value={filters.search} onChange={(event) => { semanticRequestId.current += 1; setSemanticLoading(false); setSemanticResult(null); setSemanticStatus(""); updateFilter("search", event.target.value); }} onKeyDown={(event) => event.key === "Enter" && (searchMode === "semantic" ? runSemanticSearch() : applyTextSearch())} placeholder={searchMode === "semantic" ? "Search by meaning in any language…" : "Search conversations or messages…"} aria-label="Search conversations or messages" />{filters.search && <button onClick={() => { semanticRequestId.current += 1; setSemanticLoading(false); updateFilter("search", ""); setSemanticResult(null); setSemanticStatus(""); setTextSearchQuery(""); }} aria-label="Clear search"><Icon name="close" size={16} /></button>}</div>
              <div className="search-mode-group" role="group" aria-label="Search mode"><button className={searchMode === "text" ? "active" : ""} onClick={() => { setSearchMode("text"); applyTextSearch(); }}>Text</button><button className={searchMode === "semantic" ? "active" : ""} onClick={() => setSearchMode("semantic")}>Semantic</button></div>
              <button className="semantic-search-button" onClick={() => searchMode === "semantic" ? runSemanticSearch() : applyTextSearch()} disabled={(searchMode === "semantic" && semanticLoading) || !filters.search.trim()}>{searchMode === "semantic" && semanticLoading ? <span className="search-spinner" /> : <Icon name="search" size={15} />}{searchMode === "semantic" && semanticLoading ? "Searching" : "Search"}</button>
              <label className="date-select"><Icon name="calendar" size={17} /><select value={filters.date} onChange={(event) => updateFilter("date", event.target.value)} aria-label="Date window">{DATE_OPTIONS.map((option) => <option key={option}>{option}</option>)}</select><Icon name="chevron" size={15} /></label>
              <button className={`filter-toggle ${filtersOpen ? "is-open" : ""}`} onClick={() => setFiltersOpen((value) => !value)}><Icon name="filter" size={16} /> Filters <span className="filter-count">{[filters.country !== "All countries", filters.source !== "All sources", filters.chatbot !== "All chatbots", filters.sentiment !== "All sentiments", filters.segment !== "All users", filters.form !== "All submissions", filters.minMessages !== "Any"].filter(Boolean).length || ""}</span></button>
            </div>
            {searchMode === "semantic" && semanticStatus && <div className={`semantic-status ${semanticStatus.includes("could not") || semanticStatus.includes("not configured") || semanticStatus.includes("too large") ? "has-error" : ""}`} role="status">{semanticStatus}</div>}
            {filters.date === "Custom range" && <div className="custom-date-row"><label>From<input type="date" min={minDate} max={maxDate} value={filters.from} onChange={(event) => updateFilter("from", event.target.value)} /></label><label>To<input type="date" min={minDate} max={maxDate} value={filters.to} onChange={(event) => updateFilter("to", event.target.value)} /></label></div>}
            {filtersOpen && <div className="advanced-filters">
              <SelectFilter label="Country" value={filters.country} onChange={(value) => updateFilter("country", value)} options={["All countries", ...choices.countries]} />
              <SelectFilter label="Source" value={filters.source} onChange={(value) => updateFilter("source", value)} options={["All sources", ...choices.sources]} />
              <SelectFilter label="Chatbot" value={filters.chatbot} onChange={(value) => updateFilter("chatbot", value)} options={["All chatbots", ...choices.chatbots]} />
              <SelectFilter label="Sentiment" value={filters.sentiment} onChange={(value) => updateFilter("sentiment", value)} options={["All sentiments", ...choices.sentiments]} />
              <SelectFilter label="User type" value={filters.segment} onChange={(value) => updateFilter("segment", value)} options={["All users", "New users", "Returning users"]} />
              <SelectFilter label="Form" value={filters.form} onChange={(value) => updateFilter("form", value)} options={["All submissions", "Submitted", "Not submitted"]} />
              <SelectFilter label="Min. user messages" value={filters.minMessages} onChange={(value) => updateFilter("minMessages", value)} options={["Any", "1", "2", "3", "5", "10"]} />
              <button className="reset-filters" onClick={resetFilters}>Reset filters</button>
            </div>}
            <div className="result-line"><span>Showing <strong>{filtered.length.toLocaleString()}</strong> of {conversations.length.toLocaleString()} conversations</span>{(filters.search || filters.country !== "All countries" || filters.source !== "All sources" || filters.chatbot !== "All chatbots" || filters.sentiment !== "All sentiments" || filters.segment !== "All users" || filters.form !== "All submissions" || filters.date !== "All dates" || filters.minMessages !== "Any") && <button onClick={resetFilters}>Clear all</button>}</div>
          </section>

          {section === "overview" && <>
            <section className="metrics-grid">
              <MetricCard label="Sessions" value={stats.sessions.toLocaleString()} icon="messages" detail="Total conversations" />
              <MetricCard label="Total messages" value={stats.totalMessages.toLocaleString()} icon="overview" detail={`${stats.averageMessages} per session`} />
              <MetricCard label="Median duration" value={formatDuration(stats.medianDuration)} icon="calendar" detail={`P90 ${formatDuration(stats.p90Duration)} · ${stats.durationOutliers} over 24h excluded`} />
              <MetricCard label="Bounce rate" value={stats.bounceRate} icon="arrow" detail="Sessions with one user message" />
              <MetricCard label="Unique users" value={stats.uniqueUsers.toLocaleString()} icon="users" detail={`${stats.newUsers.toLocaleString()} new`} />
              <MetricCard label="Returning users" value={stats.returningUsers.toLocaleString()} icon="users" detail={`${stats.engagementRate} engaged sessions`} />
            </section>
            <section className="charts-grid overview-charts">
              <ChartCard title="Sessions over time" subtitle="Daily conversation volume" className="wide-chart"><AreaChart data={byDay} /></ChartCard>
              <ChartCard title="Messages per session" subtitle="Conversation length distribution"><BarList data={messageDistribution} valueLabel="chats" maxItems={10} /></ChartCard>
              <ChartCard title="Sessions by country" subtitle="Top countries"><BarList data={countryCounts} /></ChartCard>
              <ChartCard title="Sessions by source" subtitle="Where conversations started"><DonutChart data={sourceCounts} /></ChartCard>
            </section>
          </>}

          {section === "quality" && <>
            <section className="metrics-grid compact-metrics">
              <MetricCard label="Assistant messages" value={assistantMessages.length.toLocaleString()} icon="messages" detail="For selected conversations" />
              <MetricCard label="Scored responses" value={assistantMessages.filter((message) => message.score != null && message.score !== "" && Number.isFinite(Number(message.score))).length.toLocaleString()} icon="quality" detail="Responses with an available score" />
              <MetricCard label="Responses with sources" value={`${assistantMessages.length ? ((feedbackStats.withSources / assistantMessages.length) * 100).toFixed(1) : "0.0"}%`} icon="overview" detail={`${feedbackStats.withSources.toLocaleString()} linked to at least one source`} />
              <MetricCard label="Positive feedback" value={feedbackStats.rated ? `${((feedbackStats.positive / feedbackStats.rated) * 100).toFixed(1)}%` : "—"} icon="quality" detail={`${feedbackStats.positive} positive of ${feedbackStats.rated} rated responses`} />
              <MetricCard label="Source types" value={assistantSourceCounts.length.toLocaleString()} icon="overview" detail="Distinct assistant source labels" />
            </section>
            <section className="charts-grid">
              <ChartCard title="Assistant score distribution" subtitle="Scores grouped into deciles"><BarList data={scoreCounts} valueLabel="answers" maxItems={10} /></ChartCard>
              <ChartCard title="Assistant sources" subtitle="Response source labels"><DonutChart data={assistantSourceCounts} /></ChartCard>
            </section>
            {!assistantMessages.length && <div className="empty-state"><span className="empty-state-mark"><Icon name="quality" size={23} /></span><h3>No assistant responses in this view</h3><p>Adjust the filters to see quality and source data.</p></div>}
          </>}

          {section === "users" && <>
            <section className="metrics-grid compact-metrics">
              <MetricCard label="Unique users" value={stats.uniqueUsers.toLocaleString()} icon="users" detail="Known or anonymous IDs" />
              <MetricCard label="New users" value={stats.newUsers.toLocaleString()} icon="arrow" detail="One session in this selection" />
              <MetricCard label="Returning users" value={stats.returningUsers.toLocaleString()} icon="users" detail="More than one session" />
            </section>
            <section className="charts-grid">
              <ChartCard title="Sessions per user" subtitle="Repeat usage distribution"><BarList data={sessionsPerUser} valueLabel="users" /></ChartCard>
              <ChartCard title="Activity by weekday" subtitle="Conversation start day"><BarList data={byWeekday} valueLabel="sessions" maxItems={7} /></ChartCard>
              <ChartCard title="Activity by hour" subtitle="Conversation start time (UTC)" className="wide-chart"><BarList data={byHour} valueLabel="sessions" maxItems={24} /></ChartCard>
            </section>
          </>}

          {section === "conversations" && <section className="data-section">
            <div className="data-heading"><div><h2>Conversation data</h2><p>Browse the filtered conversations and their messages.</p></div><button className="export-button" onClick={() => downloadCsv(filtered, [["id", "Conversation ID"], ["title", "Title"], ["created_at", "Created at"], ["country", "Country"], ["source", "Source"], ["chatbot_id", "Chatbot"], ["sentiment", "Sentiment"], ["userMessages", "User messages"], ["assistantMessages", "Assistant messages"]], "conversations.csv")}><Icon name="download" size={17} /> Export CSV</button></div>
            <div className="table-wrap"><table><thead><tr><th>Conversation</th><th>Started</th><th>Country</th><th>Source</th><th>Messages</th><th>Sentiment</th><th /></tr></thead><tbody>
              {visibleConversations.map((item) => <tr key={item.id}><td><button className="conversation-link" onClick={() => setSelectedConversation(item)}><span className="conversation-avatar">{(item.title || item.id || "C").slice(0, 1).toUpperCase()}</span><span><strong>{item.title || item.id || "Untitled conversation"}</strong><small>{item.id || "No ID"}</small></span></button></td><td>{formatDate(item.created)}</td><td>{displayValue(item.country)}</td><td><span className="table-pill">{displayValue(item.source)}</span></td><td>{item.totalMessages}</td><td>{item.sentiment ? <span className={`sentiment-pill sentiment-${String(item.sentiment).toLowerCase().replace(/[^a-z]/g, "")}`}>{displayValue(item.sentiment)}</span> : "—"}</td><td><button className="open-row" onClick={() => setSelectedConversation(item)} aria-label="Open conversation"><Icon name="arrow" size={16} /></button></td></tr>)}
              {!visibleConversations.length && <tr><td colSpan="7"><EmptyChart text="No conversations match the current filters" /></td></tr>}
            </tbody></table></div>
            <div className="table-footer"><span>{filtered.length ? `Showing ${page * pageSize + 1}–${Math.min((page + 1) * pageSize, filtered.length)} of ${filtered.length.toLocaleString()}` : "No results"}</span><div><button disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</button><button disabled={(page + 1) * pageSize >= filtered.length} onClick={() => setPage((value) => value + 1)}>Next</button></div></div>
            <div className="messages-export"><div><h3>Messages</h3><p>{visibleMessages.length.toLocaleString()} messages in the selected conversations</p></div><button className="secondary-button" onClick={() => downloadCsv(visibleMessages, [["conversationId", "Conversation ID"], ["role", "Role"], ["type", "Type"], ["content", "Content"], ["source", "Assistant source"], ["score", "Score"]], "messages.csv")}><Icon name="download" size={16} /> Export messages</button></div>
          </section>}

          <footer className="page-footer">Chatbot analytics <span>·</span> Filtered views update across charts and exports</footer>
        </main>
      </div>
      <ConversationDrawer conversation={selectedConversation} onClose={() => setSelectedConversation(null)} />
    </div>
  );
}

function countBy(items, getLabel, limit = Infinity) {
  const counts = new Map();
  for (const item of items) {
    const label = getLabel(item);
    if (label != null) counts.set(String(label), (counts.get(String(label)) || 0) + 1);
  }
  return [...counts.entries()].map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value || a.label.localeCompare(b.label)).slice(0, limit);
}

export default App;
