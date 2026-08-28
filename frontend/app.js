const API_BASE = getApiBase();

const state = {
  documents: [],
  selectedDocumentId: null,
  selectedFile: null,
};

const elements = {
  apiStatus: document.querySelector("#apiStatus"),
  fileInput: document.querySelector("#fileInput"),
  fileName: document.querySelector("#fileName"),
  dropZone: document.querySelector("#dropZone"),
  chunkSize: document.querySelector("#chunkSize"),
  overlap: document.querySelector("#overlap"),
  embeddingProvider: document.querySelector("#embeddingProvider"),
  uploadButton: document.querySelector("#uploadButton"),
  refreshButton: document.querySelector("#refreshButton"),
  documentList: document.querySelector("#documentList"),
  selectedTitle: document.querySelector("#selectedTitle"),
  selectedMeta: document.querySelector("#selectedMeta"),
  questionInput: document.querySelector("#questionInput"),
  topK: document.querySelector("#topK"),
  modelSelect: document.querySelector("#modelSelect"),
  answerToggle: document.querySelector("#answerToggle"),
  debugToggle: document.querySelector("#debugToggle"),
  askButton: document.querySelector("#askButton"),
  promptButton: document.querySelector("#promptButton"),
  answerBox: document.querySelector("#answerBox"),
  answerModel: document.querySelector("#answerModel"),
  sourceCount: document.querySelector("#sourceCount"),
  sourcesList: document.querySelector("#sourcesList"),
  promptPanel: document.querySelector("#promptPanel"),
  promptBox: document.querySelector("#promptBox"),
  clearPromptButton: document.querySelector("#clearPromptButton"),
  debugPanel: document.querySelector("#debugPanel"),
  debugBox: document.querySelector("#debugBox"),
  clearDebugButton: document.querySelector("#clearDebugButton"),
  messageBar: document.querySelector("#messageBar"),
};

init();

function init() {
  bindEvents();
  checkApi();
  refreshDocuments();
}

function bindEvents() {
  elements.fileInput.addEventListener("change", () => {
    setSelectedFile(elements.fileInput.files[0] || null);
  });

  elements.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });

  elements.dropZone.addEventListener("dragleave", () => {
    elements.dropZone.classList.remove("dragging");
  });

  elements.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
    setSelectedFile(event.dataTransfer.files[0] || null);
  });

  elements.uploadButton.addEventListener("click", uploadDocument);
  elements.refreshButton.addEventListener("click", refreshDocuments);
  elements.askButton.addEventListener("click", () => queryDocument({ promptOnly: false }));
  elements.promptButton.addEventListener("click", () => queryDocument({ promptOnly: true }));
  elements.clearPromptButton.addEventListener("click", () => {
    elements.promptPanel.classList.add("hidden");
    elements.promptBox.textContent = "";
  });
  elements.clearDebugButton.addEventListener("click", clearDebugPanel);
}

async function checkApi() {
  try {
    const health = await apiFetch("/health");
    elements.apiStatus.textContent = `OK ${health.processed_documents}`;
    elements.apiStatus.classList.add("online");
    elements.apiStatus.classList.remove("offline");
  } catch (error) {
    elements.apiStatus.textContent = "OFF";
    elements.apiStatus.classList.add("offline");
    elements.apiStatus.classList.remove("online");
    showMessage(error.message, "error");
  }
}

async function refreshDocuments() {
  try {
    const data = await apiFetch("/documents");
    state.documents = data.documents || [];

    if (!state.selectedDocumentId && state.documents.length > 0) {
      state.selectedDocumentId = state.documents[0].document_id;
    }

    if (!state.documents.some((document) => document.document_id === state.selectedDocumentId)) {
      state.selectedDocumentId = state.documents[0]?.document_id || null;
    }

    renderDocuments();
    renderSelectedDocument();
    checkApi();
  } catch (error) {
    showMessage(error.message, "error");
  }
}

async function uploadDocument() {
  if (!state.selectedFile) {
    showMessage("Choose a document first.", "error");
    return;
  }

  if (state.selectedFile.size === 0) {
    showMessage("The selected file is empty. Choose the real PDF file, not an empty copied curl body.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("chunk_size", elements.chunkSize.value);
  formData.append("overlap", elements.overlap.value);
  formData.append("embedding_provider", elements.embeddingProvider.value);

  setBusy(true);
  showMessage("Uploading...");

  try {
    const document = await apiFetch("/documents/upload", {
      method: "POST",
      body: formData,
    });
    state.selectedDocumentId = document.document_id;
    await refreshDocuments();
    renderAnswer("Document ready.");
    renderSources([]);
    showMessage("Document uploaded.", "success");
  } catch (error) {
    showMessage(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function queryDocument({ promptOnly }) {
  const selected = getSelectedDocument();
  const query = elements.questionInput.value.trim();

  if (!selected) {
    showMessage("Upload or select a document.", "error");
    return;
  }

  if (!query) {
    showMessage("Enter a question.", "error");
    return;
  }

  setBusy(true);
  showMessage(promptOnly ? "Building prompt..." : "Searching...");

  try {
    const debug = elements.debugToggle.checked;
    const payload = {
      query,
      top_k: Number(elements.topK.value),
      answer: !promptOnly && elements.answerToggle.checked,
      dry_run_answer: promptOnly,
      llm_model: elements.modelSelect.value,
      debug,
      agentic: true,
    };
    const data = await apiFetch(`/documents/${selected.document_id}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    renderSources(data.results || []);

    if (promptOnly) {
      elements.promptPanel.classList.remove("hidden");
      elements.promptBox.textContent = data.prompt || "";
      renderAnswer("Prompt generated.");
      elements.answerModel.textContent = "";
    } else {
      elements.promptPanel.classList.add("hidden");
      renderAnswer(data.answer || summarizeResults(data.results || []), data.verification);
      elements.answerModel.textContent = formatAnswerModel(data.answer_model, data.verification, data.routing);
    }

    renderDebug(debug ? data.trace : null);

    showMessage("Query complete.", "success");
  } catch (error) {
    showMessage(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function renderDocuments() {
  elements.documentList.innerHTML = "";

  if (state.documents.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No documents";
    elements.documentList.appendChild(empty);
    return;
  }

  for (const documentRecord of state.documents) {
    const card = document.createElement("article");
    card.className = "document-card";
    card.classList.toggle("active", documentRecord.document_id === state.selectedDocumentId);
    card.innerHTML = `
      <button class="document-select" type="button">
        <strong>${escapeHtml(documentRecord.filename)}</strong>
        <small>${documentRecord.chunk_count} chunks | ${documentRecord.vector_count} vectors</small>
        <small>${escapeHtml(documentRecord.document_id)}</small>
      </button>
      <button class="document-delete" type="button" title="Delete document">Delete</button>
    `;

    card.querySelector(".document-select").addEventListener("click", () => {
      state.selectedDocumentId = documentRecord.document_id;
      renderDocuments();
      renderSelectedDocument();
    });

    card.querySelector(".document-delete").addEventListener("click", () => {
      deleteDocument(documentRecord.document_id);
    });

    elements.documentList.appendChild(card);
  }
}

async function deleteDocument(documentId) {
  const documentRecord = state.documents.find((document) => document.document_id === documentId);
  const filename = documentRecord?.filename || documentId;

  if (!window.confirm(`Delete ${filename} and clear its vectors from memory?`)) {
    return;
  }

  setBusy(true);
  showMessage("Deleting document...");

  try {
    const result = await apiFetch(`/documents/${documentId}`, {
      method: "DELETE",
    });

    if (state.selectedDocumentId === documentId) {
      state.selectedDocumentId = null;
      renderAnswer("Document deleted.");
      renderSources([]);
      elements.answerModel.textContent = "";
      elements.promptPanel.classList.add("hidden");
      elements.promptBox.textContent = "";
      clearDebugPanel();
    }

    await refreshDocuments();
    showMessage(`Deleted ${result.filename}; cleared ${result.vectors_deleted} vectors.`, "success");
  } catch (error) {
    showMessage(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function renderSelectedDocument() {
  const selected = getSelectedDocument();

  if (!selected) {
    elements.selectedTitle.textContent = "No document selected";
    elements.selectedMeta.innerHTML = "";
    return;
  }

  elements.selectedTitle.textContent = selected.filename;
  elements.selectedMeta.innerHTML = `
    <span>${selected.chunk_count} chunks</span>
    <span>${selected.vector_count} vectors</span>
    <span>${escapeHtml(selected.embedding_provider || "local")}</span>
    <span>${escapeHtml(selected.embedding_model || `${selected.embedding_dimensions} dims`)}</span>
  `;
}

function renderAnswer(text, verification = null) {
  elements.answerBox.classList.remove("empty-state");

  if (!verification) {
    elements.answerBox.textContent = text;
    return;
  }

  elements.answerBox.textContent = [
    `Verification: ${verification.verification_status}`,
    verification.notes ? `Notes: ${verification.notes}` : "",
    "",
    text,
  ].filter(Boolean).join("\n");
}

function renderDebug(trace) {
  if (!trace) {
    clearDebugPanel();
    return;
  }

  elements.debugBox.textContent = formatTrace(trace);
  elements.debugPanel.classList.remove("hidden");
}

function clearDebugPanel() {
  elements.debugPanel.classList.add("hidden");
  elements.debugBox.textContent = "";
}

function formatTrace(trace) {
  const lines = [`Query: ${trace.query}`];
  const documentLabel = trace.filename
    ? `${trace.filename} (${trace.document_id})`
    : trace.document_id;
  lines.push(`Document: ${documentLabel}`);

  if (trace.embedding) {
    const embedding = trace.embedding;
    lines.push(
      `Embedding: ${embedding.provider} / ${embedding.model} (${embedding.dimensions} dims)`
    );
  }

  if (trace.retrieval) {
    const retrieval = trace.retrieval;
    lines.push(
      `Retriever candidates: ${retrieval.candidate_count} ` +
        `(query_type: ${retrieval.query_type}, mode: ${retrieval.query_mode})`
    );
    const kept = (retrieval.selected_chunks || [])
      .map((chunk) => chunk.chunk_index)
      .join(", ");
    lines.push(`Reranker (${retrieval.reranker}) selected: chunks ${kept || "-"}`);
    for (const chunk of retrieval.selected_chunks || []) {
      lines.push(
        `  - chunk ${chunk.chunk_index}: ` +
          `score ${formatScore(chunk.score)}, ` +
          `vector ${formatScore(chunk.vector_score)}, ` +
          `lexical ${formatScore(chunk.lexical_score)}, ` +
          `rerank ${formatScore(chunk.rerank_score)}`
      );
    }
  }

  if (trace.prompt) {
    lines.push(
      `Prompt: ${trace.prompt.chars} chars (~${trace.prompt.tokens_estimate} tokens)`
    );
  }

  if (trace.answer) {
    lines.push(`Answer model: ${trace.answer.model}`);
  }

  if (trace.verification) {
    lines.push(`Verification: ${trace.verification.status}`);
  }

  for (const stage of trace.stages || []) {
    lines.push(`Stage ${stage.name}: ${stage.duration_ms} ms`);
  }

  if (trace.latency_ms) {
    lines.push(`Latency: ${trace.latency_ms.total} ms`);
  }

  return lines.join("\n");
}

function formatScore(value) {
  if (value === null || value === undefined) {
    return "-";
  }

  return Number(value).toFixed(4);
}

function formatAnswerModel(model, verification, routing = null) {
  const route = routing?.action ? ` | ${routing.action}` : "";
  const label = model || "retrieval";

  if (!verification) {
    return `${label}${route}`;
  }

  return `${label} | ${verification.verification_status}${route}`;
}

function renderSources(results) {
  elements.sourcesList.innerHTML = "";
  elements.sourceCount.textContent = results.length ? `${results.length} found` : "";

  if (results.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No sources";
    elements.sourcesList.appendChild(empty);
    return;
  }

  for (const result of results) {
    const article = document.createElement("article");
    article.className = "source-card";
    const scoreLabel = result.rerank_strategy
      ? `${result.rerank_strategy} ${Number(result.rerank_score ?? result.score).toFixed(4)}`
      : Number(result.score).toFixed(4);
    article.innerHTML = `
      <header>
        <span>Chunk ${result.chunk_index}</span>
        <span>${escapeHtml(scoreLabel)}</span>
      </header>
      <pre class="source-full-text">${escapeHtml(result.text || result.text_preview || "")}</pre>
    `;
    elements.sourcesList.appendChild(article);
  }
}

function summarizeResults(results) {
  if (results.length === 0) {
    return "No matching chunks found.";
  }

  return results
    .map((result) => `Chunk ${result.chunk_index}: ${result.text_preview}`)
    .join("\n\n");
}

function setSelectedFile(file) {
  state.selectedFile = file;
  elements.fileName.textContent = file ? file.name : "No file";
}

function getSelectedDocument() {
  return state.documents.find((document) => document.document_id === state.selectedDocumentId);
}

function setBusy(isBusy) {
  elements.uploadButton.disabled = isBusy;
  elements.askButton.disabled = isBusy;
  elements.promptButton.disabled = isBusy;
}

function showMessage(message, type = "") {
  elements.messageBar.textContent = message;
  elements.messageBar.className = `message-bar ${type}`.trim();
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const text = await response.text();
  const contentType = response.headers.get("content-type") || "";
  const data = parseResponseBody(text, contentType);

  if (!response.ok) {
    throw new Error(formatApiError(data, response.status));
  }

  return data;
}

function parseResponseBody(text, contentType) {
  if (!text) {
    return {};
  }

  if (!contentType.includes("application/json")) {
    return { detail: text };
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    return { detail: text };
  }
}

function formatApiError(data, status) {
  const detail = data?.detail;

  if (Array.isArray(detail)) {
    return detail
      .map((item) => item.msg || JSON.stringify(item))
      .join("; ");
  }

  if (detail) {
    return String(detail);
  }

  return `Request failed with ${status}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getApiBase() {
  if (window.location.port === "8000") {
    return window.location.origin;
  }

  return `${window.location.protocol}//${window.location.hostname}:8000`;
}
