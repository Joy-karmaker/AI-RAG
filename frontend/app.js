const API_BASE = "http://127.0.0.1:8000";

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
  uploadButton: document.querySelector("#uploadButton"),
  refreshButton: document.querySelector("#refreshButton"),
  documentList: document.querySelector("#documentList"),
  selectedTitle: document.querySelector("#selectedTitle"),
  selectedMeta: document.querySelector("#selectedMeta"),
  questionInput: document.querySelector("#questionInput"),
  topK: document.querySelector("#topK"),
  answerToggle: document.querySelector("#answerToggle"),
  askButton: document.querySelector("#askButton"),
  promptButton: document.querySelector("#promptButton"),
  answerBox: document.querySelector("#answerBox"),
  answerModel: document.querySelector("#answerModel"),
  sourceCount: document.querySelector("#sourceCount"),
  sourcesList: document.querySelector("#sourcesList"),
  promptPanel: document.querySelector("#promptPanel"),
  promptBox: document.querySelector("#promptBox"),
  clearPromptButton: document.querySelector("#clearPromptButton"),
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

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("chunk_size", elements.chunkSize.value);
  formData.append("overlap", elements.overlap.value);

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
    const payload = {
      query,
      top_k: Number(elements.topK.value),
      answer: !promptOnly && elements.answerToggle.checked,
      dry_run_answer: promptOnly,
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
      renderAnswer(data.answer || summarizeResults(data.results || []));
      elements.answerModel.textContent = data.answer_model || "retrieval";
    }

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
    const button = document.createElement("button");
    button.type = "button";
    button.className = "document-card";
    button.classList.toggle("active", documentRecord.document_id === state.selectedDocumentId);
    button.innerHTML = `
      <strong>${escapeHtml(documentRecord.filename)}</strong>
      <small>${documentRecord.chunk_count} chunks | ${documentRecord.vector_count} vectors</small>
      <small>${escapeHtml(documentRecord.document_id)}</small>
    `;
    button.addEventListener("click", () => {
      state.selectedDocumentId = documentRecord.document_id;
      renderDocuments();
      renderSelectedDocument();
    });
    elements.documentList.appendChild(button);
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
    <span>${selected.embedding_dimensions} dims</span>
  `;
}

function renderAnswer(text) {
  elements.answerBox.classList.remove("empty-state");
  elements.answerBox.textContent = text;
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
    article.innerHTML = `
      <header>
        <span>Chunk ${result.chunk_index}</span>
        <span>${Number(result.score).toFixed(4)}</span>
      </header>
      <p>${escapeHtml(result.text_preview || result.text || "")}</p>
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
  const data = text ? JSON.parse(text) : {};

  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }

  return data;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
