const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("structure-file");
const dropzone = document.getElementById("dropzone");
const uploadPreview = document.getElementById("upload-preview");
const uploadStatus = document.getElementById("upload-status");
const predictionCard = document.getElementById("prediction-card");
const predictButton = document.getElementById("predict-button");

if (!uploadForm || !fileInput || !dropzone || !uploadPreview || !uploadStatus || !predictionCard || !predictButton) {
  throw new Error("Frontend failed to initialize because required UI elements are missing.");
}

const state = {
  uploadId: "",
  uploaded: false,
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setTheme(theme) {
  document.body.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
}

function loadTheme() {
  const savedTheme = localStorage.getItem("theme");
  setTheme(savedTheme === "dark" ? "dark" : "light");
}

function setUploadStatus(message, success = false) {
  uploadStatus.textContent = message;
  uploadStatus.classList.toggle("success", success);
}

function setPredictionCard(prediction) {
  if (!prediction) {
    predictionCard.classList.add("empty");
    predictionCard.innerHTML = `
      <p class="prediction-value">No prediction yet</p>
      <p class="prediction-meta">Upload a CIF file first.</p>
    `;
    return;
  }

  predictionCard.classList.remove("empty");
  predictionCard.innerHTML = `
    <p class="prediction-value">${prediction.ionic_conductivity_s_cm} S/cm</p>
    <p class="prediction-meta"><strong>Log Sigma:</strong> ${prediction.ionic_conductivity_log_sigma}</p>
    <p class="prediction-meta"><strong>Formula:</strong> ${escapeHtml(prediction.formula || "Unknown")}</p>
    <p class="prediction-meta"><strong>Source:</strong> ${escapeHtml(prediction.source)}</p>
    <p class="prediction-meta"><strong>Status:</strong> Prediction completed successfully.</p>
  `;
}

async function uploadSelectedFile(event) {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setUploadStatus("Please choose a CIF file first.");
    uploadPreview.textContent = "Waiting for file upload.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setUploadStatus("Uploading file...");
  uploadPreview.textContent = "Uploading file...";

  const response = await fetch("/api/upload-structure", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();

  if (!response.ok) {
    state.uploadId = "";
    state.uploaded = false;
    predictButton.disabled = true;
    setUploadStatus(payload.detail || "Upload failed.");
    uploadPreview.textContent = "Upload failed.";
    return;
  }

  state.uploadId = payload.upload_id;
  state.uploaded = true;
  predictButton.disabled = false;
  setUploadStatus(`File uploaded successfully: ${payload.filename}`, true);
  uploadPreview.textContent = `${payload.structure_type} uploaded\nUpload ID: ${payload.upload_id}\n\n${payload.preview}`;
}

async function predictConductivity() {
  if (!state.uploaded || !state.uploadId) {
    setPredictionCard(null);
    return;
  }

  predictButton.disabled = true;
  predictButton.textContent = "Predicting...";

  const response = await fetch("/api/predict-conductivity", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      upload_id: state.uploadId,
    }),
  });
  const payload = await response.json();

  predictButton.disabled = false;
  predictButton.textContent = "Predict Conductivity";

  if (!response.ok) {
    predictionCard.classList.remove("empty");
    predictionCard.innerHTML = `
      <p class="prediction-value">Prediction failed</p>
      <p class="prediction-meta">${escapeHtml(payload.detail || "Something went wrong.")}</p>
    `;
    return;
  }

  setPredictionCard(payload);
}

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const files = event.dataTransfer.files;
  if (files.length) {
    fileInput.files = files;
    setUploadStatus(`Selected file: ${files[0].name}`);
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    setUploadStatus(`Selected file: ${file.name}`);
    uploadPreview.textContent = "File selected. Click Upload File to continue.";
  }
});

uploadForm.addEventListener("submit", (event) => {
  uploadSelectedFile(event).catch((error) => {
    setUploadStatus(error.message || "Upload failed.");
  });
});

predictButton.addEventListener("click", () => {
  predictConductivity().catch((error) => {
    predictionCard.classList.remove("empty");
    predictionCard.innerHTML = `
      <p class="prediction-value">Prediction failed</p>
      <p class="prediction-meta">${escapeHtml(error.message || "Something went wrong.")}</p>
    `;
    predictButton.disabled = false;
    predictButton.textContent = "Predict Conductivity";
  });
});

loadTheme();
setPredictionCard(null);
