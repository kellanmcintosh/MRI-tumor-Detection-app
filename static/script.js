// Minimal vanilla-JS frontend for the MRI tumor classifier.
// No build step, no framework: talks to the FastAPI backend's /predict
// endpoint directly via fetch + FormData.

(() => {
  "use strict";

  // Must match app/config.py's CLASS_NAMES order.
  const CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"];

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzoneEmpty = document.getElementById("dropzone-empty");
  const dropzonePreview = document.getElementById("dropzone-preview");
  const previewImage = document.getElementById("preview-image");
  const changeImageBtn = document.getElementById("change-image-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const errorMessage = document.getElementById("error-message");

  const resultsPanel = document.getElementById("results-panel");
  const loadingState = document.getElementById("loading-state");
  const resultsContent = document.getElementById("results-content");
  const predictedClassEl = document.getElementById("predicted-class");
  const confidenceList = document.getElementById("confidence-list");
  const gradcamImage = document.getElementById("gradcam-image");

  let selectedFile = null;
  let previewObjectUrl = null;

  // ---------- Selection & preview ----------

  function openFilePicker() {
    fileInput.click();
  }

  dropzone.addEventListener("click", openFilePicker);
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  });

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
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  changeImageBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    resetToEmptyState();
  });

  function handleFileSelected(file) {
    hideError();
    hideResults();

    // Client-side sanity check for a fast UX signal; the backend is the
    // real source of truth and still validates/rejects non-images itself.
    if (!file.type || !file.type.startsWith("image/")) {
      showError(
        `Unsupported file type: "${file.type || "unknown"}". Please choose an image file.`
      );
      resetToEmptyState({ keepError: true });
      return;
    }

    selectedFile = file;

    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }
    previewObjectUrl = URL.createObjectURL(file);
    previewImage.src = previewObjectUrl;

    dropzoneEmpty.hidden = true;
    dropzonePreview.hidden = false;
    changeImageBtn.hidden = false;
    analyzeBtn.disabled = false;
  }

  function resetToEmptyState({ keepError = false } = {}) {
    selectedFile = null;
    fileInput.value = "";
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
    previewImage.removeAttribute("src");

    dropzoneEmpty.hidden = false;
    dropzonePreview.hidden = true;
    changeImageBtn.hidden = true;
    analyzeBtn.disabled = true;

    hideResults();
    if (!keepError) {
      hideError();
    }
  }

  // ---------- Analyze ----------

  analyzeBtn.addEventListener("click", analyze);

  async function analyze() {
    if (!selectedFile) {
      return;
    }

    hideError();
    analyzeBtn.disabled = true;
    changeImageBtn.disabled = true;
    showLoading();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      let body;
      try {
        body = await response.json();
      } catch (parseError) {
        throw new Error("The server returned an unexpected response. Please try again.");
      }

      if (!response.ok) {
        const detail =
          (body && body.detail) || `Request failed with status ${response.status}.`;
        throw new Error(detail);
      }

      renderResults(body);
    } catch (err) {
      hideResults();
      showError(
        err instanceof Error && err.message
          ? err.message
          : "Something went wrong analyzing this image. Please try again."
      );
    } finally {
      analyzeBtn.disabled = false;
      changeImageBtn.disabled = false;
    }
  }

  function renderResults(body) {
    const predictedClass = body.predicted_class;
    const confidences = body.confidences || {};

    predictedClassEl.textContent = predictedClass;
    gradcamImage.src = `data:image/png;base64,${body.gradcam_overlay}`;

    confidenceList.innerHTML = "";
    CLASS_NAMES.forEach((className) => {
      const score = typeof confidences[className] === "number" ? confidences[className] : 0;
      const percent = Math.round(score * 1000) / 10; // one decimal place

      const row = document.createElement("li");
      row.className = "confidence-row" + (className === predictedClass ? " is-top" : "");

      const nameEl = document.createElement("span");
      nameEl.className = "class-name";
      nameEl.textContent = className;

      const trackEl = document.createElement("span");
      trackEl.className = "confidence-bar-track";
      const fillEl = document.createElement("span");
      fillEl.className = "confidence-bar-fill";
      fillEl.style.width = `${Math.max(0, Math.min(100, percent))}%`;
      trackEl.appendChild(fillEl);

      const valueEl = document.createElement("span");
      valueEl.className = "confidence-value";
      valueEl.textContent = `${percent.toFixed(1)}%`;

      row.appendChild(nameEl);
      row.appendChild(trackEl);
      row.appendChild(valueEl);
      confidenceList.appendChild(row);
    });

    hideLoading();
    resultsPanel.hidden = false;
    resultsContent.hidden = false;
  }

  // ---------- UI state helpers ----------

  function showLoading() {
    resultsPanel.hidden = false;
    resultsContent.hidden = true;
    loadingState.hidden = false;
  }

  function hideLoading() {
    loadingState.hidden = true;
  }

  function hideResults() {
    resultsPanel.hidden = true;
    resultsContent.hidden = true;
    loadingState.hidden = true;
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
  }

  function hideError() {
    errorMessage.hidden = true;
    errorMessage.textContent = "";
  }
})();
