// Vanilla-JS frontend for the MRI tumor classifier.
// No build step, no framework: posts the selected image to the FastAPI
// backend's /predict endpoint and renders the response.

(() => {
  "use strict";

  // Order must match app/config.py's CLASS_NAMES.
  const CLASSES = [
    {
      id: "glioma",
      label: "Glioma",
      desc: "A tumor that arises from glial cells, the supporting cells of the brain and spinal cord.",
    },
    {
      id: "meningioma",
      label: "Meningioma",
      desc: "A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.",
    },
    {
      id: "notumor",
      label: "No tumor",
      desc: "The model found no tumor pattern in this scan.",
    },
    {
      id: "pituitary",
      label: "Pituitary",
      desc: "A tumor that arises in the pituitary gland at the base of the brain.",
    },
  ];
  const CLASS_BY_ID = Object.fromEntries(CLASSES.map((c) => [c.id, c]));
  const LOW_CONFIDENCE = 0.6;
  const IDLE_DESC = "Upload a scan or pick a sample. Analysis starts automatically.";

  const $ = (id) => document.getElementById(id);
  const workbench = $("workbench");
  const film = $("film");
  const filmName = $("film-name");
  const filmStatus = $("film-status");
  const filmControls = $("film-controls");
  const replaceBtn = $("replace-btn");
  const fileInput = $("file-input");
  const drop = $("drop");
  const stage = document.querySelector(".stage");
  const originalImage = $("original-image");
  const gradcamImage = $("gradcam-image");
  const stageView = $("stage-view");
  const stageSize = $("stage-size");
  const viewButtons = document.querySelectorAll(".segmented button");
  const errorBox = $("error-box");
  const errorMessage = $("error-message");
  const retryBtn = $("retry-btn");
  const sampleButtons = document.querySelectorAll(".sample-btn");
  const reportMeta = $("report-meta");
  const verdict = $("verdict");
  const verdictScore = $("verdict-score");
  const verdictDesc = $("verdict-desc");
  const lowNote = $("low-note");
  const sampleNote = $("sample-note");
  const bars = $("bars");

  let selectedFile = null;
  let expectedLabel = null; // set when the scan came from the sample strip
  let previewUrl = null;
  let requestId = 0;
  let controller = null;

  const STATUS = { empty: "Waiting", analyzing: "Scanning", result: "Done", error: "Failed" };
  const PASS_MS = 3200; // one top-to-bottom sweep of the scan line
  const PASS_GAP_MS = 260; // pause before the line starts again from the top
  const REVEAL_WINDOW = 0.25; // a result that lands this early in a pass is revealed on that pass
  const CATCH_UP_RATE = 4; // speed-up for the rest of a pass once a late result is waiting
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let scan = null;

  let modelReady = false;

  function setState(state) {
    workbench.dataset.state = state;
    filmStatus.textContent = state === "empty" && !modelReady ? "Waking model" : STATUS[state];
  }

  // After a cold start the server answers before its model has loaded.
  // /ready returns once it has, and keeping this request open is what lets
  // the server finish warming up while the page is already usable.
  fetch("/ready")
    .catch(() => null)
    .then(() => {
      modelReady = true;
      if (workbench.dataset.state === "empty") {
        setState("empty");
      }
    });

  // ---------- Confidence bars ----------

  const barRows = {};
  CLASSES.forEach((cls) => {
    const row = document.createElement("li");
    row.className = "bar-row";

    const name = document.createElement("span");
    name.textContent = cls.label;

    const value = document.createElement("span");
    value.className = "bar-value";

    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    track.appendChild(fill);

    row.append(name, value, track);
    bars.appendChild(row);
    barRows[cls.id] = { row, value, fill };
  });

  function resetBars() {
    CLASSES.forEach((cls) => {
      const { row, value, fill } = barRows[cls.id];
      row.classList.remove("is-top");
      value.textContent = "";
      fill.style.width = "0%";
    });
  }

  // ---------- Selecting a scan ----------

  drop.addEventListener("click", () => fileInput.click());
  replaceBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      selectFile(fileInput.files[0]);
    }
    fileInput.value = "";
  });

  // Stop the browser from navigating away when a file misses the viewer.
  ["dragover", "drop"].forEach((type) => {
    window.addEventListener(type, (event) => event.preventDefault());
  });

  film.addEventListener("dragover", () => film.classList.add("dragover"));
  film.addEventListener("dragleave", (event) => {
    if (!film.contains(event.relatedTarget)) {
      film.classList.remove("dragover");
    }
  });
  film.addEventListener("drop", (event) => {
    film.classList.remove("dragover");
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length > 0) {
      selectFile(files[0]);
    }
  });

  document.addEventListener("paste", (event) => {
    const items = (event.clipboardData && event.clipboardData.files) || [];
    const image = Array.from(items).find((f) => f.type.startsWith("image/"));
    if (image) {
      selectFile(image);
    }
  });

  sampleButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      hideError();
      const id = button.dataset.sample;
      try {
        const response = await fetch(button.querySelector("img").src);
        if (!response.ok) {
          throw new Error();
        }
        const blob = await response.blob();
        const file = new File([blob], button.dataset.filename || `${id}.jpg`, {
          type: blob.type || "image/jpeg",
        });
        selectFile(file, id);
        button.classList.add("is-selected");
      } catch (err) {
        showError("Could not load the sample scan. Please try again.", false);
      }
    });
  });

  function selectFile(file, sampleLabel = null) {
    // Fast client-side check. The backend validates again and is the
    // source of truth.
    if (!file.type || !file.type.startsWith("image/")) {
      showError(`"${file.name}" is not an image. Choose a JPEG or PNG scan.`, false);
      return;
    }

    selectedFile = file;
    expectedLabel = sampleLabel;
    sampleButtons.forEach((btn) => btn.classList.remove("is-selected"));

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    originalImage.src = previewUrl;
    originalImage.onload = () => {
      originalImage.onload = null;
      stageSize.textContent = `${originalImage.naturalWidth} × ${originalImage.naturalHeight} px`;
      stageSize.hidden = false;
    };
    showContentCrop(file);

    filmName.textContent = file.name || "Pasted image";
    drop.hidden = true;
    replaceBtn.hidden = false;

    analyze();
  }

  // The backend crops the scan to its non-black content before drawing the
  // heatmap (app.preprocessing.content_bounding_box_crop). Showing the same
  // crop here keeps the scan and the heatmap pixel-aligned, so the reveal
  // and the Heatmap / Original toggle line up. The upload itself is sent
  // unmodified.
  async function showContentCrop(file) {
    try {
      const bitmap = await createImageBitmap(file);
      const { width, height } = bitmap;
      if (width * height > 16e6) {
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0);
      const { data } = ctx.getImageData(0, 0, width, height);

      let top = height, left = width, bottom = -1, right = -1;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4;
          if (Math.max(data[i], data[i + 1], data[i + 2]) > 10) {
            if (x < left) left = x;
            if (x > right) right = x;
            if (y < top) top = y;
            bottom = y;
          }
        }
      }
      const cropW = right - left + 1;
      const cropH = bottom - top + 1;
      if (bottom < 0 || (cropW === width && cropH === height)) {
        return;
      }

      const out = document.createElement("canvas");
      out.width = cropW;
      out.height = cropH;
      out.getContext("2d").drawImage(canvas, left, top, cropW, cropH, 0, 0, cropW, cropH);
      const blob = await new Promise((resolve) => out.toBlob(resolve, "image/png"));
      if (!blob || selectedFile !== file) {
        return;
      }
      URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(blob);
      originalImage.src = previewUrl;
    } catch (err) {
      // Keep the uncropped preview.
    }
  }

  // ---------- Scan line ----------

  // The line sweeps top to bottom for as long as the request is in flight.
  // Once the heatmap has arrived, the next sweep reveals it: everything the
  // line has passed switches from the scan to the Grad-CAM overlay.
  function startScan() {
    stopScan();
    if (reducedMotion.matches) {
      return;
    }
    const s = { progress: 0, last: performance.now(), result: null, revealing: false };
    const gap = PASS_GAP_MS / PASS_MS;
    scan = s;
    stage.classList.add("is-scanning");

    const tick = (now) => {
      if (scan !== s) {
        return;
      }
      // A result that arrives too late in a pass waits for the next one, so
      // the rest of the current pass is hurried along.
      const rate = s.result && !s.revealing ? CATCH_UP_RATE : 1;
      s.progress += ((now - s.last) / PASS_MS) * rate;
      s.last = now;

      if (s.revealing && s.progress >= 1) {
        const { body, elapsedMs } = s.result;
        stopScan();
        renderResult(body, elapsedMs);
        return;
      }
      if (s.progress >= 1 + gap) {
        s.progress = 0;
      }
      if (!s.revealing && s.result && s.progress <= REVEAL_WINDOW) {
        s.revealing = true;
        stage.classList.add("is-revealing");
        gradcamImage.hidden = false;
        filmStatus.textContent = "Mapping";
      }

      stage.style.setProperty("--scan", Math.min(1, s.progress).toFixed(4));
      stage.classList.toggle("is-between-passes", s.progress > 1);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function stopScan() {
    scan = null;
    stage.classList.remove("is-scanning", "is-revealing", "is-between-passes");
    stage.style.removeProperty("--scan");
  }

  async function deliverResult(body, elapsedMs) {
    const s = scan;
    gradcamImage.src = `data:image/png;base64,${body.gradcam_overlay}`;
    try {
      await gradcamImage.decode();
    } catch (err) {
      // Render anyway; the browser shows the image once it can.
    }
    if (s && scan === s) {
      s.result = { body, elapsedMs };
    } else if (!scan) {
      renderResult(body, elapsedMs);
    }
  }

  // ---------- Analyze ----------

  retryBtn.addEventListener("click", analyze);

  async function analyze() {
    if (!selectedFile) {
      return;
    }

    // A newer selection supersedes any request still in flight.
    if (controller) {
      controller.abort();
    }
    controller = new AbortController();
    const thisRequest = ++requestId;

    hideError();
    resetReport();
    setView("original");
    filmControls.hidden = true;
    stageView.hidden = true;
    setState("analyzing");
    verdict.textContent = "Scanning";
    verdictDesc.textContent = modelReady
      ? "Running inference and computing the heatmap."
      : "The server is waking up, so this first scan takes a little longer.";
    startScan();

    const formData = new FormData();
    formData.append("file", selectedFile);
    const startedAt = performance.now();

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      let body;
      try {
        body = await response.json();
      } catch (parseError) {
        throw new Error("The server returned an unexpected response. Please try again.");
      }

      if (!response.ok) {
        const detail = body && typeof body.detail === "string" ? body.detail : null;
        throw new Error(detail || `Request failed with status ${response.status}.`);
      }

      if (thisRequest === requestId) {
        await deliverResult(body, performance.now() - startedAt);
      }
    } catch (err) {
      if (err.name === "AbortError" || thisRequest !== requestId) {
        return;
      }
      stopScan();
      resetReport();
      verdict.textContent = "No result";
      verdictDesc.textContent = "The scan could not be analyzed.";
      setState("error");
      const offline = err instanceof TypeError;
      showError(
        offline
          ? "Could not reach the server. Check your connection and retry."
          : err.message || "Something went wrong analyzing this image.",
        true
      );
    }
  }

  function renderResult(body, elapsedMs) {
    const confidences = body.confidences || {};
    const score = (id) => (typeof confidences[id] === "number" ? confidences[id] : 0);
    const top = CLASS_BY_ID[body.predicted_class] || {
      id: body.predicted_class,
      label: body.predicted_class,
      desc: "",
    };
    const topScore = score(top.id);

    filmControls.hidden = false;
    setView("heatmap");

    verdict.classList.remove("is-idle");
    verdict.textContent = top.label;
    verdictScore.textContent = `${formatPercent(topScore)} confidence`;
    verdictScore.hidden = false;
    verdictDesc.textContent = top.desc;
    reportMeta.textContent = `${(elapsedMs / 1000).toFixed(1)} s`;

    if (topScore < LOW_CONFIDENCE) {
      const runnerUp = CLASSES.filter((c) => c.id !== top.id).sort(
        (a, b) => score(b.id) - score(a.id)
      )[0];
      lowNote.textContent = `Low confidence. ${runnerUp.label} scored ${formatPercent(
        score(runnerUp.id)
      )}, so treat this as uncertain.`;
      lowNote.hidden = false;
    }

    if (expectedLabel && CLASS_BY_ID[expectedLabel]) {
      const expected = CLASS_BY_ID[expectedLabel].label;
      sampleNote.textContent =
        expectedLabel === top.id
          ? `Matches the sample's dataset label (${expected}).`
          : `Does not match the sample's dataset label (${expected}).`;
      sampleNote.hidden = false;
    }

    CLASSES.forEach((cls) => {
      const { row, value, fill } = barRows[cls.id];
      const s = score(cls.id);
      row.classList.toggle("is-top", cls.id === top.id);
      value.textContent = formatPercent(s);
      fill.style.width = `${Math.max(0, Math.min(100, s * 100))}%`;
    });

    setState("result");
  }

  function formatPercent(fraction) {
    return `${(Math.round(fraction * 1000) / 10).toFixed(1)}%`;
  }

  // ---------- Viewer mode ----------

  viewButtons.forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });

  function setView(view) {
    const heatmap = view === "heatmap";
    gradcamImage.hidden = !heatmap;
    originalImage.hidden = heatmap || !selectedFile;
    viewButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.view === view));
    });
    stageView.textContent = heatmap ? "Grad-CAM" : "Original";
    stageView.hidden = filmControls.hidden;
  }

  // ---------- Report and error helpers ----------

  function resetReport() {
    verdict.classList.add("is-idle");
    verdict.textContent = "No scan yet";
    verdictScore.hidden = true;
    verdictDesc.textContent = IDLE_DESC;
    reportMeta.textContent = "";
    lowNote.hidden = true;
    sampleNote.hidden = true;
    resetBars();
  }

  function showError(message, canRetry) {
    errorMessage.textContent = message;
    retryBtn.hidden = !canRetry;
    errorBox.hidden = false;
  }

  function hideError() {
    errorBox.hidden = true;
    errorMessage.textContent = "";
  }

  resetBars();
  setState("empty");
})();
