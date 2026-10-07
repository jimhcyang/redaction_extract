(function () {
  "use strict";

  const data = window.RESULT_BROWSER_DATA;
  if (!data || !Array.isArray(data.sections)) {
    document.body.textContent = "Results data did not load.";
    return;
  }

  const els = {
    tabs: document.getElementById("section-tabs"),
    summary: document.getElementById("summary-strip"),
    search: document.getElementById("search"),
    category: document.getElementById("category"),
    issuesOnly: document.getElementById("issues-only"),
    count: document.getElementById("result-count"),
    description: document.getElementById("section-description"),
    list: document.getElementById("case-list"),
    detail: document.getElementById("detail")
  };

  const state = { section: data.sections[0], selectedId: null };

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function slug(value) {
    return String(value).toLowerCase().replaceAll("_", "-").replaceAll(" ", "-");
  }

  function percent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : esc(value);
  }

  function displayMetric(key, value) {
    if (["coverage", "page union IoU", "pixel precision", "pixel recall", "OCR token F1"].includes(key)) {
      return percent(value);
    }
    return esc(value);
  }

  function filteredCases() {
    const query = els.search.value.trim().toLowerCase();
    const category = els.category.value;
    return state.section.cases.filter((item) => {
      if (els.issuesOnly.checked && !item.has_issue) return false;
      if (category && category !== "all" && !(item.tags || [item.category]).includes(category)) return false;
      return !query || item.search_text.includes(query);
    });
  }

  function renderTabs() {
    els.tabs.innerHTML = data.sections.map((section) =>
      `<button type="button" data-id="${esc(section.id)}" class="${section.id === state.section.id ? "active" : ""}">${esc(section.label)}</button>`
    ).join("");
    els.tabs.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        state.section = data.sections.find((section) => section.id === button.dataset.id);
        state.selectedId = null;
        els.search.value = "";
        els.issuesOnly.checked = true;
        render();
      });
    });
  }

  function renderSummary() {
    els.summary.innerHTML = Object.entries(state.section.summary).map(([label, value]) =>
      `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`
    ).join("");
  }

  function renderCategoryOptions() {
    const previous = els.category.value;
    els.category.innerHTML = state.section.categories.map((category) =>
      `<option value="${esc(category)}">${esc(category)}</option>`
    ).join("");
    els.category.value = state.section.categories.includes(previous) ? previous : "all";
  }

  function renderList() {
    const cases = filteredCases();
    els.count.textContent = `${cases.length.toLocaleString()} shown of ${state.section.cases.length.toLocaleString()}`;
    els.description.textContent = state.section.description;
    if (!cases.some((item) => item.id === state.selectedId)) {
      state.selectedId = cases.length ? cases[0].id : null;
    }
    els.list.innerHTML = cases.map((item) => {
      const secondary = state.section.kind === "production"
        ? `${item.tier} | ${Math.round(Number(item.metrics.coverage) * 100)}% coverage`
        : `${item.release} | ${item.metrics["gold components"]} gold / ${item.metrics["predicted components"]} detected`;
      return `<button type="button" class="case-button ${item.id === state.selectedId ? "active" : ""}" data-id="${esc(item.id)}" role="option" aria-selected="${item.id === state.selectedId}">
        <strong>${esc(item.title)}</strong>
        <span class="case-sub"><span class="badge ${slug(item.category)}">${esc(item.category)}</span><span>${esc(secondary)}</span></span>
      </button>`;
    }).join("");
    els.list.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedId = button.dataset.id;
        renderList();
        renderDetail();
      });
    });
  }

  function metricGrid(metrics) {
    return `<div class="metric-grid">${Object.entries(metrics).map(([label, value]) =>
      `<div class="metric"><b>${displayMetric(label, value)}</b><span>${esc(label)}</span></div>`
    ).join("")}</div>`;
  }

  function imageGrid(images) {
    if (!images.length) return "";
    return `<div class="image-grid">${images.map((image) =>
      `<figure><a href="${esc(image.path)}" target="_blank"><img src="${esc(image.path)}" alt="${esc(image.label)}" loading="lazy"></a><figcaption>${esc(image.label)} | click for native resolution</figcaption></figure>`
    ).join("")}</div>`;
  }

  function linkRow(links) {
    const available = links.filter((link) => link.path);
    if (!available.length) return "";
    return `<div class="links">${available.map((link) =>
      `<a href="${esc(link.path)}" target="_blank">${esc(link.label)}</a>`
    ).join("")}</div>`;
  }

  function productionDetail(item) {
    const calloutClass = item.has_issue ? "" : " pass";
    return `<div class="detail-head"><div><p class="kicker">Production target | ${esc(item.tier)}</p><h2>${esc(item.id)}</h2><p class="detail-summary">${esc(item.page_pair_key)}</p></div><span class="badge ${slug(item.category)}">${esc(item.category)}</span></div>
      <div class="callout${calloutClass}"><strong>How to interpret this status</strong><br>${esc(item.interpretation)}</div>
      ${metricGrid(item.metrics)}
      <div class="text-pair">
        <section class="text-card"><h3>Expected Astra answer</h3><p>${esc(item.expected_answer)}</p></section>
        <section class="text-card"><h3>Text recovered from later PDF layer</h3><p>${esc(item.recovered_text || "No reliable text recovered.")}</p></section>
      </div>
      ${imageGrid(item.images)}
      ${linkRow(item.links)}`;
  }

  function componentTable(rows) {
    if (!rows.length) return "<p>No component rows were emitted.</p>";
    return `<div class="table-wrap"><table><thead><tr><th>Status</th><th>Gold</th><th>Prediction</th><th>IoU</th><th>Detector route</th></tr></thead><tbody>${rows.map((row) =>
      `<tr><td>${esc(row.match_status)}</td><td>${esc(row.gold_component_id || "-")}</td><td>${esc(row.prediction_component_id || "-")}</td><td>${row.iou ? Number(row.iou).toFixed(3) : "-"}</td><td>${esc(row.prediction_source || "-")}</td></tr>`
    ).join("")}</tbody></table></div>`;
  }

  function manualDetail(item) {
    const flags = item.flags.length
      ? `<ul>${item.flags.map((flag) => `<li>${esc(flag)}</li>`).join("")}</ul>`
      : "<br>No strict component, grouping, or geometry review flag.";
    return `<div class="detail-head"><div><p class="kicker">${esc(state.section.label)} | ${esc(item.release)} release</p><h2>${esc(item.title)}</h2></div><span class="badge ${slug(item.category)}">${esc(item.category)}</span></div>
      <div class="callout ${item.has_issue ? "" : "pass"}"><strong>Review classification</strong>${flags}</div>
      ${metricGrid(item.metrics)}
      ${imageGrid(item.images)}
      ${linkRow(item.links)}
      <h3>Component-level assignment</h3>
      ${componentTable(item.component_matches)}`;
  }

  function renderDetail() {
    const item = state.section.cases.find((candidate) => candidate.id === state.selectedId);
    if (!item) {
      els.detail.innerHTML = "<div class=\"empty\"><h2>No matching cases</h2><p>Clear the search or change the review filters.</p></div>";
      return;
    }
    els.detail.innerHTML = state.section.kind === "production"
      ? productionDetail(item)
      : manualDetail(item);
  }

  function render() {
    renderTabs();
    renderSummary();
    renderCategoryOptions();
    renderList();
    renderDetail();
  }

  [els.search, els.category, els.issuesOnly].forEach((control) => {
    control.addEventListener(control === els.search ? "input" : "change", () => {
      state.selectedId = null;
      renderList();
      renderDetail();
    });
  });

  render();
}());
