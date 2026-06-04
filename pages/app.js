const state = {
  plots: [],
  search: "",
  group: "All",
  sort: "title-asc",
};

const cardsEl = document.getElementById("cards");
const statusEl = document.getElementById("status");
const totalCountEl = document.getElementById("totalCount");
const visibleCountEl = document.getElementById("visibleCount");
const groupNavEl = document.getElementById("groupNav");
const searchInputEl = document.getElementById("searchInput");
const sortSelectEl = document.getElementById("sortSelect");
const themeToggleEl = document.getElementById("themeToggle");
const themeToggleTextEl = document.getElementById("themeToggleText");
const templateEl = document.getElementById("plotCardTemplate");
const apaCitationEl = document.getElementById("apaCitation");
const bibtexCitationEl = document.getElementById("bibtexCitation");
const copyApaEl = document.getElementById("copyApa");
const copyBibtexEl = document.getElementById("copyBibtex");

const SITE_TITLE = "DiffusionRWR Scientific Plot Atlas";
const REPO_URL = "https://github.com/InigoSerjeant/DiffusionRWR_model_repo";

function deriveSiteUrl() {
  const u = new URL(window.location.href);
  if (u.pathname.endsWith("index.html")) {
    u.pathname = u.pathname.slice(0, -"index.html".length);
  }
  u.search = "";
  u.hash = "";
  return u.toString();
}

function formatIsoDate(dateValue) {
  const d = new Date(dateValue);
  if (Number.isNaN(d.getTime())) {
    return "n.d.";
  }
  return d.toISOString().slice(0, 10);
}

function formatAccessDate() {
  return new Date().toISOString().slice(0, 10);
}

function buildCitations(manifest) {
  const publishedDate = formatIsoDate(manifest.generated_at);
  const accessDate = formatAccessDate();
  const siteUrl = deriveSiteUrl();
  const total = Number.isFinite(manifest.total_plots) ? manifest.total_plots : state.plots.length;

  const apa = [
    "Serjeant, I. (" + publishedDate + "). ",
    SITE_TITLE + " [Interactive scientific dashboard; " + total + " embedded plots]. ",
    "GitHub Pages. " + siteUrl,
  ].join("");

  const bibtex = [
    "@misc{serjeant_diffusionrwr_plot_atlas_" + publishedDate.slice(0, 4) + ",",
    "  author       = {Serjeant, Inigo},",
    "  title        = {" + SITE_TITLE + "},",
    "  year         = {" + publishedDate.slice(0, 4) + "},",
    "  howpublished = {GitHub Pages},",
    "  note         = {Interactive scientific dashboard; " + total + " embedded plots. Accessed: " + accessDate + "},",
    "  url          = {" + siteUrl + "},",
    "  repository   = {" + REPO_URL + "}",
    "}",
  ].join("\n");

  apaCitationEl.textContent = apa;
  bibtexCitationEl.textContent = bibtex;
}

async function copyText(text, buttonEl, label) {
  try {
    await navigator.clipboard.writeText(text);
    const old = buttonEl.textContent;
    buttonEl.textContent = label + " copied";
    setTimeout(() => {
      buttonEl.textContent = old;
    }, 1200);
  } catch (_) {
    buttonEl.textContent = "Clipboard blocked";
    setTimeout(() => {
      buttonEl.textContent = label;
    }, 1200);
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  themeToggleTextEl.textContent = theme === "dark" ? "Light mode" : "Dark mode";
}

function initTheme() {
  const stored = localStorage.getItem("theme");
  if (stored === "dark" || stored === "light") {
    applyTheme(stored);
    return;
  }
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(prefersDark ? "dark" : "light");
}

function sortPlots(plots, mode) {
  const sorted = [...plots];
  const key = mode.startsWith("path") ? "path" : "title";
  const direction = mode.endsWith("desc") ? -1 : 1;

  sorted.sort((a, b) =>
    a[key].localeCompare(b[key], undefined, { sensitivity: "base" }) * direction
  );

  return sorted;
}

function filterPlots() {
  const q = state.search.trim().toLowerCase();

  let filtered = state.plots;

  if (state.group !== "All") {
    filtered = filtered.filter((plot) => plot.group === state.group);
  }

  if (q) {
    filtered = filtered.filter((plot) => {
      const haystack = `${plot.title} ${plot.path} ${plot.group}`.toLowerCase();
      return haystack.includes(q);
    });
  }

  return sortPlots(filtered, state.sort);
}

function renderGroups(plots) {
  const groups = ["All", ...new Set(plots.map((p) => p.group))];
  groupNavEl.innerHTML = "";

  for (const group of groups) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "group-btn";
    if (group === state.group) {
      btn.classList.add("active");
    }

    const count = group === "All" ? plots.length : plots.filter((p) => p.group === group).length;
    btn.textContent = `${group} (${count})`;
    btn.addEventListener("click", () => {
      state.group = group;
      renderAll();
    });

    groupNavEl.appendChild(btn);
  }
}

function createCard(plot, idx) {
  const fragment = templateEl.content.cloneNode(true);
  const card = fragment.querySelector(".plot-card");
  const title = fragment.querySelector(".plot-title");
  const meta = fragment.querySelector(".plot-meta");
  const link = fragment.querySelector(".plot-open");
  const iframe = fragment.querySelector("iframe");

  card.style.animationDelay = `${Math.min(idx, 10) * 40}ms`;
  title.textContent = plot.title;
  meta.textContent = `${plot.group} | ${plot.path}`;
  link.href = plot.path;
  iframe.src = plot.path;
  iframe.title = plot.title;

  return fragment;
}

function renderCards(plots) {
  cardsEl.innerHTML = "";

  if (plots.length === 0) {
    statusEl.textContent = "No plots match the current filters.";
    return;
  }

  statusEl.textContent = "";
  const docFrag = document.createDocumentFragment();
  plots.forEach((plot, idx) => docFrag.appendChild(createCard(plot, idx)));
  cardsEl.appendChild(docFrag);
}

function renderAll() {
  renderGroups(state.plots);
  const filtered = filterPlots();
  visibleCountEl.textContent = String(filtered.length);
  totalCountEl.textContent = String(state.plots.length);
  renderCards(filtered);
}

async function bootstrap() {
  try {
    statusEl.textContent = "Loading plot manifest...";
    const response = await fetch("./plots-manifest.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const manifest = await response.json();
    state.plots = Array.isArray(manifest.plots) ? manifest.plots : [];
    buildCitations(manifest);

    if (state.plots.length === 0) {
      statusEl.textContent = "No HTML plots found in Paper_results.";
      totalCountEl.textContent = "0";
      visibleCountEl.textContent = "0";
      return;
    }

    renderAll();
  } catch (error) {
    console.error(error);
    statusEl.textContent = "Failed to load plot manifest. Check plots-manifest.json generation.";
  }
}

searchInputEl.addEventListener("input", (event) => {
  state.search = event.target.value || "";
  renderAll();
});

sortSelectEl.addEventListener("change", (event) => {
  state.sort = event.target.value;
  renderAll();
});

themeToggleEl.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

copyApaEl.addEventListener("click", () => {
  copyText(apaCitationEl.textContent, copyApaEl, "Copy APA");
});

copyBibtexEl.addEventListener("click", () => {
  copyText(bibtexCitationEl.textContent, copyBibtexEl, "Copy BibTeX");
});

initTheme();
bootstrap();
const statusEl = document.getElementById("status");
const cardsEl = document.getElementById("cards");
const groupNavEl = document.getElementById("groupNav");
const searchInputEl = document.getElementById("searchInput");
const sortSelectEl = document.getElementById("sortSelect");
const visibleCountEl = document.getElementById("visibleCount");
const totalCountEl = document.getElementById("totalCount");
const cardTemplate = document.getElementById("plotCardTemplate");
const themeToggleEl = document.getElementById("themeToggle");
const themeToggleTextEl = document.getElementById("themeToggleText");

let allPlots = [];
let activeGroup = "All";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("plotAtlasTheme", theme);
  themeToggleTextEl.textContent = theme === "dark" ? "Light mode" : "Dark mode";
}

function bootstrapTheme() {
  const stored = localStorage.getItem("plotAtlasTheme");
  if (stored === "light" || stored === "dark") {
    applyTheme(stored);
    return;
  }

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(prefersDark ? "dark" : "light");
}

function renderGroupNav(plots) {
  const groups = new Map();
  groups.set("All", plots.length);

  for (const plot of plots) {
    groups.set(plot.group, (groups.get(plot.group) || 0) + 1);
  }

  groupNavEl.innerHTML = "";

  for (const [groupName, count] of groups.entries()) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `group-btn ${groupName === activeGroup ? "active" : ""}`;
    btn.textContent = `${groupName} (${count})`;
    btn.addEventListener("click", () => {
      activeGroup = groupName;
      renderGroupNav(allPlots);
      renderCards();
    });
    groupNavEl.appendChild(btn);
  }
}

function currentSortComparator(mode) {
  switch (mode) {
    case "title-desc":
      return (a, b) => b.title.localeCompare(a.title);
    case "path-asc":
      return (a, b) => a.path.localeCompare(b.path);
    case "path-desc":
      return (a, b) => b.path.localeCompare(a.path);
    case "title-asc":
    default:
      return (a, b) => a.title.localeCompare(b.title);
  }
}

function filteredPlots() {
  const query = searchInputEl.value.trim().toLowerCase();
  const sortMode = sortSelectEl.value;

  return allPlots
    .filter((plot) => activeGroup === "All" || plot.group === activeGroup)
    .filter((plot) => {
      if (!query) {
        return true;
      }
      return (
        plot.title.toLowerCase().includes(query) ||
        plot.path.toLowerCase().includes(query) ||
        plot.group.toLowerCase().includes(query) ||
        plot.filename.toLowerCase().includes(query)
      );
    })
    .sort(currentSortComparator(sortMode));
}

function renderCards() {
  const plots = filteredPlots();
  visibleCountEl.textContent = String(plots.length);

  cardsEl.innerHTML = "";

  if (plots.length === 0) {
    statusEl.textContent = "No plots match the current filter.";
    return;
  }

  statusEl.textContent = `Rendering ${plots.length} embedded plots.`;

  const fragment = document.createDocumentFragment();
  plots.forEach((plot) => {
    const node = cardTemplate.content.cloneNode(true);

    const title = node.querySelector(".plot-title");
    const meta = node.querySelector(".plot-meta");
    const open = node.querySelector(".plot-open");
    const iframe = node.querySelector("iframe");

    title.textContent = plot.title;
    meta.textContent = `${plot.group}  |  ${plot.path}`;

    open.href = plot.path;
    iframe.src = plot.path;
    iframe.title = `${plot.title} iframe`;

    fragment.appendChild(node);
  });

  cardsEl.appendChild(fragment);
}

async function loadManifest() {
  try {
    const res = await fetch("./plots-manifest.json", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const manifest = await res.json();
    allPlots = Array.isArray(manifest.plots) ? manifest.plots : [];

    totalCountEl.textContent = String(allPlots.length);
    visibleCountEl.textContent = String(allPlots.length);

    if (allPlots.length === 0) {
      statusEl.textContent = "No HTML plots found in Paper_results.";
      return;
    }

    renderGroupNav(allPlots);
    renderCards();
  } catch (err) {
    statusEl.textContent = `Failed to load plot manifest: ${err}`;
  }
}

bootstrapTheme();

themeToggleEl.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

searchInputEl.addEventListener("input", renderCards);
sortSelectEl.addEventListener("change", renderCards);

loadManifest();
