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

initTheme();
bootstrap();
