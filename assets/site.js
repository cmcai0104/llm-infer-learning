(() => {
  const toggle = document.getElementById("nav-toggle");
  toggle.addEventListener("click", () => {
    const opened = document.body.classList.toggle("nav-open");
    toggle.setAttribute("aria-expanded", String(opened));
  });

  const dialog = document.getElementById("search-dialog");
  const input = document.getElementById("search-input");
  const results = document.getElementById("search-results");
  const status = document.getElementById("search-status");
  const root = document.body.dataset.root;
  const openSearch = () => {
    if (!dialog.open) dialog.showModal();
    input.focus();
  };
  document.getElementById("search-open").addEventListener("click", openSearch);
  document.getElementById("search-close").addEventListener("click", () => dialog.close());
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
      event.preventDefault();
      openSearch();
    }
  });

  function search() {
    const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    results.replaceChildren();
    if (!terms.length) {
      status.textContent = "输入主题、术语或问题";
      return;
    }
    const matches = (window.INFRA_SEARCH || []).map((entry) => {
      const title = entry.title.toLowerCase();
      const text = entry.text.toLowerCase();
      if (!terms.every((term) => title.includes(term) || text.includes(term))) return null;
      return { entry, score: terms.reduce((sum, term) => sum + (title.includes(term) ? 10 : 1), 0) };
    }).filter(Boolean).sort((a, b) => b.score - a.score);
    status.textContent = matches.length ? `找到 ${matches.length} 篇资料，显示前 ${Math.min(30, matches.length)} 篇` : "没有匹配结果，试试更短的术语。";
    for (const { entry } of matches.slice(0, 30)) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `${root}/${entry.url}`;
      a.textContent = entry.title;
      const excerpt = document.createElement("p");
      const position = entry.text.toLowerCase().indexOf(terms[0]);
      const start = Math.max(0, position - 55);
      excerpt.textContent = (start ? "…" : "") + entry.text.slice(start, start + 180) + "…";
      li.append(a, excerpt);
      results.append(li);
    }
  }
  input.addEventListener("input", search);
  const query = new URLSearchParams(window.location.search).get("q");
  if (query) {
    input.value = query;
    search();
    openSearch();
  }

  async function renderDiagrams() {
    const diagrams = document.querySelectorAll(".diagram");
    if (!diagrams.length) return;
    if (!window.mermaid) return;
    mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "neutral", fontFamily: "sans-serif" });
    let number = 0;
    for (const figure of diagrams) {
      const source = figure.querySelector(".mermaid-source").textContent;
      try {
        const { svg } = await mermaid.render(`infra-diagram-${++number}`, source);
        figure.querySelector(".mermaid-target").innerHTML = svg;
        figure.querySelector("details").open = false;
        figure.dataset.status = "rendered";
      } catch (error) {
        figure.dataset.status = "error";
        const message = document.createElement("p");
        message.className = "diagram-error";
        message.textContent = "此图暂未渲染，可阅读图源或返回设计图册查看主线静态图。";
        figure.prepend(message);
        console.error("Diagram rendering failed", error);
      }
    }
    document.body.dataset.diagramsReady = "true";
  }
  renderDiagrams();
})();
