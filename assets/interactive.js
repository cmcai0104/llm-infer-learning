(() => {
  const GiB = 1024 ** 3;
  const fmtBytes = (n) => {
    const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return `${n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2)} ${units[i]}`;
  };
  const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(ms < 10 ? 2 : 1)} ms`);
  const num = (n) => n.toLocaleString("zh-CN");

  function bind(id, compute) {
    const root = document.getElementById(id);
    if (!root) return;
    const out = root.querySelector(".lab-output");
    const read = () => {
      const v = {};
      root.querySelectorAll("[data-k]").forEach((el) => { v[el.dataset.k] = Number(el.value); });
      root.querySelectorAll("[data-show]").forEach((el) => { el.textContent = ` ${v[el.dataset.show]}`; });
      return v;
    };
    const update = () => { out.innerHTML = compute(read(), root); };
    root.addEventListener("input", update);
    root.addEventListener("change", update);
    root._update = update;
    update();
  }

  const row = (label, value, note = "") =>
    `<div class="lab-row"><span>${label}</span><strong>${value}</strong>${note ? `<em>${note}</em>` : ""}</div>`;
  const verdict = (cls, text) => `<p class="lab-verdict ${cls}">${text}</p>`;

  bind("lab-capacity", (v) => {
    const mla = v.kind === 1;
    const perToken = mla ? v.layers * v.headDim * v.dtype : 2 * v.layers * v.kvHeads * v.headDim * v.dtype;
    const total = perToken * v.tokens;
    const chunks = Math.ceil(v.tokens / v.chunk);
    const tail = v.tokens % v.chunk;
    const evenSplit = !mla && v.kvHeads >= v.tp && v.kvHeads % v.tp === 0;
    let html = row("每 token KV", fmtBytes(perToken), "整个模型、所有层");
    html += row("整个前缀 KV", fmtBytes(total), `${num(v.tokens)} token`);
    html += row("chunk 数", num(chunks), tail ? `最后 ${tail} 个 token 不满一个 chunk，按 chunk 对齐的方案通常无法复用这段尾巴` : "正好对齐");
    html += row("每个逻辑 chunk", fmtBytes(perToken * v.chunk));
    if (evenSplit) {
      html += row("每个 TP rank 持有", fmtBytes(total / v.tp), "前提：KV head 均匀切分、无复制");
      html += row("存储对象数（chunk × rank）", num(chunks * v.tp), `单对象约 ${fmtBytes((perToken * v.chunk) / v.tp)}`);
      html += verdict("ok", "对象数越多，元数据和 RPC 压力越大；单对象越小，传输效率越容易受固定开销影响。");
    } else if (mla) {
      html += row("每个 TP rank 持有", fmtBytes(total), "MLA 的 latent 被所有 head 共用，常见实现每个 rank 各存一份");
      html += verdict("warn", "MLA 下增加 TP 不会降低每个 rank 的 KV；各 rank 对象内容可能相同，能否只存一份取决于缓存系统设计。");
    } else {
      html += verdict("warn", `KV head（${v.kvHeads}）少于 TP（${v.tp}）或不能整除：常见实现会复制 KV head，每个 rank 的容量不能简单除以 TP，要看实际布局。`);
    }
    return html;
  });

  bind("lab-breakeven", (v) => {
    const bytes = v.kib * 1024 * v.tokens;
    const netGiBs = ((v.gbps * 1e9) / 8 / GiB) * (v.eff / 100);
    const netMs = (bytes / GiB / netGiBs) * 1000;
    const h2dMs = (bytes / GiB / v.h2d) * 1000;
    const serial = v.fixed + netMs + h2dMs;
    const pipeline = v.fixed + Math.max(netMs, h2dMs);
    const recompute = (v.tokens / v.prefill) * 1000;
    let html = row("要搬的数据", fmtBytes(bytes));
    html += row("网络有效带宽", `${netGiBs.toFixed(2)} GiB/s`, `${v.gbps} Gbit/s × ${v.eff}%`);
    html += row("网络耗时", fmtMs(netMs));
    html += row("H2D 耗时", fmtMs(h2dMs));
    html += row("加载：串行", fmtMs(serial), "固定开销 + 网络 + H2D");
    html += row("加载：理想流水", fmtMs(pipeline), "乐观下界，真实系统介于两者之间");
    html += row("重算 Prefill", fmtMs(recompute), "按吞吐线性估算，长序列时注意力项会更贵");
    if (serial < recompute) html += verdict("ok", `加载更快：串行也能省约 ${fmtMs(recompute - serial)}。`);
    else if (pipeline < recompute) html += verdict("warn", "只有网络和 H2D 充分重叠时才划算；串行实现会比重算慢。");
    else html += verdict("bad", `重算更快：即使理想流水也慢约 ${fmtMs(pipeline - recompute)}，这种情况应跳过远端加载。`);
    return html;
  });

  (() => {
    const root = document.getElementById("lab-fold");
    if (!root) return;
    const grid = root.querySelector(".fold-grid");
    let state = [];
    const shape = () => ({
      ranks: Math.max(1, Math.min(8, Number(root.querySelector('[data-k="ranks"]').value))),
      chunks: Math.max(1, Math.min(12, Number(root.querySelector('[data-k="chunks"]').value))),
    });
    const resize = () => {
      const { ranks, chunks } = shape();
      state = Array.from({ length: ranks }, (_, r) => Array.from({ length: chunks }, (_, c) => (state[r] && state[r][c] !== undefined ? state[r][c] : true)));
    };
    const draw = () => {
      const { ranks, chunks } = shape();
      grid.style.gridTemplateColumns = `5em repeat(${chunks}, minmax(2.6em, 1fr))`;
      let cells = '<span class="fold-head"></span>';
      for (let c = 0; c < chunks; c++) cells += `<span class="fold-head">chunk ${c}</span>`;
      for (let r = 0; r < ranks; r++) {
        cells += `<span class="fold-head">rank ${r}</span>`;
        for (let c = 0; c < chunks; c++) {
          cells += `<button type="button" class="fold-cell ${state[r][c] ? "hit" : "miss"}" data-r="${r}" data-c="${c}" aria-pressed="${state[r][c]}" aria-label="rank ${r} chunk ${c}">${state[r][c] ? "有" : "缺"}</button>`;
        }
      }
      grid.innerHTML = cells;
    };
    bind("lab-fold", (v) => {
      resize();
      draw();
      const { ranks, chunks } = shape();
      let filled = 0;
      state.forEach((r) => r.forEach((x) => { filled += x ? 1 : 0; }));
      let usable = 0;
      while (usable < chunks && state.every((r) => r[usable])) usable++;
      grid.querySelectorAll(".fold-cell").forEach((el) => {
        if (Number(el.dataset.c) < usable) el.classList.add("usable");
      });
      const wasted = filled - usable * ranks;
      let html = row("对象命中率", `${((filled / (ranks * chunks)) * 100).toFixed(0)}%`, `${filled} / ${ranks * chunks} 个对象存在`);
      html += row("可跳过的 chunk", `${usable} / ${chunks}`, "所有 rank 都完整的最长连续前缀");
      html += row("可跳过的 token", num(usable * v.chunkTokens), `占前缀 ${((usable / chunks) * 100).toFixed(0)}%`);
      html += row("搬了也用不上的对象", num(wasted), "缺口之后的对象：加载它们只浪费带宽");
      html += usable === chunks ? verdict("ok", "整段前缀可用。") : verdict("warn", `chunk ${usable} 有缺口，之后的 chunk 即使存在也不能用于普通前缀复用。滑窗或混合模型的规则不同。`);
      return html;
    });
    grid.addEventListener("click", (e) => {
      const cell = e.target.closest(".fold-cell");
      if (!cell) return;
      state[cell.dataset.r][cell.dataset.c] = !state[cell.dataset.r][cell.dataset.c];
      root._update();
    });
    root.addEventListener("click", (e) => {
      const action = e.target.dataset && e.target.dataset.action;
      if (!action) return;
      state = state.map((r) => r.map(() => (action === "fill" ? true : Math.random() > 0.15)));
      root._update();
    });
  })();

  bind("lab-lease", (v, root) => {
    const done = v.queue + v.transfer;
    const span = Math.max(v.lease, v.timeout, done) * 1.05;
    const pct = (t) => `${((t / span) * 100).toFixed(2)}%`;
    root.querySelector(".timeline").innerHTML =
      `<div class="tl-bar tl-queue" style="left:0;width:${pct(v.queue)}"></div>` +
      `<div class="tl-bar tl-transfer" style="left:${pct(v.queue)};width:${pct(v.transfer)}"></div>` +
      `<div class="tl-mark tl-lease" style="left:${pct(v.lease)}"><span>租约到期</span></div>` +
      `<div class="tl-mark tl-timeout" style="left:${pct(v.timeout)}"><span>调用方超时</span></div>`;
    let html = row("传输完成时刻", fmtMs(done), "排队 + 传输");
    html += row("租约到期时刻", fmtMs(v.lease));
    html += row("调用方超时时刻", fmtMs(v.timeout));
    if (v.timeout < done) {
      html += verdict("bad", `调用方先超时：本次请求改走重算。但底层任务还会继续约 ${fmtMs(done - v.timeout)}，目标 buffer 必须隔离到完成或取消确认之后才能复用，迟到回调按 generation 丢弃。`);
    } else if (done > v.lease) {
      html += verdict("bad", `数据搬完了，但租约已在 ${fmtMs(done - v.lease)} 前过期：空间可能已被回收或复用，必须拒绝这次数据，换副本、有界重试或重算。`);
    } else {
      const margin = v.lease - done;
      html += margin < v.lease * 0.2
        ? verdict("warn", `成功，但租约余量只有 ${fmtMs(margin)}：排队稍有抖动就会过期，应缩短排队或评估租约时长。`)
        : verdict("ok", `成功：完成后还需校验完整性与布局，再交给 GPU 使用。租约余量 ${fmtMs(margin)}。`);
    }
    return html;
  });

  bind("lab-plan", (v) => {
    const logical = v.prefixes * v.gib;
    const payload = logical * v.replicas;
    const perHost = v.hostGib * (v.usable / 100) * (1 - v.reserve / 100);
    const available = v.hosts * perHost;
    const afterFailure = Math.max(0, v.hosts - v.domain) * perHost;
    const needed = Math.ceil(payload / perHost);
    const tib = (g) => `${(g / 1024).toFixed(1)} TiB`;
    let html = row("逻辑 KV", tib(logical), "唯一前缀数 × 每前缀大小");
    html += row("副本后 payload", tib(payload), `× ${v.replicas} 副本`);
    html += row("每台可用于缓存", `${perHost.toFixed(0)} GiB`, "扣除可用比例和预留");
    html += row("当前总可用", tib(available), `${num(v.hosts)} 台`);
    html += row("最少需要机器", num(needed), "仅按容量，不含带宽与热点");
    html += row("去掉一个故障域后", tib(afterFailure), `${num(v.domain)} 台退出`);
    if (available < payload) html += verdict("bad", `容量不够：还差约 ${tib(payload - available)}，至少需要 ${num(needed)} 台。`);
    else if (afterFailure < payload) html += verdict("warn", "正常时够用，但一个故障域退出后放不下全部副本：要么接受淘汰和更多重算，要么增加机器。");
    else html += verdict("ok", "正常和单故障域退出时容量都够。接着检查读写带宽、热点和修复流量。");
    return html;
  });
})();
