(() => {
  // Step-through animation: elements use data-on="from-to" (visible range) and
  // data-hl="2,5" (highlighted steps); captions are <li> items in .stepper-captions.
  const inRange = (spec, step) => spec.split(",").some((part) => {
    const [a, b] = part.split("-").map((x) => (x === "" ? Infinity : Number(x)));
    return part.includes("-") ? step >= a && step <= (Number.isNaN(b) ? Infinity : b) : step === a;
  });
  document.querySelectorAll(".stepper").forEach((root) => {
    const captions = [...root.querySelectorAll(".stepper-captions > li")];
    const total = captions.length;
    let step = 1;
    let timer = null;
    const bar = document.createElement("div");
    bar.className = "stepper-controls";
    bar.innerHTML = '<button type="button" data-act="prev">上一步</button>' +
      '<button type="button" data-act="play">自动播放</button>' +
      '<button type="button" data-act="next">下一步</button>' +
      '<button type="button" data-act="reset">重来</button>' +
      '<span class="stepper-count" aria-live="polite"></span>';
    root.insertBefore(bar, root.querySelector(".stepper-stage"));
    const count = bar.querySelector(".stepper-count");
    const play = bar.querySelector('[data-act="play"]');
    function render() {
      root.querySelectorAll("[data-on]").forEach((el) => el.classList.toggle("st-hidden", !inRange(el.dataset.on, step)));
      root.querySelectorAll("[data-hl]").forEach((el) => el.classList.toggle("st-hl", inRange(el.dataset.hl, step)));
      captions.forEach((li, i) => li.classList.toggle("st-current", i === step - 1));
      count.textContent = `第 ${step} / ${total} 步`;
    }
    function stop() { clearInterval(timer); timer = null; play.textContent = "自动播放"; }
    bar.addEventListener("click", (e) => {
      const act = e.target.dataset.act;
      if (!act) return;
      if (act === "play") {
        if (timer) return stop();
        if (step >= total) step = 0;
        play.textContent = "暂停";
        timer = setInterval(() => { step += 1; render(); if (step >= total) stop(); }, 2200);
        step += 1;
      } else {
        stop();
        if (act === "prev") step = Math.max(1, step - 1);
        if (act === "next") step = Math.min(total, step + 1);
        if (act === "reset") step = 1;
      }
      render();
    });
    captions.forEach((li, i) => li.addEventListener("click", () => { stop(); step = i + 1; render(); }));
    render();
  });
})();
