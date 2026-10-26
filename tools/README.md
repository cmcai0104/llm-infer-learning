# HTML 资料维护

直接打开仓库根目录的 [index.html](../index.html)，或 [site/index.html](../site/index.html)。

`site/` 是正式维护内容，不是构建产物。无需 Python、Node、安装依赖或运行编译命令。可以直接让 AI 修改相应 HTML。

- 正文、标题和页内目录：编辑对应 `.html`。
- 样式：编辑 `site/assets/site.css`。
- 浏览器交互：编辑 `site/assets/site.js`。
- 主题搜索：在 `site/assets/search-index.js` 维护各页的标题、地址、摘要和关键词。
- 图片：直接编辑 SVG，或修改页面中 Mermaid 图源；后者由浏览器显示。
- 新增或移动页面：同时检查主题入口、相关链接与搜索目录。

详细说明见 [HTML 维护页面](../site/tools/index.html)。原 Markdown/RST 和 `_sources/` 是参考笔记；后续内容以 HTML 为准。
