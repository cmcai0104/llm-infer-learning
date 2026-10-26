# HTML 学习站：构建与使用

> 历史参考：以下构建流程已经停用。当前直接维护 HTML，打开 `site/index.html` 即可，最新说明见 `site/tools/index.html`。

[返回总目录](../README.md)

Markdown/RST 是唯一内容源，`build_site.py` 自动生成 HTML。阅读站提供主题导航、专家路线快捷入口、页内目录、全文搜索、静态 SVG、原有 Mermaid 图、数学公式和打印样式。

## 构建

建议使用 Python 3.12 虚拟环境，在资料库根目录执行：

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python -r tools/requirements-docs.txt
.venv-docs/bin/python tools/build_site.py
```

第一次构建会下载固定版本 Mermaid 10.9.3 并核对 SHA-256，随后保存在 `site/assets/`；再次构建可复用，不需要浏览时联网。网络受限时可提前准备同版本文件：

```bash
.venv-docs/bin/python tools/build_site.py --mermaid-bundle /path/to/mermaid.min.js
```

依赖许可证位于生成站点的 `assets/MERMAID-LICENSE.txt`。站点样式和搜索不依赖在线 CDN。数学公式使用浏览器 MathML；无法转换的 Markdown 公式保留可读 TeX 源文。

## 阅读

构建后打开 `site/index.html`。建议也可以启动本地服务：

```bash
.venv-docs/bin/python -m http.server 8765 --bind 127.0.0.1 --directory site
```

浏览 `http://127.0.0.1:8765`。按 `/` 打开搜索，输入“租约”“取消”“P7”等主题。窄屏下点击“目录”展开侧栏。原有 Mermaid 图提供图源折叠区；新增设计图是直接可见的 SVG。

`site/` 是生成目录，已忽略，不手工编辑其中正文。修改教材、图源或 `tools/web/` 后重新构建即可；源文和生成页面的相对链接会自动转换。

## 图与维护

新增设计图的 DOT→SVG 命令见[设计图册](../01-foundations/design-atlas.md)。维护时检查源文链接、图的完成条件、版本依据，再构建 HTML；浏览长表、代码、公式与宽图，避免只有构建成功而无法正常阅读。
