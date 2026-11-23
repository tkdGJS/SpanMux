#!/usr/bin/env python3
"""Render the vendor_v8 backend study guide as a self-contained HTML page."""

from __future__ import annotations

import html
import re
from argparse import ArgumentParser
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "55_VENDOR_V8_BACKEND_CODE_STUDY_GUIDE.md"
OUTPUT = ROOT / "docs" / "55_VENDOR_V8_BACKEND_CODE_STUDY_GUIDE.html"


def slugify(value: str, used: set[str]) -> str:
    slug = re.sub(r"[^\w\s-]", "", value.lower(), flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-") or "section"
    candidate = slug
    suffix = 2
    while candidate in used:
        candidate = f"{slug}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _find_heading_anchor(
    headings: list[tuple[int, str, str]], keyword: str, fallback: str
) -> str:
    for _, title, anchor in headings:
        if keyword in title:
            return anchor
    return fallback


def render(source_path: Path = SOURCE, *, snapshot: str = "2026.07.31") -> str:
    source = source_path.read_text(encoding="utf-8")
    md = MarkdownIt("commonmark", {"html": False, "linkify": False})
    md.enable("table")
    tokens = md.parse(source)
    used: set[str] = set()
    headings: list[tuple[int, str, str]] = []

    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1]
        title = inline.content
        level = int(token.tag[1])
        anchor = slugify(title, used)
        token.attrSet("id", anchor)
        headings.append((level, title, anchor))

    document_title = headings[0][1] if headings else "vendor_v8 Backend Code Study Guide"
    article = md.renderer.render(tokens, md.options, {})
    toc_items = []
    for level, title, anchor in headings:
        if level == 1:
            continue
        toc_items.append(
            f'<li class="toc-level-{level}"><a href="#{anchor}">'
            f"{html.escape(title)}</a></li>"
        )

    return TEMPLATE.format(
        title=html.escape(document_title),
        toc="\n".join(toc_items),
        article=article,
        source_href=html.escape(source_path.name, quote=True),
        snapshot=html.escape(snapshot),
        section_count=sum(level == 2 for level, _, _ in headings),
        overview_anchor=_find_heading_anchor(
            headings, "문서 목적", headings[0][2] if headings else "top"
        ),
        optimization_anchor=_find_heading_anchor(
            headings, "최적화 이력", headings[0][2] if headings else "top"
        ),
        scenario_anchor=_find_heading_anchor(
            headings, "학습 시나리오", headings[-1][2] if headings else "top"
        ),
    )


TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="vendor_v8 scheduler, KV residency, Context Ring, P/D Future와 GPU data path를 따라가는 코드 학습 가이드">
  <title>{title}</title>
  <style>
    :root {{
      --primary: #0066cc;
      --primary-focus: #0071e3;
      --primary-dark: #2997ff;
      --canvas: #ffffff;
      --parchment: #f5f5f7;
      --pearl: #fafafc;
      --tile: #272729;
      --tile-2: #2a2a2c;
      --black: #000000;
      --ink: #1d1d1f;
      --muted: #6e6e73;
      --hairline: #e0e0e0;
      --soft-line: rgba(0, 0, 0, .08);
      --nav-height: 44px;
      --subnav-height: 52px;
      --content-width: 1080px;
      --sidebar-width: 264px;
      --sans: "SF Pro Text", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --display: "SF Pro Display", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; scroll-padding-top: 116px; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--canvas);
      font-family: var(--sans);
      font-size: 17px;
      line-height: 1.47;
      letter-spacing: -.022em;
      -webkit-font-smoothing: antialiased;
    }}
    a {{ color: var(--primary); text-decoration-thickness: 1px; text-underline-offset: 3px; }}
    a:hover {{ text-decoration: underline; }}
    button, input {{ font: inherit; }}
    button:focus-visible, input:focus-visible, a:focus-visible {{
      outline: 2px solid var(--primary-focus);
      outline-offset: 3px;
    }}

    .reading-progress {{
      position: fixed; inset: 0 auto auto 0; z-index: 100;
      width: 0; height: 3px; background: var(--primary-dark);
    }}
    .global-nav {{
      position: fixed; inset: 0 0 auto; z-index: 90;
      height: var(--nav-height); color: #fff; background: var(--black);
    }}
    .global-nav-inner, .subnav-inner {{
      width: min(100% - 40px, 1200px); height: 100%; margin: 0 auto;
      display: flex; align-items: center; justify-content: space-between;
    }}
    .brand {{ display: flex; align-items: center; gap: 10px; color: #fff; font-size: 12px; text-decoration: none; letter-spacing: -.01em; }}
    .brand:hover {{ text-decoration: none; }}
    .brand-mark {{ width: 18px; height: 18px; display: grid; place-items: center; }}
    .global-meta {{ color: #ccc; font-size: 12px; letter-spacing: -.01em; }}
    .subnav {{
      position: fixed; inset: var(--nav-height) 0 auto; z-index: 80;
      height: var(--subnav-height); background: rgba(245, 245, 247, .82);
      border-bottom: 1px solid var(--soft-line);
      backdrop-filter: saturate(180%) blur(20px);
      -webkit-backdrop-filter: saturate(180%) blur(20px);
    }}
    .subnav-title {{ font-family: var(--display); font-size: 21px; font-weight: 600; letter-spacing: .011em; }}
    .subnav-actions {{ display: flex; align-items: center; gap: 16px; }}
    .subnav-link {{ color: var(--ink); font-size: 14px; text-decoration: none; }}
    .source-button {{
      display: inline-flex; min-height: 30px; align-items: center; padding: 6px 14px;
      color: #fff; background: var(--primary); border-radius: 9999px;
      font-size: 14px; line-height: 1.2; text-decoration: none;
    }}
    .source-button:hover {{ color: #fff; text-decoration: none; }}
    .source-button:active, .icon-button:active, .copy-button:active {{ transform: scale(.95); }}
    .menu-button {{ display: none; }}

    .hero {{
      margin-top: calc(var(--nav-height) + var(--subnav-height));
      padding: 104px 24px 96px; text-align: center; background: var(--parchment);
    }}
    .eyebrow {{ margin: 0 0 16px; color: var(--primary); font-size: 14px; font-weight: 600; letter-spacing: .01em; }}
    .hero h1 {{
      max-width: 920px; margin: 0 auto; font-family: var(--display);
      font-size: clamp(40px, 6vw, 64px); font-weight: 600; line-height: 1.06; letter-spacing: -.035em;
    }}
    .hero-lead {{ max-width: 720px; margin: 24px auto 0; font-size: 24px; font-weight: 300; line-height: 1.5; letter-spacing: 0; }}
    .hero-meta {{
      display: flex; justify-content: center; flex-wrap: wrap; gap: 8px 22px;
      margin-top: 34px; color: var(--muted); font-size: 14px;
    }}
    .system-map {{ padding: 72px 24px; color: #fff; background: var(--tile); }}
    .system-map-inner {{ max-width: 1120px; margin: 0 auto; }}
    .system-map h2 {{ margin: 0 0 12px; color: #fff; font: 600 40px/1.1 var(--display); letter-spacing: -.025em; }}
    .system-map > div > p {{ margin: 0 0 36px; color: #ccc; }}
    .flow {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; align-items: stretch; }}
    .flow-card {{
      position: relative; min-height: 126px; padding: 20px; border: 1px solid rgba(255,255,255,.14);
      border-radius: 18px; background: var(--tile-2);
    }}
    .flow-card:not(:last-child)::after {{ content: "→"; position: absolute; right: -16px; top: 45px; z-index: 2; color: var(--primary-dark); font-size: 20px; }}
    .flow-index {{ color: var(--primary-dark); font-size: 12px; font-weight: 600; }}
    .flow-card strong {{ display: block; margin: 8px 0 5px; color: #fff; font-size: 17px; font-weight: 600; }}
    .flow-card span:last-child {{ color: #ccc; font-size: 14px; line-height: 1.43; }}

    .study-layout {{
      display: grid; grid-template-columns: var(--sidebar-width) minmax(0, var(--content-width));
      justify-content: center; gap: 64px; padding: 72px 32px 120px; background: var(--canvas);
    }}
    .sidebar {{ position: relative; min-width: 0; }}
    .sidebar-inner {{ position: sticky; top: 120px; max-height: calc(100vh - 144px); display: flex; flex-direction: column; }}
    .search-wrap {{ position: relative; margin-bottom: 18px; }}
    .search-icon {{ position: absolute; left: 14px; top: 50%; width: 16px; height: 16px; transform: translateY(-50%); color: var(--muted); pointer-events: none; }}
    #doc-search {{
      width: 100%; height: 44px; padding: 10px 38px; color: var(--ink); background: var(--canvas);
      border: 1px solid var(--soft-line); border-radius: 9999px; font-size: 14px;
    }}
    .search-key {{ position: absolute; right: 12px; top: 12px; color: var(--muted); font: 12px/20px var(--mono); }}
    .search-status {{ min-height: 20px; margin: -8px 2px 10px; color: var(--muted); font-size: 12px; }}
    .toc-label {{ margin: 0 0 8px 12px; color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
    .toc {{ margin: 0; padding: 0 10px 0 0; overflow: auto; list-style: none; scrollbar-width: thin; }}
    .toc li {{ margin: 0; }}
    .toc a {{
      display: block; padding: 5px 10px; border-left: 2px solid transparent;
      color: var(--muted); font-size: 13px; line-height: 1.35; text-decoration: none;
    }}
    .toc .toc-level-3 a {{ padding-left: 22px; font-size: 12px; }}
    .toc a.active {{ border-left-color: var(--primary); color: var(--ink); font-weight: 600; }}
    .toc a:hover {{ color: var(--ink); text-decoration: none; }}

    .article {{ min-width: 0; }}
    .article > h1 {{ display: none; }}
    .article h2, .article h3 {{ position: relative; font-family: var(--display); color: var(--ink); }}
    .article h2 {{ margin: 88px 0 24px; padding-top: 4px; font-size: 40px; line-height: 1.1; letter-spacing: -.025em; }}
    .article h2:first-of-type {{ margin-top: 0; }}
    .article h3 {{ margin: 52px 0 16px; font-size: 28px; line-height: 1.18; letter-spacing: -.018em; }}
    .article h4 {{ margin: 32px 0 12px; font-size: 17px; line-height: 1.24; font-weight: 600; }}
    .article h2:hover::before, .article h3:hover::before {{
      content: "#"; position: absolute; left: -24px; color: var(--primary); font-weight: 400;
    }}
    .article p {{ margin: 0 0 20px; }}
    .article ul, .article ol {{ margin: 0 0 24px; padding-left: 25px; }}
    .article li {{ margin: 6px 0; }}
    .article li > ul, .article li > ol {{ margin-top: 6px; margin-bottom: 8px; }}
    .article blockquote {{
      margin: 0 0 32px; padding: 20px 24px; border-left: 4px solid var(--primary);
      border-radius: 0 18px 18px 0; background: var(--parchment); color: #333;
    }}
    .article blockquote p:last-child {{ margin-bottom: 0; }}
    .article code {{
      padding: .14em .36em; border-radius: 5px; background: var(--parchment);
      font: .86em/1.45 var(--mono); letter-spacing: -.02em; overflow-wrap: anywhere;
    }}
    .article pre {{
      position: relative; margin: 24px 0 32px; padding: 24px; overflow: auto;
      border-radius: 18px; color: #f5f5f7; background: var(--tile);
      font: 14px/1.65 var(--mono); letter-spacing: -.015em; tab-size: 2;
    }}
    .article pre code {{ padding: 0; border-radius: 0; color: inherit; background: transparent; font: inherit; overflow-wrap: normal; }}
    .copy-button {{
      position: absolute; top: 10px; right: 10px; padding: 7px 11px; color: #fff;
      border: 1px solid rgba(255,255,255,.18); border-radius: 8px; background: rgba(0,0,0,.32);
      font-size: 12px; line-height: 1; cursor: pointer;
    }}
    .table-scroll {{ margin: 24px 0 32px; overflow-x: auto; border: 1px solid var(--hairline); border-radius: 18px; }}
    .article table {{ width: 100%; border-collapse: collapse; font-size: 14px; letter-spacing: -.012em; }}
    .article th, .article td {{ padding: 13px 16px; border-bottom: 1px solid var(--hairline); text-align: left; vertical-align: top; }}
    .article th {{ background: var(--parchment); font-weight: 600; }}
    .article tr:last-child td {{ border-bottom: 0; }}
    .article hr {{ margin: 64px 0; border: 0; border-top: 1px solid var(--hairline); }}
    mark {{ padding: 0 .08em; border-radius: 3px; background: #fff0a6; color: inherit; }}
    .section-hidden {{ display: none; }}

    .back-to-top {{
      position: fixed; right: 24px; bottom: 24px; z-index: 30; width: 44px; height: 44px;
      display: grid; place-items: center; border: 0; border-radius: 50%; color: var(--ink);
      background: rgba(210,210,215,.78); backdrop-filter: blur(12px); cursor: pointer;
      opacity: 0; pointer-events: none; transition: opacity .2s ease;
    }}
    .back-to-top.visible {{ opacity: 1; pointer-events: auto; }}
    .back-to-top:active {{ transform: scale(.95); }}
    .footer {{ padding: 56px 24px; background: var(--parchment); color: var(--muted); font-size: 12px; letter-spacing: -.01em; }}
    .footer-inner {{ max-width: 1080px; margin: 0 auto; display: flex; justify-content: space-between; gap: 24px; }}
    .footer a {{ color: var(--muted); }}

    @media (max-width: 1068px) {{
      .flow {{ grid-template-columns: repeat(3, 1fr); }}
      .flow-card:not(:last-child)::after {{ display: none; }}
      .study-layout {{ grid-template-columns: 240px minmax(0, 1fr); gap: 36px; }}
      .article h2 {{ font-size: 34px; }}
    }}
    @media (max-width: 833px) {{
      .global-meta, .subnav-link {{ display: none; }}
      .menu-button {{
        display: grid; place-items: center; width: 36px; height: 36px; padding: 0;
        border: 0; border-radius: 8px; color: var(--ink); background: transparent; cursor: pointer;
      }}
      .hero {{ padding: 80px 22px 72px; }}
      .hero h1 {{ font-size: 40px; }}
      .hero-lead {{ font-size: 21px; }}
      .flow {{ grid-template-columns: 1fr 1fr; }}
      .study-layout {{ display: block; padding: 56px 24px 96px; }}
      .sidebar {{
        position: fixed; inset: calc(var(--nav-height) + var(--subnav-height)) 0 0; z-index: 70;
        padding: 24px; background: rgba(245,245,247,.98); visibility: hidden; opacity: 0;
        transition: opacity .18s ease, visibility .18s ease;
      }}
      .sidebar.open {{ visibility: visible; opacity: 1; }}
      .sidebar-inner {{ position: static; max-width: 520px; max-height: 100%; margin: 0 auto; }}
      .toc {{ padding-right: 0; }}
      .article h2:hover::before, .article h3:hover::before {{ display: none; }}
    }}
    @media (max-width: 640px) {{
      .global-nav-inner, .subnav-inner {{ width: calc(100% - 28px); }}
      .subnav-title {{ font-size: 17px; }}
      .source-button {{ padding-inline: 12px; font-size: 12px; }}
      .hero {{ padding: 64px 20px; }}
      .hero h1 {{ font-size: 34px; }}
      .hero-lead {{ font-size: 18px; line-height: 1.47; }}
      .hero-meta {{ display: grid; gap: 5px; }}
      .system-map {{ padding: 52px 20px; }}
      .system-map h2 {{ font-size: 34px; }}
      .flow {{ grid-template-columns: 1fr; }}
      .flow-card {{ min-height: 0; }}
      .study-layout {{ padding: 48px 20px 80px; }}
      .article h2 {{ margin-top: 68px; font-size: 30px; }}
      .article h3 {{ margin-top: 42px; font-size: 24px; }}
      .article pre {{ margin-inline: -20px; border-radius: 0; padding: 20px; font-size: 12px; }}
      .table-scroll {{ margin-inline: -20px; border-left: 0; border-right: 0; border-radius: 0; }}
      .footer-inner {{ display: block; }}
    }}
    @media (max-width: 419px) {{ .hero h1 {{ font-size: 28px; }} }}
    @media (prefers-reduced-motion: reduce) {{
      html {{ scroll-behavior: auto; }}
      *, *::before, *::after {{ transition-duration: .01ms !important; animation-duration: .01ms !important; }}
    }}
    @media print {{
      .global-nav, .subnav, .sidebar, .reading-progress, .back-to-top, .system-map, .footer {{ display: none !important; }}
      .hero {{ margin: 0; padding: 24px 0 48px; background: #fff; }}
      .study-layout {{ display: block; padding: 0; }}
      .article h2 {{ break-after: avoid; margin-top: 42px; }}
      .article pre, .table-scroll {{ break-inside: avoid; }}
      .article pre {{ white-space: pre-wrap; color: #111; background: #f5f5f7; }}
      a {{ color: #111; text-decoration: none; }}
    }}
  </style>
</head>
<body>
  <div class="reading-progress" aria-hidden="true"></div>
  <nav class="global-nav" aria-label="전역 내비게이션">
    <div class="global-nav-inner">
      <a class="brand" href="#top">
        <span class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="M4 6.5h6l2 3h8v9H4z" stroke="currentColor" stroke-width="1.6"/><path d="M7 13h10M7 16h7" stroke="currentColor" stroke-width="1.6"/></svg>
        </span>
        zoneatten_gpu · Research Notes
      </a>
      <span class="global-meta">vendor_v8 / Code Study</span>
    </div>
  </nav>
  <nav class="subnav" aria-label="문서 내비게이션">
    <div class="subnav-inner">
      <span class="subnav-title">Backend Code Study Guide</span>
      <div class="subnav-actions">
        <a class="subnav-link" href="#{overview_anchor}">Overview</a>
        <a class="subnav-link" href="#{optimization_anchor}">Optimizations</a>
        <a class="subnav-link" href="#{scenario_anchor}">Scenarios</a>
        <a class="source-button" href="{source_href}">Markdown 원문</a>
        <button class="menu-button" type="button" aria-label="목차 열기" aria-expanded="false">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
        </button>
      </div>
    </div>
  </nav>

  <header class="hero" id="top">
    <p class="eyebrow">VENDOR V8 · CODE SNAPSHOT {snapshot}</p>
    <h1>정책에서 GPU data path까지,<br>한 흐름으로 읽는 backend guide.</h1>
    <p class="hero-lead">Scheduler authority, KV residency, Context Ring, P/D Future, GreenContext의 경계를 따라가며 실제 실행 경로를 이해합니다.</p>
    <div class="hero-meta">
      <span>{section_count}개 핵심 섹션</span><span>Lifecycle-first reading</span><span>Optimization history</span><span>Symbol-first navigation</span>
    </div>
  </header>

  <section class="system-map" aria-labelledby="system-map-title">
    <div class="system-map-inner">
      <h2 id="system-map-title">The authority chain.</h2>
      <p>정책의 제안과 물리 상태의 확정을 분리해서 읽는 것이 핵심입니다.</p>
      <div class="flow" role="list">
        <div class="flow-card" role="listitem"><span class="flow-index">01</span><strong>Policy proposal</strong><span>Prefill · Decode · Residency · SM split</span></div>
        <div class="flow-card" role="listitem"><span class="flow-index">02</span><strong>Scheduler</strong><span>Token admission · KV allocation · victim commit</span></div>
        <div class="flow-card" role="listitem"><span class="flow-index">03</span><strong>EngineCore</strong><span>Phase submission · Future lifecycle · recovery</span></div>
        <div class="flow-card" role="listitem"><span class="flow-index">04</span><strong>GPU data path</strong><span>Native · Ring · D2H/H2D · GreenContext</span></div>
        <div class="flow-card" role="listitem"><span class="flow-index">05</span><strong>Commit</strong><span>Progress · KV state · TTFT/TBT · completion</span></div>
      </div>
    </div>
  </section>

  <main class="study-layout">
    <aside class="sidebar" aria-label="문서 목차">
      <div class="sidebar-inner">
        <div class="search-wrap">
          <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></svg>
          <input id="doc-search" type="search" placeholder="문서 검색" autocomplete="off" aria-label="문서 검색">
          <span class="search-key" aria-hidden="true">/</span>
        </div>
        <div class="search-status" aria-live="polite"></div>
        <p class="toc-label">Contents</p>
        <ol class="toc">{toc}</ol>
      </div>
    </aside>
    <article class="article" id="article">{article}</article>
  </main>

  <button class="back-to-top" type="button" aria-label="맨 위로 이동">
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m6 14 6-6 6 6"/></svg>
  </button>
  <footer class="footer">
    <div class="footer-inner">
      <span>Generated from the repository Markdown source. Symbols are authoritative; line ranges are navigation hints.</span>
      <a href="{source_href}">View source document</a>
    </div>
  </footer>

  <script>
    (() => {{
      const article = document.querySelector('#article');
      const sidebar = document.querySelector('.sidebar');
      const menuButton = document.querySelector('.menu-button');
      const progress = document.querySelector('.reading-progress');
      const backToTop = document.querySelector('.back-to-top');
      const tocLinks = [...document.querySelectorAll('.toc a')];
      const search = document.querySelector('#doc-search');
      const searchStatus = document.querySelector('.search-status');

      article.querySelectorAll('table').forEach((table) => {{
        const wrapper = document.createElement('div');
        wrapper.className = 'table-scroll';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      }});

      article.querySelectorAll('pre').forEach((pre) => {{
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.type = 'button';
        button.textContent = '복사';
        button.setAttribute('aria-label', '코드 블록 복사');
        button.addEventListener('click', async () => {{
          await navigator.clipboard.writeText(pre.querySelector('code').textContent);
          button.textContent = '완료';
          window.setTimeout(() => button.textContent = '복사', 1200);
        }});
        pre.appendChild(button);
      }});

      const updateScroll = () => {{
        const max = document.documentElement.scrollHeight - innerHeight;
        progress.style.width = `${{max > 0 ? (scrollY / max) * 100 : 0}}%`;
        backToTop.classList.toggle('visible', scrollY > 700);
      }};
      addEventListener('scroll', updateScroll, {{ passive: true }});
      addEventListener('resize', updateScroll);
      updateScroll();

      backToTop.addEventListener('click', () => scrollTo({{ top: 0, behavior: 'smooth' }}));
      menuButton.addEventListener('click', () => {{
        const open = sidebar.classList.toggle('open');
        menuButton.setAttribute('aria-expanded', String(open));
        menuButton.setAttribute('aria-label', open ? '목차 닫기' : '목차 열기');
      }});
      tocLinks.forEach((link) => link.addEventListener('click', () => {{
        sidebar.classList.remove('open');
        menuButton.setAttribute('aria-expanded', 'false');
      }}));

      const observed = [...article.querySelectorAll('h2, h3')];
      const linkById = new Map(tocLinks.map((link) => [link.hash.slice(1), link]));
      const observer = new IntersectionObserver((entries) => {{
        const visible = entries.filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (!visible.length) return;
        tocLinks.forEach((link) => link.classList.remove('active'));
        const active = linkById.get(visible[0].target.id);
        if (active) {{ active.classList.add('active'); active.scrollIntoView({{ block: 'nearest' }}); }}
      }}, {{ rootMargin: '-108px 0px -72% 0px', threshold: 0 }});
      observed.forEach((heading) => observer.observe(heading));

      const sections = [];
      article.querySelectorAll('h2').forEach((heading) => {{
        const nodes = [heading];
        let node = heading.nextElementSibling;
        while (node && node.tagName !== 'H2') {{ nodes.push(node); node = node.nextElementSibling; }}
        sections.push({{ heading, nodes, text: nodes.map((item) => item.textContent).join(' ').toLowerCase() }});
      }});
      const filter = () => {{
        const query = search.value.trim().toLowerCase();
        let matches = 0;
        sections.forEach((section) => {{
          const show = !query || section.text.includes(query);
          section.nodes.forEach((node) => node.classList.toggle('section-hidden', !show));
          if (show && query) matches += 1;
        }});
        tocLinks.forEach((link) => {{
          const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
          link.parentElement.hidden = Boolean(query) && Boolean(target?.classList.contains('section-hidden'));
        }});
        searchStatus.textContent = query ? `${{matches}}개 섹션에서 발견` : '';
      }};
      search.addEventListener('input', filter);
      addEventListener('keydown', (event) => {{
        if (event.key === '/' && document.activeElement !== search) {{ event.preventDefault(); search.focus(); }}
        if (event.key === 'Escape') {{ search.value = ''; filter(); search.blur(); sidebar.classList.remove('open'); }}
      }});
    }})();
  </script>
</body>
</html>
'''


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--snapshot", default="2026.07.31")
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.write_text(render(source, snapshot=args.snapshot), encoding="utf-8")
    try:
        rendered_path = output.relative_to(ROOT)
    except ValueError:
        rendered_path = output
    print(f"rendered {rendered_path}")


if __name__ == "__main__":
    main()
