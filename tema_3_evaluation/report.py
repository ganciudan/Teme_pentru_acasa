import html as _html
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "#22c55e"
    elif score >= 0.7:
        return "#f59e0b"
    return "#ef4444"


def _score_label(score: float) -> str:
    if score >= 0.9:
        return "Excelent"
    elif score >= 0.7:
        return "Acceptabil"
    return "Slab"


def _card_html(i: int, r: dict) -> str:
    c1 = _score_color(r["relevanta_score"])
    c2 = _score_color(r["bias_score"])

    return f"""
    <div class="card">
        <div class="card-header">
            <span class="case-number">#{i}</span>
            <p class="question">{_html.escape(r["input"])}</p>
        </div>
        <div class="scores-row">
            <div class="score-box" style="border-color:{c1}">
                <div class="score-label">Relevanță Fitness</div>
                <div class="score-value" style="color:{c1}">{r["relevanta_score"]:.2f}</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{r['relevanta_score']*100:.0f}%;background:{c1}"></div>
                </div>
                <div class="score-tag" style="background:{c1}">{_score_label(r["relevanta_score"])}</div>
                <p class="score-reason">{_html.escape(r["relevanta_reason"] or "")}</p>
            </div>
            <div class="score-box" style="border-color:{c2}">
                <div class="score-label">Bias Fitness</div>
                <div class="score-value" style="color:{c2}">{r["bias_score"]:.2f}</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{r['bias_score']*100:.0f}%;background:{c2}"></div>
                </div>
                <div class="score-tag" style="background:{c2}">{_score_label(r["bias_score"])}</div>
                <p class="score-reason">{_html.escape(r["bias_reason"] or "")}</p>
            </div>
        </div>
        <div class="response-section">
            <div class="response-label">Răspuns LLM</div>
            <div class="markdown-body" data-markdown="{_html.escape(r["response"])}"></div>
        </div>
    </div>"""


def save_report(results: list[dict], scores1: list[float], scores2: list[float], threshold: float) -> Path:
    relevance_pct = sum(s >= threshold for s in scores1) / len(scores1) * 100
    bias_pct = sum(s >= threshold for s in scores2) / len(scores2) * 100
    run_time = datetime.now().strftime("%d %B %Y, %H:%M:%S")

    cards = "".join(_card_html(i, r) for i, r in enumerate(results, 1))

    html_content = f"""<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raport Evaluare — Instructor Fitness</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 2rem 1rem;
            min-height: 100vh;
        }}
        .container {{ max-width: 960px; margin: 0 auto; }}

        .report-header {{ text-align: center; margin-bottom: 2.5rem; }}
        .report-header h1 {{ font-size: 2rem; font-weight: 700; color: #f8fafc; margin-bottom: .4rem; }}
        .report-header .subtitle {{ color: #94a3b8; font-size: .95rem; }}

        .summary {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 2.5rem;
        }}
        .summary-box {{
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #334155;
        }}
        .summary-box .s-label {{ color: #94a3b8; font-size: .85rem; margin-bottom: .5rem; text-transform: uppercase; letter-spacing: .05em; }}
        .summary-box .s-value {{ font-size: 2.5rem; font-weight: 700; }}
        .summary-box .s-sub {{ color: #64748b; font-size: .8rem; margin-top: .3rem; }}

        .card {{
            background: #1e293b;
            border-radius: 14px;
            border: 1px solid #334155;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}
        .card-header {{
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid #334155;
            display: flex;
            gap: 1rem;
            align-items: flex-start;
        }}
        .case-number {{
            background: #3b82f6;
            color: #fff;
            font-weight: 700;
            font-size: .8rem;
            padding: .2rem .55rem;
            border-radius: 6px;
            white-space: nowrap;
            margin-top: 2px;
        }}
        .question {{ font-size: 1rem; color: #f1f5f9; line-height: 1.5; }}

        .scores-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: #334155;
        }}
        .score-box {{
            background: #1e293b;
            padding: 1.25rem 1.5rem;
            border-top: 3px solid transparent;
        }}
        .score-label {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; color: #94a3b8; margin-bottom: .4rem; }}
        .score-value {{ font-size: 2rem; font-weight: 700; margin-bottom: .5rem; }}
        .score-bar-bg {{ background: #0f172a; border-radius: 4px; height: 6px; margin-bottom: .6rem; }}
        .score-bar-fill {{ height: 6px; border-radius: 4px; }}
        .score-tag {{
            display: inline-block;
            color: #fff;
            font-size: .72rem;
            font-weight: 600;
            padding: .15rem .55rem;
            border-radius: 20px;
            margin-bottom: .75rem;
            text-transform: uppercase;
            letter-spacing: .04em;
        }}
        .score-reason {{ font-size: .83rem; color: #94a3b8; line-height: 1.55; }}

        .response-section {{
            border-top: 1px solid #334155;
            padding: 1.25rem 1.5rem;
        }}
        .response-label {{
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: #64748b;
            margin-bottom: .75rem;
        }}
        .markdown-body {{ font-size: .88rem; line-height: 1.7; color: #cbd5e1; }}
        .markdown-body h1,.markdown-body h2,.markdown-body h3 {{ color:#f1f5f9; margin: 1rem 0 .5rem; }}
        .markdown-body strong {{ color: #f8fafc; }}
        .markdown-body table {{ border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .82rem; }}
        .markdown-body th {{ background: #0f172a; color: #94a3b8; padding: .5rem .75rem; text-align: left; border: 1px solid #334155; }}
        .markdown-body td {{ padding: .45rem .75rem; border: 1px solid #334155; }}
        .markdown-body tr:nth-child(even) td {{ background: #0f172a33; }}
        .markdown-body ul,.markdown-body ol {{ padding-left: 1.4rem; }}
        .markdown-body li {{ margin: .25rem 0; }}
        .markdown-body blockquote {{ border-left: 3px solid #3b82f6; padding-left: 1rem; color: #64748b; margin: .6rem 0; }}
        .markdown-body hr {{ border: none; border-top: 1px solid #334155; margin: 1rem 0; }}

        .footer {{ text-align: center; color: #334155; font-size: .8rem; margin-top: 3rem; }}
    </style>
</head>
<body>
<div class="container">
    <div class="report-header">
        <h1>Raport Evaluare — Instructor Fitness</h1>
        <p class="subtitle">Generat la {run_time} &nbsp;·&nbsp; {len(results)} cazuri de test &nbsp;·&nbsp; Prag: {threshold}</p>
    </div>

    <div class="summary">
        <div class="summary-box">
            <div class="s-label">Relevanță Fitness</div>
            <div class="s-value" style="color:{_score_color(relevance_pct/100)}">{relevance_pct:.0f}%</div>
            <div class="s-sub">cazuri cu scor ≥ {threshold}</div>
        </div>
        <div class="summary-box">
            <div class="s-label">Fără Bias</div>
            <div class="s-value" style="color:{_score_color(bias_pct/100)}">{bias_pct:.0f}%</div>
            <div class="s-sub">cazuri cu scor ≥ {threshold}</div>
        </div>
    </div>

    {cards}

    <div class="footer">Instructor Fitness — Evaluation Report</div>
</div>
<script>
    document.querySelectorAll(".markdown-body[data-markdown]").forEach(el => {{
        el.innerHTML = marked.parse(el.getAttribute("data-markdown"));
    }});
</script>
</body>
</html>"""

    output_file = OUTPUT_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    output_file.write_text(html_content, encoding="utf-8")
    return output_file
