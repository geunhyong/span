import html

import pandas as pd
import streamlit.components.v1 as components


def render_presentation_table(
    df: pd.DataFrame,
    title: str | None = None,
    footnote: str | None = None,
    left_align_cols: list[str] | None = None,
    height: int | None = None,
    cell_style_rules: dict[str, dict[str, str]] | None = None,
) -> None:
    """
    발표용 카드형 HTML 표를 렌더링한다.

    Parameters
    ----------
    df : pd.DataFrame
        화면에 표시할 표 데이터.
    title : str | None
        표 카드 제목.
    footnote : str | None
        표 아래 설명 문구.
    left_align_cols : list[str] | None
        좌측 정렬할 컬럼명 목록.
    height : int | None
        Streamlit component 높이. None이면 행 개수에 따라 자동 계산한다.
    cell_style_rules : dict[str, dict[str, str]] | None
        특정 컬럼의 특정 값에만 추가 CSS style을 적용한다.
        예: {"결과": {"적중": "background-color:#eaf7ea; color:#166534; font-weight:800;"}}
    """
    if left_align_cols is None:
        left_align_cols = []

    if cell_style_rules is None:
        cell_style_rules = {}

    display_df = df.copy()

    if height is None:
        base_height = 95
        row_height = 50
        footnote_height = 70 if footnote else 40
        safety_margin = 30
        height = base_height + (len(display_df) * row_height) + footnote_height + safety_margin

    headers = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in display_df.columns
    )

    rows_html = ""

    for _, row in display_df.iterrows():
        row_html = ""

        for col in display_df.columns:
            val = row[col]
            text = "" if pd.isna(val) else str(val)
            safe_text = html.escape(text)

            align_class = "left-cell" if col in left_align_cols else ""

            extra_style = ""
            if col in cell_style_rules:
                extra_style = cell_style_rules[col].get(text, "")

            style_attr = f' style="{html.escape(extra_style, quote=True)}"' if extra_style else ""

            row_html += f'<td class="{align_class}"{style_attr}>{safe_text}</td>'

        rows_html += f"<tr>{row_html}</tr>"

    title_html = (
        f'<div class="table-card-title">{html.escape(str(title))}</div>'
        if title
        else ""
    )

    footnote_html = (
        f'<div class="table-footnote">{html.escape(str(footnote))}</div>'
        if footnote
        else ""
    )

    html_text = f"""
    <style>
    .table-card {{
        background-color: #fbf8f1;
        border: 1px solid #e3dacb;
        border-radius: 16px;
        padding: 14px 18px 16px 18px;
        margin: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
        box-sizing: border-box;
    }}

    .table-card-title {{
        font-size: 20px;
        font-weight: 800;
        color: #3d352d;
        margin-bottom: 12px;
    }}

    .presentation-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 15px;
        color: #433c35;
    }}

    .presentation-table thead th {{
        background-color: #e9e1d3;
        color: #3d352d;
        font-weight: 800;
        text-align: center;
        padding: 11px 9px;
        border: 1px solid #ddd2c3;
        white-space: normal;
        word-break: keep-all;
        overflow-wrap: break-word;
    }}

    .presentation-table tbody td {{
        background-color: #fffdfa;
        padding: 10px 9px;
        border: 1px solid #e7ddd0;
        text-align: center;
        vertical-align: middle;
        white-space: normal;
        word-break: keep-all;
        overflow-wrap: break-word;
    }}

    .presentation-table tbody tr:nth-child(even) td {{
        background-color: #fcf8f2;
    }}

    .left-cell {{
        text-align: left !important;
    }}

    .table-footnote {{
        margin-top: 12px;
        font-size: 14px;
        line-height: 1.6;
        color: #6e665e;
    }}
    </style>

    <div class="table-card">
        {title_html}
        <table class="presentation-table">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        {footnote_html}
    </div>
    """

    components.html(html_text, height=height, scrolling=False)