from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from ml_pipeline import ProcessedData, process_all

PURPLE = "#2A0A4A"
GOLD = "#D4AF37"
WHITE = "#FFFFFF"


def inject_css() -> None:
    st.markdown(
        f"""
<style>
/* App background */
.stApp {{
  background: {PURPLE};
  color: {WHITE};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: {PURPLE};
  border-right: 1px solid rgba(212,175,55,0.25);
}}

/* Headings */
h1, h2, h3, h4, h5, h6, p, label, span, div {{
  color: {WHITE};
}}

/* Buttons */
.stButton > button {{
  background: {GOLD};
  color: #1a1a1a;
  border: 0;
}}

/* File uploader */
div[data-testid="stFileUploader"] {{
  border: 1px solid rgba(212,175,55,0.35);
  border-radius: 10px;
  padding: 6px 10px;
}}

/* Dataframe header text */
div[data-testid="stDataFrame"] thead tr th {{
  color: {WHITE} !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    with st.sidebar:
        st.subheader("Filtro")

        if "experiencia_compra" in d.columns:
            options = sorted(
                [
                    x
                    for x in d["experiencia_compra"].dropna().unique().tolist()
                    if str(x).strip() and str(x) != "nan"
                ]
            )
            if options:
                chosen = st.multiselect("Experiência de compra", options, default=options)
                d = d[d["experiencia_compra"].isin(chosen)]

        if "estoque_deposito" in d.columns:
            mode = st.selectbox("Estoque", ["Qualquer", "<", "<=", "=", ">=", ">"], index=0)
            if mode != "Qualquer":
                qty = st.number_input("Quantidade", min_value=0, value=10, step=1)
                if mode == "<":
                    d = d[d["estoque_deposito"] < qty]
                elif mode == "<=":
                    d = d[d["estoque_deposito"] <= qty]
                elif mode == "=":
                    d = d[d["estoque_deposito"] == qty]
                elif mode == ">=":
                    d = d[d["estoque_deposito"] >= qty]
                elif mode == ">":
                    d = d[d["estoque_deposito"] > qty]

    return d


def style_ads_rows(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def brl(x: object) -> str:
        if x is None:
            return ""
        try:
            v = float(x)
        except Exception:
            return str(x)
        s = f"{v:,.2f}"
        # 1,234.56 -> 1.234,56
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"

    def row_style(row: pd.Series) -> list[str]:
        is_ads = str(row.get("ADS", "")).strip().casefold() == "sim"
        if is_ads:
            return [
                f"background-color: rgba(0,0,0,0); color: {GOLD}; font-weight: 700;"
                for _ in range(len(row))
            ]
        return [f"background-color: rgba(0,0,0,0); color: {WHITE}; font-weight: 400;" for _ in range(len(row))]

    styler = df.style.apply(row_style, axis=1)
    fmt: dict[str, object] = {}
    for c in ["Faturamento Total", "Ticket Médio (BRL)"]:
        if c in df.columns:
            fmt[c] = brl
    styler = styler.format(fmt, na_rep="")
    return styler


def main() -> None:
    st.set_page_config(page_title="Curva A - Mercado Livre", layout="wide")
    inject_css()

    st.title("Curva A (80/20) — Mercado Livre")
    st.caption("Suba os 3 Excel, eu limpo, cruzo e listo os produtos que somam ~80% do faturamento.")

    c1, c2, c3 = st.columns(3)
    with c1:
        estoque_file = st.file_uploader("Arquivo 1 — Estoque (aba: Anúncios)", type=["xlsx"])
    with c2:
        metricas_file = st.file_uploader("Arquivo 2 — Métricas (aba: Relatório)", type=["xlsx"])
    with c3:
        ads_file = st.file_uploader("Arquivo 3 — ADS (aba: Relatório Anúncios patrocinados)", type=["xlsx"])

    if "processed" not in st.session_state:
        st.session_state["processed"] = None

    run = st.button("Processar")

    if run:
        if not (estoque_file and metricas_file and ads_file):
            st.error("Envie os 3 arquivos para processar.")
            st.stop()

        try:
            st.session_state["processed"] = process_all(estoque_file, metricas_file, ads_file)
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            st.stop()

    data: ProcessedData | None = st.session_state.get("processed")
    if data is None:
        st.info("Envie os 3 arquivos e clique em Processar.")
        st.stop()

    base = data.curva_a

    cols = [
        c
        for c in ["ad_id", "estoque_deposito", "vendas_brutas_brl", "ticket_medio", "experiencia_compra", "em_ads"]
        if c in base.columns
    ]
    table = base.loc[:, cols].copy()
    with st.sidebar:
        st.divider()
        st.subheader("Ordenação")
        sort_choices = []
        if "vendas_brutas_brl" in table.columns:
            sort_choices.append("Faturamento Total")
        if "ticket_medio" in table.columns:
            sort_choices.append("Ticket Médio (BRL)")
        if "estoque_deposito" in table.columns:
            sort_choices.append("Estoque")
        sort_choice = st.selectbox("Ordenar por", sort_choices, index=0 if sort_choices else None)
        asc = st.toggle("Crescente", value=False)

    sort_map = {
        "Faturamento Total": "vendas_brutas_brl",
        "Ticket Médio (BRL)": "ticket_medio",
        "Estoque": "estoque_deposito",
    }
    sort_col = sort_map.get(sort_choice) if sort_choices else None
    if sort_col and sort_col in table.columns:
        table = table.sort_values(by=sort_col, ascending=bool(asc), kind="mergesort")

    table = apply_filters(table)

    st.subheader("MLBs Curva A")
    table_display = table.copy()
    if "em_ads" in table_display.columns:
        table_display["ADS"] = table_display["em_ads"].map(lambda x: "Sim" if bool(x) else "Não")
        table_display.drop(columns=["em_ads"], inplace=True)

    table_display = table_display.rename(
        columns={
            "ad_id": "MLB",
            "estoque_deposito": "Estoque",
            "vendas_brutas_brl": "Faturamento Total",
            "ticket_medio": "Ticket Médio (BRL)",
            "experiencia_compra": "Experiência de compra",
        }
    )
    if "MLB" in table_display.columns:
        table_display["MLB"] = pd.to_numeric(table_display["MLB"], errors="coerce").astype("Int64")

    cdl1, cdl2 = st.columns([1, 1])
    with cdl1:
        csv_bytes = table_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Baixar CSV",
            data=csv_bytes,
            file_name="mlbs_curva_a.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cdl2:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            table_display.to_excel(writer, index=False, sheet_name="CurvaA")
        st.download_button(
            "Baixar Excel",
            data=bio.getvalue(),
            file_name="mlbs_curva_a.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.dataframe(style_ads_rows(table_display), use_container_width=True, height=650)


if __name__ == "__main__":
    main()

