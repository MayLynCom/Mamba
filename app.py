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
.stApp {{
  background: {PURPLE};
  color: {WHITE};
}}
section[data-testid="stSidebar"] {{
  background: {PURPLE};
  border-right: 1px solid rgba(212,175,55,0.25);
}}
h1, h2, h3, h4, h5, h6, p, label, span, div {{
  color: {WHITE};
}}
.stButton > button {{
  background: {GOLD};
  color: #1a1a1a;
  border: 0;
}}
div[data-testid="stFileUploader"] {{
  border: 1px solid rgba(212,175,55,0.35);
  border-radius: 10px;
  padding: 6px 10px;
}}
div[data-testid="metric-container"] {{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(212,175,55,0.25);
  border-radius: 10px;
  padding: 12px 16px;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def brl_fmt(v: float) -> str:
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def pct_fmt(v: float) -> str:
    return f"{v * 100:.2f}%"


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    with st.sidebar:
        st.subheader("Filtro")

        if "experiencia_compra" in d.columns:
            opts = sorted(
                [x for x in d["experiencia_compra"].dropna().unique().tolist() if str(x).strip() and str(x) != "nan"]
            )
            if opts:
                chosen = st.multiselect("Experiência de compra", opts, default=opts)
                d = d[d["experiencia_compra"].isin(chosen)]

        if "estoque_deposito" in d.columns:
            mode = st.selectbox("Estoque", ["Qualquer", "<", "<=", "=", ">=", ">"], index=0)
            if mode != "Qualquer":
                qty = st.number_input("Quantidade", min_value=0, value=10, step=1)
                ops = {"<": "__lt__", "<=": "__le__", "=": "__eq__", ">=": "__ge__", ">": "__gt__"}
                d = d[getattr(d["estoque_deposito"], ops[mode])(qty)]

    return d


def style_ads_rows(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        is_ads = str(row.get("ADS", "")).strip().casefold() == "sim"
        if is_ads:
            return [f"background-color: rgba(0,0,0,0); color: {GOLD}; font-weight: 700;" for _ in range(len(row))]
        return [f"background-color: rgba(0,0,0,0); color: {WHITE}; font-weight: 400;" for _ in range(len(row))]

    styler = df.style.apply(row_style, axis=1)

    fmt: dict[str, object] = {}
    for c in ["Faturamento Total", "Ticket Médio (BRL)"]:
        if c in df.columns:
            fmt[c] = lambda x: brl_fmt(float(x)) if x is not None and str(x) not in ("", "nan") else ""
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

    if st.button("Processar"):
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

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    tacos = data.ads_investimento_total / data.gmv_total if data.gmv_total > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("GMV (Faturamento total)", brl_fmt(data.gmv_total))
    with k2:
        st.metric("Investimento ADS", brl_fmt(data.ads_investimento_total))
    with k3:
        st.metric("Receita ADS", brl_fmt(data.ads_receita_total))
    with k4:
        st.metric("TACOS", pct_fmt(tacos))
    st.markdown("---")

    # ── Tabela ────────────────────────────────────────────────────────────────
    base = data.curva_a

    display_cols = [
        c for c in [
            "ad_id", "nome_produto", "estoque_deposito", "unidades_vendidas",
            "vendas_brutas_brl", "ticket_medio", "experiencia_compra", "em_ads",
        ] if c in base.columns
    ]
    table = base.loc[:, display_cols].copy()

    with st.sidebar:
        st.divider()
        st.subheader("Ordenação")
        sort_map = {}
        if "vendas_brutas_brl" in table.columns:
            sort_map["Faturamento Total"] = "vendas_brutas_brl"
        if "ticket_medio" in table.columns:
            sort_map["Ticket Médio (BRL)"] = "ticket_medio"
        if "estoque_deposito" in table.columns:
            sort_map["Estoque"] = "estoque_deposito"
        if "unidades_vendidas" in table.columns:
            sort_map["Unidades vendidas"] = "unidades_vendidas"

        sort_choice = st.selectbox("Ordenar por", list(sort_map.keys()), index=0)
        asc = st.toggle("Crescente", value=False)

    sort_col = sort_map.get(sort_choice)
    if sort_col and sort_col in table.columns:
        table = table.sort_values(by=sort_col, ascending=bool(asc), kind="mergesort")

    table = apply_filters(table)

    # Montar display
    table_display = table.copy()
    if "em_ads" in table_display.columns:
        table_display["ADS"] = table_display["em_ads"].map(lambda x: "Sim" if bool(x) else "Não")
        table_display.drop(columns=["em_ads"], inplace=True)

    rename_map = {
        "ad_id": "MLB",
        "nome_produto": "Nome do produto",
        "estoque_deposito": "Estoque",
        "unidades_vendidas": "Unidades vendidas",
        "vendas_brutas_brl": "Faturamento Total",
        "ticket_medio": "Ticket Médio (BRL)",
        "experiencia_compra": "Experiência de compra",
    }
    table_display = table_display.rename(columns=rename_map)

    if "MLB" in table_display.columns:
        table_display["MLB"] = pd.to_numeric(table_display["MLB"], errors="coerce").astype("Int64")

    dl1, dl2 = st.columns(2)
    with dl1:
        csv_bytes = table_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV", data=csv_bytes, file_name="mlbs_curva_a.csv", mime="text/csv", use_container_width=True)
    with dl2:
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

    st.subheader("MLBs Curva A")
    st.dataframe(style_ads_rows(table_display), use_container_width=True, height=650)


if __name__ == "__main__":
    main()
