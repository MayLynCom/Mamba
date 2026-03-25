from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import unicodedata

import numpy as np
import pandas as pd


def _norm_str(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def normalize_ad_id(value: object) -> str:
    s = _norm_str(value).upper()
    s = s.replace("MLB", "")
    digits = re.findall(r"\d+", s)
    return "".join(digits).lstrip("0") or ("".join(digits) if digits else "")


def parse_brl(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.number)) and not (isinstance(value, float) and np.isnan(value)):
        return float(value)
    s = _norm_str(value)
    if not s:
        return 0.0
    s = s.replace("R$", "").replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(".", "")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _normalize_columns(cols: Iterable[object]) -> list[str]:
    out: list[str] = []
    for c in cols:
        s = _norm_str(c)
        s = re.sub(r"\s+", " ", s).strip()
        out.append(s)
    return out


def _col_key(s: object) -> str:
    txt = _norm_str(s)
    if not txt:
        return ""
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"\s+", " ", txt).strip().casefold()
    return txt


def _find_header_row_by_groups(raw: pd.DataFrame, required_groups: Sequence[Sequence[str]]) -> Optional[int]:
    groups_norm = [[_col_key(x) for x in g] for g in required_groups]
    for i in range(min(len(raw), 120)):
        row = raw.iloc[i].tolist()
        cells = {_col_key(x) for x in row if _col_key(x)}
        ok = all(any(g in cells for g in group) for group in groups_norm)
        if ok:
            return i
    return None


def _pick_col_by_prefix(col_map: dict[str, str], prefix: str) -> Optional[str]:
    p = _col_key(prefix)
    if not p:
        return None
    for k, original in col_map.items():
        if k.startswith(p):
            return original
    return None


def _read_with_detected_header_groups(
    file_like, sheet_name: str, required_groups: Sequence[Sequence[str]]
) -> pd.DataFrame:
    raw = pd.read_excel(file_like, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_row = _find_header_row_by_groups(raw, required_groups)
    if header_row is None:
        raise ValueError(
            f"Não consegui achar o cabeçalho na aba '{sheet_name}'. "
            f"Esperava grupos de colunas: {required_groups}"
        )
    header = _normalize_columns(raw.iloc[header_row].tolist())
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")
    return df


def _rewind_file(file_like) -> None:
    if hasattr(file_like, "seek"):
        file_like.seek(0)


@dataclass(frozen=True)
class ProcessedData:
    merged: pd.DataFrame
    curva_a: pd.DataFrame
    gmv_total: float
    ads_investimento_total: float
    ads_receita_total: float


# ─── Leitores ───────────────────────────────────────────────────────────────

def load_estoque(file_like) -> pd.DataFrame:
    df = _read_with_detected_header_groups(
        file_like,
        sheet_name="Anúncios",
        required_groups=[
            ["Código do Anúncio", "Codigo do Anuncio", "Código do anúncio", "Codigo do anuncio"],
            ["Estoque no depósito", "Estoque no deposito"],
        ],
    )
    col_map = {_col_key(c): c for c in df.columns}
    codigo_col = (
        col_map.get(_col_key("Código do Anúncio"))
        or col_map.get(_col_key("Código do anúncio"))
    )
    estoque_col = col_map.get(_col_key("Estoque no depósito")) or col_map.get(_col_key("Estoque no deposito"))
    if not (codigo_col and estoque_col):
        raise ValueError("Não encontrei as colunas de Código/Estoque no arquivo de estoque.")
    df = df.loc[:, [codigo_col, estoque_col]].copy()
    df.rename(columns={codigo_col: "ad_id", estoque_col: "estoque_deposito"}, inplace=True)
    df["ad_id"] = df["ad_id"].map(normalize_ad_id)
    df = df[df["ad_id"].astype(bool)]
    df["estoque_deposito"] = pd.to_numeric(df["estoque_deposito"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["ad_id"], keep="last")
    return df


def load_metricas(file_like) -> pd.DataFrame:
    raw = pd.read_excel(file_like, sheet_name="Relatório", header=None, engine="openpyxl")
    raw = raw.iloc[5:].reset_index(drop=True)

    header_row = _find_header_row_by_groups(
        raw,
        required_groups=[
            ["Status atual", "Status Atual", "Status"],
            ["Experiência de compra", "Experiencia de compra"],
            ["Unidades vendidas", "Unidades Vendidas"],
            ["Vendas brutas(BRL)", "Vendas brutas (BRL)", "Vendas brutas", "Vendas brutas BRL"],
        ],
    )
    if header_row is None:
        header_row = 0

    header = _normalize_columns(raw.iloc[header_row].tolist())
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")

    first_col = df.columns[0]
    df.rename(columns={first_col: "ad_id"}, inplace=True)
    df["ad_id"] = df["ad_id"].map(normalize_ad_id)

    col_map = {_col_key(c): c for c in df.columns}
    status_col = col_map.get(_col_key("Status atual")) or col_map.get(_col_key("Status"))
    anuncio_col = col_map.get(_col_key("Anúncio")) or _pick_col_by_prefix(col_map, "Anuncio")
    exp_col = col_map.get(_col_key("Experiência de compra")) or col_map.get(_col_key("Experiencia de compra"))
    qtd_col = col_map.get(_col_key("Quantidade de vendas")) or col_map.get(_col_key("Quantidade vendas"))
    unid_col = col_map.get(_col_key("Unidades vendidas"))
    fat_col = (
        col_map.get(_col_key("Vendas brutas(BRL)"))
        or col_map.get(_col_key("Vendas brutas (BRL)"))
        or col_map.get(_col_key("Vendas brutas"))
    )

    missing = [
        n for n, c in [
            ("Status atual", status_col),
            ("Experiência de compra", exp_col),
            ("Unidades vendidas", unid_col),
            ("Vendas brutas(BRL)", fat_col),
        ] if not c
    ]
    if missing:
        raise ValueError(f"Faltando colunas na aba 'Relatório': {missing}")

    use_cols: list[str] = ["ad_id"]
    if anuncio_col:
        use_cols.append(anuncio_col)
    use_cols.extend([status_col, exp_col])
    if qtd_col:
        use_cols.append(qtd_col)
    use_cols.extend([unid_col, fat_col])
    df = df.loc[:, use_cols].copy()

    df[status_col] = df[status_col].map(_norm_str)
    status_key = df[status_col].astype(str).str.strip().str.casefold()
    df = df[status_key.str.startswith("ativ")]

    rename_map: dict[str, str] = {
        status_col: "status_atual",
        exp_col: "experiencia_compra",
        (qtd_col or "Quantidade de vendas"): "quantidade_vendas",
        unid_col: "unidades_vendidas",
        fat_col: "vendas_brutas_brl",
    }
    if anuncio_col:
        rename_map[anuncio_col] = "nome_produto"
    df.rename(columns=rename_map, inplace=True)

    if "quantidade_vendas" in df.columns:
        df["quantidade_vendas"] = pd.to_numeric(df["quantidade_vendas"], errors="coerce").fillna(0).astype(int)
    df["unidades_vendidas"] = pd.to_numeric(df["unidades_vendidas"], errors="coerce").fillna(0).astype(int)
    df["vendas_brutas_brl"] = df["vendas_brutas_brl"].map(parse_brl)
    if "nome_produto" in df.columns:
        df["nome_produto"] = df["nome_produto"].map(_norm_str)

    denom = df["unidades_vendidas"].replace(0, np.nan)
    df["ticket_medio"] = (df["vendas_brutas_brl"] / denom).fillna(0.0)
    df = df[df["ad_id"].astype(bool)]
    return df


def load_ads(file_like) -> pd.DataFrame:
    raw = pd.read_excel(file_like, sheet_name="Relatório Anúncios patrocinados", header=None, engine="openpyxl")
    raw = raw.iloc[1:].reset_index(drop=True)

    header_row = _find_header_row_by_groups(
        raw,
        required_groups=[
            ["Código do anúncio", "Codigo do anuncio", "Código do Anúncio", "Codigo do Anuncio"],
            ["Status"],
            ["Receita"],
            ["Investimento"],
            ["Roas", "ROAS"],
            ["Vendas por publicidade", "Vendas por Publicidade"],
        ],
    )
    if header_row is None:
        header_row = 0

    header = _normalize_columns(raw.iloc[header_row].tolist())
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")

    col_map = {_col_key(c): c for c in df.columns}
    codigo_col = (
        col_map.get(_col_key("Código do anúncio"))
        or col_map.get(_col_key("Codigo do anuncio"))
        or col_map.get(_col_key("Código do Anúncio"))
        or col_map.get(_col_key("Codigo do Anuncio"))
    )
    status_col = col_map.get(_col_key("Status"))
    receita_col = _pick_col_by_prefix(col_map, "Receita")
    inv_col = _pick_col_by_prefix(col_map, "Investimento")
    roas_col = _pick_col_by_prefix(col_map, "Roas") or _pick_col_by_prefix(col_map, "ROAS")
    vpp_col = _pick_col_by_prefix(col_map, "Vendas por publicidade")

    missing = [
        n for n, c in [
            ("Código do anúncio", codigo_col),
            ("Status", status_col),
            ("Receita", receita_col),
            ("Investimento", inv_col),
            ("Roas", roas_col),
            ("Vendas por publicidade", vpp_col),
        ] if not c
    ]
    if missing:
        raise ValueError(f"Faltando colunas na aba de ADS: {missing}")

    df = df.loc[:, [codigo_col, status_col, receita_col, inv_col, roas_col, vpp_col]].copy()
    df[status_col] = df[status_col].map(_norm_str)
    status_key = df[status_col].astype(str).str.strip().str.casefold()
    df = df[status_key.str.startswith("ativ")]

    df.rename(
        columns={
            codigo_col: "ad_id",
            receita_col: "ads_receita",
            inv_col: "ads_investimento",
            roas_col: "ads_roas",
            vpp_col: "ads_vendas_publicidade",
        },
        inplace=True,
    )
    df["ad_id"] = df["ad_id"].map(normalize_ad_id)
    df = df[df["ad_id"].astype(bool)]

    df["ads_receita"] = df["ads_receita"].map(parse_brl)
    df["ads_investimento"] = df["ads_investimento"].map(parse_brl)
    df["ads_roas"] = pd.to_numeric(df["ads_roas"], errors="coerce").fillna(0.0)
    df["ads_vendas_publicidade"] = df["ads_vendas_publicidade"].map(parse_brl)
    df["em_ads"] = True

    df = df.drop_duplicates(subset=["ad_id"], keep="last")
    return df[["ad_id", "em_ads", "ads_receita", "ads_investimento", "ads_roas", "ads_vendas_publicidade"]]


def load_ads_totals(file_like) -> tuple[float, float]:
    """Soma toda a coluna Investimento e Receita da aba ADS (sem filtrar por status)."""
    raw = pd.read_excel(file_like, sheet_name="Relatório Anúncios patrocinados", header=None, engine="openpyxl")
    raw = raw.iloc[1:].reset_index(drop=True)

    header_row = _find_header_row_by_groups(
        raw,
        required_groups=[
            ["Código do anúncio", "Codigo do anuncio", "Código do Anúncio", "Codigo do Anuncio"],
            ["Receita"],
            ["Investimento"],
        ],
    )
    if header_row is None:
        header_row = 0

    header = _normalize_columns(raw.iloc[header_row].tolist())
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")

    col_map = {_col_key(c): c for c in df.columns}
    receita_col = _pick_col_by_prefix(col_map, "Receita")
    inv_col = _pick_col_by_prefix(col_map, "Investimento")
    if not receita_col or not inv_col:
        raise ValueError("Não encontrei colunas Receita/Investimento para totais de ADS.")

    total_inv = float(df[inv_col].map(parse_brl).sum())
    total_rec = float(df[receita_col].map(parse_brl).sum())
    return total_inv, total_rec


def compute_curva_a(df: pd.DataFrame, faturamento_col: str = "vendas_brutas_brl", frac: float = 0.80) -> pd.DataFrame:
    if df.empty:
        d0 = df.copy()
        d0["participacao"] = 0.0
        d0["acumulado"] = 0.0
        d0["curva"] = "A"
        return d0

    d = df.copy()
    d[faturamento_col] = pd.to_numeric(d[faturamento_col], errors="coerce").fillna(0.0)
    d = d.sort_values(by=faturamento_col, ascending=False, kind="mergesort").reset_index(drop=True)

    total = float(d[faturamento_col].sum())
    if total <= 0:
        d["participacao"] = 0.0
        d["acumulado"] = 0.0
        d["curva"] = "A"
        return d

    d["participacao"] = d[faturamento_col] / total
    d["acumulado"] = d["participacao"].cumsum()

    idx = int(np.searchsorted(d["acumulado"].to_numpy(), frac, side="left"))
    d["curva"] = np.where(np.arange(len(d)) <= idx, "A", "B/C")
    return d


def process_all(estoque_file, metricas_file, ads_file) -> ProcessedData:
    _rewind_file(estoque_file)
    estoque = load_estoque(estoque_file)

    _rewind_file(metricas_file)
    metricas = load_metricas(metricas_file)
    gmv_total = float(pd.to_numeric(metricas["vendas_brutas_brl"], errors="coerce").fillna(0.0).sum())

    _rewind_file(ads_file)
    ads_inv_total, ads_rec_total = load_ads_totals(ads_file)

    _rewind_file(ads_file)
    ads = load_ads(ads_file)

    merged = metricas.merge(estoque, on="ad_id", how="left").merge(ads, on="ad_id", how="left")
    merged["em_ads"] = merged["em_ads"].fillna(False)
    merged["estoque_deposito"] = merged["estoque_deposito"].fillna(0).astype(int)

    scored = compute_curva_a(merged, faturamento_col="vendas_brutas_brl", frac=0.80)
    curva_a = scored[scored["curva"].eq("A")].copy()

    return ProcessedData(
        merged=scored,
        curva_a=curva_a,
        gmv_total=gmv_total,
        ads_investimento_total=ads_inv_total,
        ads_receita_total=ads_rec_total,
    )
