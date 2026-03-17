# Curva A (80/20) Mercado Livre — Streamlit

App local em Python para subir 3 arquivos Excel do Mercado Livre, limpar/normalizar dados, cruzar por ID do anúncio (MLB **sem** o prefixo `MLB`), calcular **Curva A** (produtos que somam ~80% do faturamento) e exibir uma tabela filtrável.

## Requisitos
- Python 3.10+ (recomendado)

## Instalação

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Executar

```bash
streamlit run app.py
```

## Arquivos esperados
- **Estoque**: aba `Anúncios`
- **Métricas**: aba `Relatório`
- **ADS**: aba `Relatório Anúncios patrocinados`

