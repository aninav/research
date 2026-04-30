from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / 'data'
TABLES_DIR = DATA_DIR / 'tables'
MASTER_PATH = DATA_DIR / 'master_dataset.csv'
TICKERS = ['SPY', 'QQQ']
CATEGORIES = ['inflation', 'labor', 'monetary_policy', 'consumption', 'growth']
HIGH_INFLATION_YEARS = {2021, 2022}
HIGH_INFLATION_CUTOFF = pd.Timestamp('2023-07-01', tz='America/New_York')

def load_dataset(path: Path) -> pd.DataFrame:
    log.info('Loading dataset from: %s', path)
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_high_inflation_regime'] = (df['timestamp'] < HIGH_INFLATION_CUTOFF).astype(int)
    log.info('Loaded %d events.', len(df))
    return df

def build_feature_matrix(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    target_col = f'{ticker}_rv_post_30m'
    pre_vol_col = f'{ticker}_rv_pre_30m'
    if target_col not in df.columns:
        raise ValueError(f'Column {target_col} not found in dataset.')
    sub = df[['event_category', 'year', 'month', 'quarter', 'is_high_inflation_regime', target_col, pre_vol_col]].dropna(subset=[target_col, pre_vol_col]).copy()
    cats_to_dummy = ['inflation', 'labor', 'monetary_policy', 'consumption']
    for cat in cats_to_dummy:
        sub[f'D_{cat}'] = (sub['event_category'] == cat).astype(int)
    for q in [2, 3, 4]:
        sub[f'Q{q}'] = (sub['quarter'] == q).astype(int)
    optional_cols = ['kw_inflation', 'kw_labor', 'kw_monetary', 'kw_consumption', 'kw_growth', 'ent_bls', 'ent_fed', 'ent_bea', 'ent_census']
    present_optional = [c for c in optional_cols if c in df.columns]
    for col in present_optional:
        sub[col] = df.loc[sub.index, col].fillna(0).astype(int)
    y = sub[target_col].reset_index(drop=True)
    return (sub, y)

def _run_ols(X: pd.DataFrame, y: pd.Series, model_name: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    X_const = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_const).fit(cov_type='HC3')
    log.info('%s | R²=%.3f | Adj-R²=%.3f | F=%.2f (p=%.4f) | n=%d', model_name, model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue, int(model.nobs))
    return model

def results_to_df(model, model_name: str, ticker: str) -> pd.DataFrame:
    rows = []
    for var in model.params.index:
        rows.append({'model': model_name, 'ticker': ticker, 'variable': var, 'coef': model.params[var], 'std_err': model.bse[var], 't_stat': model.tvalues[var], 'p_value': model.pvalues[var], 'sig': _sig_stars(model.pvalues[var]), 'r2': model.rsquared, 'adj_r2': model.rsquared_adj, 'n': int(model.nobs), 'f_stat': model.fvalue, 'f_pvalue': model.f_pvalue})
    return pd.DataFrame(rows)

def _sig_stars(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    if p < 0.1:
        return '.'
    return ''

def check_vif(X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    X_const = sm.add_constant(X, has_constant='add')
    vif_data = pd.DataFrame({'model': model_name, 'variable': X_const.columns, 'VIF': [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]})
    high_vif = vif_data[vif_data['VIF'] > 10]
    if not high_vif.empty:
        log.warning('%s: high VIF detected (>10):\n%s', model_name, high_vif.to_string())
    return vif_data

def model_ols_baseline(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub, y = build_feature_matrix(df, ticker)
    X = sub[['D_inflation', 'D_labor', 'D_monetary_policy', 'D_consumption']]
    result = _run_ols(X, y, f'OLS Baseline [{ticker}]')
    return results_to_df(result, 'OLS_Baseline', ticker)

def model_ols_temporal(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub, y = build_feature_matrix(df, ticker)
    X = sub[['D_inflation', 'D_labor', 'D_monetary_policy', 'D_consumption', 'year', 'Q2', 'Q3', 'Q4']]
    result = _run_ols(X, y, f'OLS Temporal [{ticker}]')
    return results_to_df(result, 'OLS_Temporal', ticker)

def model_ols_full(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub, y = build_feature_matrix(df, ticker)
    pre_vol_col = f'{ticker}_rv_pre_30m'
    base_cols = ['D_inflation', 'D_labor', 'D_monetary_policy', 'D_consumption', 'year', 'Q2', 'Q3', 'Q4', 'month', pre_vol_col]
    optional_cols = ['kw_inflation', 'kw_labor', 'kw_monetary', 'kw_consumption', 'ent_bls', 'ent_fed', 'ent_bea']
    feature_cols = base_cols + [c for c in optional_cols if c in sub.columns]
    X = sub[feature_cols].copy()
    result = _run_ols(X, y, f'OLS Full [{ticker}]')
    check_vif(X, f'OLS Full [{ticker}]')
    return results_to_df(result, 'OLS_Full', ticker)

def model_panel_ols(df: pd.DataFrame) -> pd.DataFrame:
    rows_list = []
    for ticker in TICKERS:
        target_col = f'{ticker}_rv_post_30m'
        pre_vol_col = f'{ticker}_rv_pre_30m'
        if target_col not in df.columns:
            continue
        sub = df[['event_category', 'year', 'quarter', 'month', 'is_high_inflation_regime', target_col, pre_vol_col]].dropna(subset=[target_col, pre_vol_col]).copy()
        sub = sub.rename(columns={target_col: 'rv_post_30m', pre_vol_col: 'rv_pre_30m'})
        sub['ticker'] = ticker
        sub['D_QQQ'] = int(ticker == 'QQQ')
        cats = ['inflation', 'labor', 'monetary_policy', 'consumption']
        for cat in cats:
            sub[f'D_{cat}'] = (sub['event_category'] == cat).astype(int)
        for q in [2, 3, 4]:
            sub[f'Q{q}'] = (sub['quarter'] == q).astype(int)
        rows_list.append(sub)
    panel = pd.concat(rows_list, ignore_index=True)
    y = panel['rv_post_30m']
    X = panel[['D_QQQ', 'D_inflation', 'D_labor', 'D_monetary_policy', 'D_consumption', 'year', 'Q2', 'Q3', 'Q4', 'rv_pre_30m', 'is_high_inflation_regime']]
    result = _run_ols(X, y, 'Panel OLS [SPY+QQQ]')
    return results_to_df(result, 'Panel_OLS', 'SPY+QQQ')

def model_interaction(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub, y = build_feature_matrix(df, ticker)
    pre_vol_col = f'{ticker}_rv_pre_30m'
    sub['D_inflation_x_regime'] = sub['D_inflation'] * sub['is_high_inflation_regime']
    X = sub[['D_inflation', 'D_labor', 'D_monetary_policy', 'D_consumption', 'is_high_inflation_regime', 'D_inflation_x_regime', 'year', 'Q2', 'Q3', 'Q4', pre_vol_col]]
    result = _run_ols(X, y, f'Interaction [{ticker}]')
    return results_to_df(result, 'Interaction', ticker)

def build_summary_table(all_results: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(all_results, ignore_index=True)
    summary = combined.groupby(['model', 'ticker']).agg(r2=('r2', 'first'), adj_r2=('adj_r2', 'first'), n=('n', 'first'), f_stat=('f_stat', 'first'), f_pvalue=('f_pvalue', 'first')).reset_index()
    summary['r2'] = summary['r2'].round(4)
    summary['adj_r2'] = summary['adj_r2'].round(4)
    summary['f_stat'] = summary['f_stat'].round(2)
    summary['f_pvalue'] = summary['f_pvalue'].apply(lambda p: f'{p:.4f}')
    return summary

def run_regression_analysis(input_path: Path, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(input_path)
    all_results: list[pd.DataFrame] = []
    log.info('\n' + '=' * 60)
    log.info('MODEL 1: OLS Baseline (category dummies only)')
    log.info('=' * 60)
    baseline_dfs = []
    for ticker in TICKERS:
        res = model_ols_baseline(df, ticker)
        all_results.append(res)
        baseline_dfs.append(res)
    pd.concat(baseline_dfs).to_csv(tables_dir / 'table_ols_baseline.csv', index=False)
    log.info('Saved: table_ols_baseline.csv')
    log.info('\n' + '=' * 60)
    log.info('MODEL 2: OLS with Temporal Controls')
    log.info('=' * 60)
    temporal_dfs = []
    for ticker in TICKERS:
        res = model_ols_temporal(df, ticker)
        all_results.append(res)
        temporal_dfs.append(res)
    pd.concat(temporal_dfs).to_csv(tables_dir / 'table_ols_temporal.csv', index=False)
    log.info('Saved: table_ols_temporal.csv')
    log.info('\n' + '=' * 60)
    log.info('MODEL 3: OLS Full Feature Set')
    log.info('=' * 60)
    full_dfs = []
    for ticker in TICKERS:
        res = model_ols_full(df, ticker)
        all_results.append(res)
        full_dfs.append(res)
    pd.concat(full_dfs).to_csv(tables_dir / 'table_ols_full.csv', index=False)
    log.info('Saved: table_ols_full.csv')
    log.info('\n' + '=' * 60)
    log.info('MODEL 4: Panel OLS (pooled SPY + QQQ)')
    log.info('=' * 60)
    panel_res = model_panel_ols(df)
    all_results.append(panel_res)
    panel_res.to_csv(tables_dir / 'table_panel_ols.csv', index=False)
    log.info('Saved: table_panel_ols.csv')
    log.info('\n' + '=' * 60)
    log.info('MODEL 5: Interaction (inflation x high-inflation regime)')
    log.info('=' * 60)
    interact_dfs = []
    for ticker in TICKERS:
        res = model_interaction(df, ticker)
        all_results.append(res)
        interact_dfs.append(res)
    pd.concat(interact_dfs).to_csv(tables_dir / 'table_interaction.csv', index=False)
    log.info('Saved: table_interaction.csv')
    summary = build_summary_table(all_results)
    summary.to_csv(tables_dir / 'table_regression_summary.csv', index=False)
    log.info('Saved: table_regression_summary.csv')
    log.info('\n=== REGRESSION SUMMARY ===')
    print(summary.to_string(index=False))
    full_combined = pd.concat(full_dfs)
    log.info('\n=== OLS FULL: COEFFICIENT TABLE ===')
    display_cols = ['ticker', 'variable', 'coef', 'std_err', 't_stat', 'p_value', 'sig']
    print(full_combined[display_cols].to_string(index=False))
    interact_combined = pd.concat(interact_dfs)
    log.info('\n=== INTERACTION MODEL: COEFFICIENT TABLE ===')
    print(interact_combined[display_cols].to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description='Run OLS and panel regression analysis on macro event volatility data. Requires master_dataset.csv to exist.')
    parser.add_argument('--input', default=str(MASTER_PATH), help=f'Path to master_dataset.csv. Default: {MASTER_PATH}')
    parser.add_argument('--tables-dir', default=str(TABLES_DIR), help=f'Directory for output tables. Default: {TABLES_DIR}')
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        log.error('Input file not found: %s\nRun build_market_dataset.py first to generate master_dataset.csv.', input_path)
        sys.exit(1)
    run_regression_analysis(input_path=input_path, tables_dir=Path(args.tables_dir))
if __name__ == '__main__':
    main()
