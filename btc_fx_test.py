# -*- coding: utf-8 -*-
"""
btc_fx_test.py
비트코인(BTC-USD) 예측 곡선에 '환율'을 반영하면 결과가 달라지는지 확인하는 실험.

all_might.py 의 파이프라인(fetch → 지표 → 잔차 → PCA → XGBoost CV)을 그대로 재사용해
비트코인의 '다음 주 수익률' 예측을 뽑은 뒤, USD/KRW 환율을 두 가지 방식으로 적용한다.

  (1) 곱하기(레벨)  : 예측곡선 × 환율레벨(예: 1350)
        → 부호(방향)는 그대로라 '방향 성공률'은 변하지 않는다. (스케일만 커짐)
  (2) 더하기(수익률): 원화환산 수익률 ≈ 달러수익률 + 환율수익률
        → 부호가 바뀔 수 있어 '방향 성공률/상관계수'가 실제로 달라진다.

결과는 콘솔 표 + btc_fx_result.png 로 출력.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt   # all_might import 시 backend=Agg 로 고정됨

# all_might.py 의 함수/상수 재사용 (main()은 __main__ 가드라 import 시 실행 안 됨)
from all_might import (
    fetch_weekly, build_indicators, calc_residuals, calc_pca, make_features,
    run_cv, direction_accuracy, MODEL_FEATS,
    START_DATE, END_DATE, WARMUP_START, COLORS,
)


# ─────────────────────────────────────────────────────────────
# 1. USD/KRW 환율 주봉 → '다음 주 환율 수익률' 시계열
#    비트코인 타깃이 t→t+1 수익률이므로, 환율도 같은 t→t+1 구간을 맞춰야 한다.
#    fx_next[t] = (환율[t+1] / 환율[t]) - 1   ← shift(-1) 로 정렬
# ─────────────────────────────────────────────────────────────
def fetch_fx_next_return() -> tuple[pd.Series, pd.Series]:
    raw = yf.download(
        "KRW=X",
        start=WARMUP_START.strftime("%Y-%m-%d"),
        end=(END_DATE + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        interval="1wk", auto_adjust=True, progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    fx = raw["Close"].copy()
    fx.index = pd.to_datetime(fx.index).tz_localize(None)
    fx = fx[fx.index <= END_DATE].dropna()
    fx_next = fx.pct_change().shift(-1)        # t→t+1 환율 수익률 (소수 단위)
    print(f"  환율(KRW=X) 수집: {len(fx)}주  최근 레벨 {fx.iloc[-1]:,.1f}원/$")
    return fx, fx_next


# ─────────────────────────────────────────────────────────────
# 2. 비트코인 예측 파이프라인 (all_might 와 동일 절차)
# ─────────────────────────────────────────────────────────────
def build_btc():
    df = fetch_weekly("BTC-USD")
    df = build_indicators("BTC-USD", df)
    df = calc_residuals(df)
    df, _, _ = calc_pca(df)
    df = make_features(df)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)].copy()
    return df


# ─────────────────────────────────────────────────────────────
# 3. 한 모델의 예측(USD)에 환율을 반영해 지표 비교
# ─────────────────────────────────────────────────────────────
def compare_one(pred_s: pd.Series, true_s: pd.Series,
                fx_level: pd.Series, fx_next: pd.Series) -> dict:
    # USD 기준 (원본)
    res = {"USD_성공률": direction_accuracy(true_s.values, pred_s.values),
           "USD_상관":  float(np.corrcoef(true_s.values, pred_s.values)[0, 1])}

    # 같은 날짜의 환율 수익률/레벨 정렬
    fxr  = fx_next.reindex(pred_s.index)
    fxl  = fx_level.reindex(pred_s.index)
    mask = fxr.notna() & fxl.notna()
    p, t = pred_s[mask], true_s[mask]
    fxr, fxl = fxr[mask], fxl[mask]

    # (1) 곱하기: 예측·실제 모두 환율레벨을 곱함 → 부호 불변 → 성공률 동일
    p_mul, t_mul = p * fxl, t * fxl
    res["곱(레벨)_성공률"] = direction_accuracy(t_mul.values, p_mul.values)
    res["곱(레벨)_상관"]  = float(np.corrcoef(t_mul.values, p_mul.values)[0, 1])

    # (2) 더하기: 원화환산 수익률 = 달러수익률 + 환율수익률
    p_krw, t_krw = p + fxr.values, t + fxr.values
    res["원화(+환율)_성공률"] = direction_accuracy(t_krw.values, p_krw.values)
    res["원화(+환율)_상관"]  = float(np.corrcoef(t_krw.values, p_krw.values)[0, 1])

    return res, (t_krw, p_krw, t, p)


# ─────────────────────────────────────────────────────────────
# 4. 실행
# ─────────────────────────────────────────────────────────────
def main():
    print("=== 비트코인 예측 × 환율 실험 ===")
    fx_level, fx_next = fetch_fx_next_return()
    df = build_btc()

    rows = {}
    plot_payload = None
    for mname, feats in MODEL_FEATS.items():
        metrics, pred_s, true_s = run_cv(df, feats)
        if metrics is None:
            print(f"  {mname}: 데이터 부족"); continue
        res, payload = compare_one(pred_s, true_s, fx_level, fx_next)
        rows[mname] = res
        if mname == "Model C (전체)":
            plot_payload = (pred_s.index, payload)

    # ── 콘솔 표 ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  {'모델':<16}{'USD성공률':>10}{'원화성공률':>12}{'USD상관':>10}{'원화상관':>10}")
    print("  " + "-" * 70)
    for m, r in rows.items():
        print(f"  {m:<16}{r['USD_성공률']:>9.1%}{r['원화(+환율)_성공률']:>11.1%}"
              f"{r['USD_상관']:>10.3f}{r['원화(+환율)_상관']:>10.3f}")
    print("=" * 78)
    print("  * 곱(레벨) 방식은 부호가 안 바뀌어 성공률이 USD와 동일 → 표에서 생략")
    if rows:
        any_m = next(iter(rows))
        print(f"    (검증: {any_m} 곱(레벨) 성공률={rows[any_m]['곱(레벨)_성공률']:.1%}"
              f" == USD {rows[any_m]['USD_성공률']:.1%})")

    # ── 그래프 (Model C) ──────────────────────────────────────
    if plot_payload is not None:
        idx, (t_krw, p_krw, t_usd, p_usd) = plot_payload
        x = idx[-len(t_krw):]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig.patch.set_facecolor(COLORS["bg"])
        fig.suptitle("비트코인 다음 주 수익률 예측  |  USD vs 원화(+환율)  [Model C]",
                     fontsize=12, fontweight="bold", color=COLORS["ink"])
        for ax, (tt, pp, ttl) in zip(
                (a1, a2),
                [((t_usd.values), p_usd.values, "USD 기준 (원본)"),
                 (t_krw, p_krw, "원화환산 = 달러수익률 + 환율수익률")]):
            ax.set_facecolor(COLORS["bg"])
            ax.plot(x, np.asarray(tt) * 100, color=COLORS["ink"], lw=1.6, label="실제")
            ax.plot(x, np.asarray(pp) * 100, color=COLORS["red"], lw=1.6, ls="--", label="예측")
            ax.axhline(0, color="#aaa", lw=0.7, ls=":")
            ax.set_title(ttl, fontsize=10); ax.set_ylabel("%", fontsize=8)
            ax.legend(fontsize=8); ax.tick_params(labelsize=8)
        fig.tight_layout()
        out = "btc_fx_result.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close(fig)
        print(f"\n✔ 그래프 저장: {out}")


if __name__ == "__main__":
    main()
