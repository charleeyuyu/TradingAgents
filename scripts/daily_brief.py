#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日美股简报 —— TradingAgents 外壳

职责：
1. 对每只股票跑一遍 TradingAgents 多 Agent 流程，拿到 BUY/HOLD/SELL
2. 把这些偏专业的分析，用一次额外的模型调用翻译成零基础能看懂的中文
3. 发邮件

设计原则：单只股票失败不影响其他股票；所有失败都在邮件里如实说明。
"""
from __future__ import annotations

import os
import re
import smtplib
import sys
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import pytz

# ---------------------------------------------------------------- 配置

TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", "NVDA,AAPL,TSLA").replace("，", ",").split(",") if t.strip()]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
QUICK_MODEL = os.getenv("QUICK_MODEL", "gemini-3-flash-preview")
DEEP_MODEL = os.getenv("DEEP_MODEL", "gemini-3-flash-preview")
DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "1"))
RISK_ROUNDS = int(os.getenv("RISK_ROUNDS", "1"))

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_TO = os.getenv("EMAIL_TO", "") or EMAIL_SENDER

TZ = pytz.timezone("America/New_York")
TODAY = datetime.now(TZ).strftime("%Y-%m-%d")

SIGNAL_CN = {"BUY": "🟢 买入", "SELL": "🔴 卖出", "HOLD": "🟡 观望"}


def log(msg: str) -> None:
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 分析

def extract_signal(text: str) -> str:
    """从 FINAL TRANSACTION PROPOSAL: **BUY** 里抠出信号。"""
    if not text:
        return "UNKNOWN"
    m = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*{0,2}\s*(BUY|SELL|HOLD)", text, re.I)
    if m:
        return m.group(1).upper()
    # 兜底：末尾 200 字里找关键词
    tail = text[-200:].upper()
    for kw in ("BUY", "SELL", "HOLD"):
        if kw in tail:
            return kw
    return "UNKNOWN"


def analyze_all(tickers: List[str]) -> List[Dict[str, Any]]:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "google"
    config["quick_think_llm"] = QUICK_MODEL
    config["deep_think_llm"] = DEEP_MODEL
    config["max_debate_rounds"] = DEBATE_ROUNDS
    config["max_risk_discuss_rounds"] = RISK_ROUNDS

    log(f"模型: {DEEP_MODEL} / {QUICK_MODEL}，辩论轮数: {DEBATE_ROUNDS}，风控轮数: {RISK_ROUNDS}")
    graph = TradingAgentsGraph(debug=False, config=config)

    results: List[Dict[str, Any]] = []
    for i, ticker in enumerate(tickers, 1):
        log(f"({i}/{len(tickers)}) 开始分析 {ticker} ...")
        try:
            state, decision = graph.propagate(ticker, TODAY)
            raw = state.get("final_trade_decision", "") or str(decision or "")
            results.append({
                "ticker": ticker,
                "signal": extract_signal(raw),
                "raw": raw,
                "error": None,
            })
            log(f"({i}/{len(tickers)}) {ticker} 完成 -> {results[-1]['signal']}")
        except Exception as exc:  # noqa: BLE001 — 单只失败不能拖垮整轮
            log(f"({i}/{len(tickers)}) {ticker} 失败: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results.append({
                "ticker": ticker,
                "signal": "ERROR",
                "raw": "",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return results


# ---------------------------------------------------------------- 翻译成人话

PLAIN_PROMPT = """你在给一位**完全不懂证券**的普通人写今天的美股简报。他有闲钱想小额投资，但看不懂任何专业术语。

下面是几只股票的专业分析结论。请把它改写成他能看懂的中文。

## 硬性要求

1. **禁止出现未经解释的术语。** 不许直接写 MA20、均线多头排列、乖离率、RSI、MACD、支撑位、压力位、量能、筹码。
   如果某个判断确实基于这些，就用大白话说清楚背后的意思，比如"股价最近半个月一直在往下走，还没有回头的迹象"。

2. **每只股票写 2-4 句话**，必须说清楚三件事：
   - 结论是买、卖还是先别动
   - **为什么** —— 尤其要说清楚是因为什么消息、什么事件、还是单纯因为股价走势
   - 有多大把握 —— 如果分析里本身就有分歧或数据不足，如实说"这个判断不太有把握"

3. **不要编造。** 原文里没提到的消息、数字、事件，一律不许出现。原文说不确定的，你也要说不确定。

4. **不要给仓位建议**（不要说买多少、几成仓）。他还没开始建仓，只需要方向。

5. 开头写一段 2-3 句的「今天大盘什么情况」，如果原文信息不足以判断大盘，就直说信息不足。

6. 结尾写一句提醒：这些是 AI 基于公开信息的推理，不是经过验证的策略信号。

## 输出格式（严格遵守，用 Markdown）

## 今天市场
（2-3句话）

## 逐只看

### 🟢/🔴/🟡 公司名（代码）
（2-4句人话）

...（每只一段）

---
（结尾提醒）

## 原始分析

{payload}
"""


def to_plain_chinese(results: List[Dict[str, Any]]) -> Optional[str]:
    ok = [r for r in results if r["error"] is None and r["raw"]]
    if not ok:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        payload = "\n\n".join(
            f"===== {r['ticker']} （系统信号：{r['signal']}）=====\n{r['raw'][:6000]}"
            for r in ok
        )
        llm = ChatGoogleGenerativeAI(model=DEEP_MODEL, google_api_key=GOOGLE_API_KEY)
        resp = llm.invoke(PLAIN_PROMPT.format(payload=payload))
        return getattr(resp, "content", None) or str(resp)
    except Exception as exc:  # noqa: BLE001
        log(f"翻译步骤失败: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------- 邮件

def build_email(results: List[Dict[str, Any]], plain: Optional[str]) -> str:
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0, "ERROR": 0, "UNKNOWN": 0}
    for r in results:
        counts[r["signal"]] = counts.get(r["signal"], 0) + 1

    head = (
        f"<p style='font-size:15px'><b>{TODAY} 美股简报</b><br>"
        f"{len(results)} 只 &nbsp; 🟢买入 {counts['BUY']} &nbsp; 🟡观望 {counts['HOLD']} &nbsp; 🔴卖出 {counts['SELL']}"
    )
    if counts["ERROR"] or counts["UNKNOWN"]:
        head += f" &nbsp; ⚠️异常 {counts['ERROR'] + counts['UNKNOWN']}"
    head += "</p><hr>"

    if plain:
        body = markdown_to_html(plain)
    else:
        body = "<p>⚠️ 人话翻译这一步失败了，下面是原始信号，详情看 GitHub Actions 日志。</p><ul>"
        for r in results:
            body += f"<li><b>{r['ticker']}</b>: {SIGNAL_CN.get(r['signal'], r['signal'])}</li>"
        body += "</ul>"

    failed = [r for r in results if r["error"]]
    if failed:
        body += "<hr><p><b>⚠️ 以下股票本次分析失败，不代表没有信号，只是没跑出来：</b></p><ul>"
        for r in failed:
            body += f"<li><b>{r['ticker']}</b>: <code>{r['error'][:200]}</code></li>"
        body += "</ul>"

    tail = (
        "<hr><p style='color:#888;font-size:12px'>"
        f"引擎 TradingAgents · 模型 {DEEP_MODEL} · 生成于 "
        f"{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')} 美东<br>"
        "本邮件由自动化脚本生成，不构成投资建议。</p>"
    )
    return f"<div style='font-family:-apple-system,sans-serif;line-height:1.7;max-width:720px'>{head}{body}{tail}</div>"


def markdown_to_html(md: str) -> str:
    """极简 Markdown -> HTML，只处理标题、粗体、分隔线、段落。"""
    html: List[str] = []
    for line in md.split("\n"):
        s = line.rstrip()
        if not s:
            continue
        if s.startswith("### "):
            html.append(f"<h3 style='margin:18px 0 6px'>{s[4:]}</h3>")
        elif s.startswith("## "):
            html.append(f"<h2 style='margin:22px 0 8px;font-size:17px'>{s[3:]}</h2>")
        elif s.startswith("# "):
            html.append(f"<h2 style='margin:22px 0 8px'>{s[2:]}</h2>")
        elif s.strip() in ("---", "***"):
            html.append("<hr>")
        else:
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            html.append(f"<p style='margin:6px 0'>{s}</p>")
    return "".join(html)


def send_email(html: str) -> bool:
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        log("⚠️ 邮箱未配置，跳过发送")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{TODAY} 美股简报"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html", "utf-8"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, [a.strip() for a in EMAIL_TO.split(",") if a.strip()], msg.as_string())
        log(f"✅ 邮件已发送 -> {EMAIL_TO}")
        return True
    except smtplib.SMTPAuthenticationError:
        log("❌ 邮件发送失败：认证错误。检查 EMAIL_SENDER 和应用专用密码（16位、无空格）。")
    except Exception as exc:  # noqa: BLE001
        log(f"❌ 邮件发送失败: {type(exc).__name__}: {exc}")
    return False


# ---------------------------------------------------------------- main

def main() -> int:
    log(f"分析日期 {TODAY}（美东），股票 {TICKERS}")
    if not GOOGLE_API_KEY:
        log("❌ GOOGLE_API_KEY 未设置")
        return 1

    results = analyze_all(TICKERS)
    plain = to_plain_chinese(results)
    html = build_email(results, plain)

    os.makedirs("out", exist_ok=True)
    with open(f"out/brief_{TODAY}.html", "w", encoding="utf-8") as f:
        f.write(html)
    if plain:
        with open(f"out/brief_{TODAY}.md", "w", encoding="utf-8") as f:
            f.write(plain)
    log("报告已写入 out/")

    send_email(html)

    n_err = sum(1 for r in results if r["error"])
    log(f"完成：成功 {len(results) - n_err} / {len(results)}")
    # 全军覆没才算失败，部分失败仍返回 0（邮件里已如实说明）
    return 1 if n_err == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
