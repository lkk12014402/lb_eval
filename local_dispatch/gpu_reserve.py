#!/usr/bin/env python3
"""
NVIDIA GPU 预约系统自动填表脚本
目标: http://10.112.229.207:8688

用法示例:
  # 列出所有服务器和 GPU 状态
  python3 gpu_reserve.py list

  # 立即预约 4090D 服务器的 2 块空闲 GPU，1 小时
  python3 gpu_reserve.py reserve --user zhangsan --server 4090D --hours 1 --gpus auto:2

  # 指定 GPU 编号 0,1,2,3，预约 H20 服务器（按 host 关键字匹配），从 14:30 开始 3 小时
  python3 gpu_reserve.py reserve --user zhangsan --server 118.195 --hours 3 \
      --gpus 0,1,2,3 --start 14:30

  # 查询某用户当前预约
  python3 gpu_reserve.py mine --user zhangsan
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib import request, parse, error

# Reservation API base URL. Override with the GPU_RESERVE_BASE_URL env var;
# defaults to the current on-prem reservation server.
BASE_URL = os.environ.get("GPU_RESERVE_BASE_URL", "")
CST = timezone(timedelta(hours=8))


def _opener():
    # 绕过环境里的 HTTP 代理（内网 IP 走代理会失败）
    handler = request.ProxyHandler({})
    return request.build_opener(handler)


def api(path, method="GET", body=None):
    url = BASE_URL.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with _opener().open(req, timeout=15) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(msg).get("detail", msg)
        except Exception:
            detail = msg
        raise SystemExit(f"HTTP {e.code} {e.reason}: {detail}")
    except error.URLError as e:
        raise SystemExit(f"网络错误: {e.reason}")


def find_server(servers, keyword):
    kw = keyword.lower()
    matches = [s for s in servers if kw in s["name"].lower() or kw in s["host"].lower() or kw == s["id"]]
    if not matches:
        raise SystemExit(f"未找到匹配 '{keyword}' 的服务器。可用服务器:\n" +
                         "\n".join(f"  - {s['name']}  ({s['host']})" for s in servers))
    if len(matches) > 1:
        raise SystemExit(f"关键字 '{keyword}' 匹配多台服务器，请更具体:\n" +
                         "\n".join(f"  - {s['name']}  ({s['host']})" for s in matches))
    return matches[0]


def parse_start(start_str):
    """把 'HH:MM' 转成今天的 ISO 时间（带 +08:00）。空值表示立即开始。"""
    if not start_str:
        return None
    try:
        hh, mm = start_str.split(":")
        hh, mm = int(hh), int(mm)
    except ValueError:
        raise SystemExit("--start 格式应为 HH:MM，例如 14:30")
    today = datetime.now(CST).date()
    dt = datetime(today.year, today.month, today.day, hh, mm, tzinfo=CST)
    return dt.isoformat(timespec="seconds")


def has_conflict(gpu, start_iso, hours):
    if start_iso is None:
        start_ms = datetime.now(CST).timestamp() * 1000
    else:
        start_ms = datetime.fromisoformat(start_iso).timestamp() * 1000
    end_ms = start_ms + hours * 3600 * 1000
    for slot in gpu.get("today_schedule") or []:
        s = datetime.fromisoformat(slot["start_time"]).timestamp() * 1000
        e = datetime.fromisoformat(slot["end_time"]).timestamp() * 1000
        if s < end_ms and e > start_ms:
            return slot
    return None


def resolve_gpus(server, gpus_arg, start_iso, hours):
    """gpus_arg: 'auto:N' 或 '0,1,2'"""
    free = [g for g in server["gpus"] if not has_conflict(g, start_iso, hours)]
    if gpus_arg.startswith("auto"):
        n = int(gpus_arg.split(":")[1]) if ":" in gpus_arg else 1
        if len(free) < n:
            raise SystemExit(f"该时段只有 {len(free)} 块空闲 GPU，无法满足 {n} 块的需求。"
                             f"空闲: {[g['device_id'] for g in free]}")
        return [g["device_id"] for g in free[:n]]
    # 显式列表
    ids = [int(x) for x in gpus_arg.split(",") if x.strip() != ""]
    free_ids = {g["device_id"] for g in free}
    bad = [i for i in ids if i not in free_ids]
    if bad:
        raise SystemExit(f"GPU {bad} 在该时段有冲突，无法预约。空闲: {sorted(free_ids)}")
    return ids


def cmd_list(_args):
    servers = api("/api/servers")
    for s in servers:
        print(f"\n=== {s['name']}  ({s['host']}) ===")
        print(f"    id: {s['id']}")
        for g in s["gpus"]:
            tag = "空闲" if g["status"] == "available" else f"占用 by {g['current_user']} → {g['current_end']}"
            upcoming = [u for u in (g.get("today_schedule") or []) if not u.get("is_current")]
            extra = ""
            if upcoming:
                extra = "  即将占用: " + "; ".join(
                    f"{u['username']} {u['start_time'][11:16]}-{u['end_time'][11:16]}"
                    for u in upcoming[:3])
            print(f"  GPU {g['device_id']:>2}  {g['model']:<8}  {tag}{extra}")


def cmd_reserve(args):
    servers = api("/api/servers")
    server = find_server(servers, args.server)
    start_iso = parse_start(args.start)
    gpu_ids = resolve_gpus(server, args.gpus, start_iso, args.hours)

    payload = {
        "username": args.user,
        "server_id": server["id"],
        "gpu_ids": gpu_ids,
        "start_time": start_iso,   # None 表示立即开始
        "duration_hours": args.hours,
    }
    print("提交预约:")
    print(f"  用户:    {args.user}")
    print(f"  服务器:  {server['name']}  ({server['host']})")
    print(f"  GPU:     {gpu_ids}")
    print(f"  开始:    {start_iso or '立即'}")
    print(f"  时长:    {args.hours} 小时")
    if args.dry_run:
        print("[dry-run] 未实际提交。")
        print("payload =", json.dumps(payload, ensure_ascii=False))
        return
    resp = api("/api/reservations", method="POST", body=payload)
    print("✓ 预约成功")
    if resp:
        print(json.dumps(resp, ensure_ascii=False, indent=2))


def cmd_mine(args):
    qs = parse.urlencode({"username": args.user})
    items = api(f"/api/reservations?{qs}") or []
    if not items:
        print(f"用户 {args.user} 没有活跃预约。")
        return
    for r in items:
        print(f"- {r.get('server_name') or r.get('server_id')}  GPU {r['gpu_ids']}  "
              f"{r['start_time']} → {r['end_time']}")


def main():
    p = argparse.ArgumentParser(description="NVIDIA GPU 预约系统自动填表")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出所有服务器和 GPU 状态").set_defaults(func=cmd_list)

    r = sub.add_parser("reserve", help="提交一个预约")
    r.add_argument("--user", required=True, help="用户名 / 工号")
    r.add_argument("--server", required=True, help="服务器关键字（匹配 name 或 host，例如 4090D / H20 / 118.195）")
    r.add_argument("--hours", type=int, required=True, choices=[1, 2, 3, 4], help="使用时长（1-4 小时）")
    r.add_argument("--gpus", required=True,
                   help="GPU 编号，逗号分隔 (如 0,1,2)；或 auto:N 自动挑选 N 块空闲 GPU")
    r.add_argument("--start", default=None,
                   help="开始时间 HH:MM (今天，CST)；不指定则立即开始")
    r.add_argument("--dry-run", action="store_true", help="只打印 payload，不真正提交")
    r.set_defaults(func=cmd_reserve)

    m = sub.add_parser("mine", help="查询某用户的活跃预约")
    m.add_argument("--user", required=True)
    m.set_defaults(func=cmd_mine)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
