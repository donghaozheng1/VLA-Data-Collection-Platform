"""冒烟测试：遍历所有页面与接口，确认均可正常返回。

不需要启动服务（用 TestClient 直接打 ASGI）。
用法：python smoke_test.py
"""
from fastapi.testclient import TestClient

from main import app

PAGES = [
    "/",
    "/dashboard",
    "/semantics",
    "/robots",
    "/scenes",
    "/collect",
    "/data",
    "/viewer",
    "/preprocess",
    "/qc",
    "/annotation",
    "/reviews",
    "/export",
    "/models",
    "/training",
    "/interfaces",
    "/system",
    "/log",
    "/auth/login",
    "/auth/settings",
]

# 平台自身的 JSON 接口（/api/*）
PLATFORM_APIS = [
    "/api/overview",
    "/api/datasets",
    "/api/jobs",
    "/api/artifacts",
    "/api/qc/runs",
    "/api/annotations",
    "/api/repair/ops",
    "/api/reviews/summary",
    "/api/collect/sessions",
    "/api/training/experiments",
    "/api/training/evaluations",
    "/api/badcases",
    "/api/inference/health",
    "/api/inference/config",
    "/api/scenes/overview",
    "/api/robots/overview",
    "/api/splits/overview",
    "/api/storage/plan",
    "/api/training/compare/1",
    "/api/spatial/summary/1",
    "/api/spatial/1",
    "/api/dedup/report",
    "/api/annotators/roster",
    "/api/annotators/stats",
    "/api/annotators/work/current",
    "/api/storage/policy",
    "/api/training/envs",
    "/api/models",
]


def main() -> int:
    # 不自动跟随重定向：页面必须直接返回 200，重定向视为失败（否则会掩盖路由问题）
    client = TestClient(app, follow_redirects=False)
    failures = []

    print("===== 页面 =====")
    for path in PAGES:
        resp = client.get(path)
        ok = resp.status_code == 200
        if not ok:
            failures.append(f"GET {path} -> {resp.status_code}")
        print(f"[{'OK ' if ok else 'FAIL'}] {resp.status_code} GET {path} ({len(resp.content)} bytes)")

    print("\n===== 平台接口 =====")
    for path in PLATFORM_APIS:
        resp = client.get(path)
        ok = resp.status_code == 200
        if not ok:
            failures.append(f"GET {path} -> {resp.status_code}")
        body = resp.json() if ok else None
        size = len(body) if isinstance(body, list) else 1
        print(f"[{'OK ' if ok else 'FAIL'}] {resp.status_code} GET {path} ({size} 项)")

    print("\n===== 日志接口 =====")
    checks = [
        ("/log/api/logs", {"limit": 5}),
        ("/log/api/logs", {"level": "ERROR", "limit": 3}),
        ("/log/api/logs", {"module": "数据质检", "limit": 3}),
    ]
    for path, params in checks:
        resp = client.get(path, params=params)
        ok = resp.status_code == 200 and resp.json().get("code") == 0
        if not ok:
            failures.append(f"GET {path} {params} -> {resp.status_code}")
        body = resp.json() if resp.status_code == 200 else {}
        print(f"[{'OK ' if ok else 'FAIL'}] {resp.status_code} GET {path} {params} "
              f"(total={body.get('total')}, modules={len(body.get('modules', []))})")

    print("\n===== 登录流程 =====")
    bad = client.post("/auth/login", data={"username": "admin", "password": "wrong"},
                      follow_redirects=False)
    ok_bad = bad.status_code == 401
    if not ok_bad:
        failures.append(f"错误密码应返回 401，实际 {bad.status_code}")
    print(f"[{'OK ' if ok_bad else 'FAIL'}] 错误密码 -> {bad.status_code}")

    good = client.post("/auth/login", data={"username": "admin", "password": "admin123"},
                       follow_redirects=False)
    ok_good = good.status_code == 303 and good.headers.get("location") == "/dashboard"
    if not ok_good:
        failures.append(f"正确密码应 303 跳转 /dashboard，实际 {good.status_code}")
    print(f"[{'OK ' if ok_good else 'FAIL'}] 正确密码 -> {good.status_code} "
          f"location={good.headers.get('location')}")

    dash = client.get("/dashboard")
    ok_dash = dash.status_code == 200 and "admin" in dash.text
    if not ok_dash:
        failures.append("登录后 /dashboard 未显示当前用户")
    print(f"[{'OK ' if ok_dash else 'FAIL'}] 登录后 /dashboard 显示用户名 -> {ok_dash}")

    out = client.get("/auth/logout", follow_redirects=False)
    ok_out = out.status_code == 303
    if not ok_out:
        failures.append(f"注销应 303，实际 {out.status_code}")
    print(f"[{'OK ' if ok_out else 'FAIL'}] 注销 -> {out.status_code}")

    print("\n===== 改密流程 =====")
    client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    wrong = client.post("/auth/settings/change-password",
                        data={"old_password": "nope", "new_password": "newpass123"})
    ok_wrong = wrong.status_code == 400 and "当前密码不正确" in wrong.text
    if not ok_wrong:
        failures.append("旧密码错误应返回 400")
    print(f"[{'OK ' if ok_wrong else 'FAIL'}] 旧密码错误 -> {wrong.status_code}")

    short = client.post("/auth/settings/change-password",
                        data={"old_password": "admin123", "new_password": "123"})
    ok_short = short.status_code == 400 and "至少 6 位" in short.text
    if not ok_short:
        failures.append("新密码过短应返回 400")
    print(f"[{'OK ' if ok_short else 'FAIL'}] 新密码过短 -> {short.status_code}")

    done = client.post("/auth/settings/change-password",
                       data={"old_password": "admin123", "new_password": "newpass123"})
    ok_done = done.status_code == 200 and "密码修改成功" in done.text
    if not ok_done:
        failures.append("改密成功应返回 200")
    print(f"[{'OK ' if ok_done else 'FAIL'}] 改密成功 -> {done.status_code}")

    # 还原密码，避免影响后续手工登录
    client.post("/auth/settings/change-password",
                data={"old_password": "newpass123", "new_password": "admin123"})

    # 前端脚本加载顺序：platform.js（定义 API/runJob 等）必须先于页面内联脚本，
    # 否则内联脚本第一行就 ReferenceError，页面永远加载不出数据。
    # 这类问题 HTTP 状态码看不出来，只能靠顺序断言兜住。
    print("\n===== 前端脚本加载顺序 =====")
    for path, marker in [("/dashboard", "loadOverview"), ("/collect", "doScan"),
                         ("/data", "loadDatasets"), ("/preprocess", "loadArtifacts"),
                         ("/qc", "loadRuns"), ("/annotation", "loadEpisodes"),
                         ("/export", "doExport"), ("/reviews", "loadAgreement"), ("/training", "loadExperiments"), ("/models", "loadAll"), ("/scenes", "toggleVariants"), ("/robots", "doProbe"), ("/export", "loadSplits"), ("/system", "loadPlan"), ("/annotation", "openCanvas"), ("/data", "loadDedup"), ("/annotation", "workStart"), ("/system", "savePolicy"), ("/training", "startTraining")]:
        text = client.get(path).text
        i_lib = text.find("/static/js/platform.js")
        i_inline = text.find(marker)
        ok = 0 <= i_lib < i_inline
        if not ok:
            failures.append(f"{path}: platform.js 未先于内联脚本加载（{i_lib} vs {i_inline}）")
        print(f"[{'OK ' if ok else 'FAIL'}] {path} platform.js@{i_lib} < 内联脚本@{i_inline}")

    print("\n===== 版本与血缘 =====")
    datasets = client.get("/api/datasets").json()
    if datasets:
        ds_id = datasets[0]["id"]
        for path in [f"/api/datasets/{ds_id}/versions", f"/api/datasets/{ds_id}/lineage",
                     f"/api/datasets/{ds_id}/derived"]:
            resp = client.get(path)
            ok = resp.status_code == 200
            if not ok:
                failures.append(f"GET {path} -> {resp.status_code}")
            print(f"[{'OK ' if ok else 'FAIL'}] {resp.status_code} GET {path}")
        versions = client.get(f"/api/datasets/{ds_id}/versions").json()["versions"]
        ok = bool(versions) and all(v.get("content_hash") for v in versions)
        if not ok:
            failures.append("版本历史缺内容哈希")
        print(f"[{'OK ' if ok else 'FAIL'}] 版本历史：{len(versions)} 个版本，均带内容哈希")
        chain = client.get(f"/api/datasets/{ds_id}/lineage").json()["chain"]
        ok = bool(chain)
        if not ok:
            failures.append("血缘链为空")
        print(f"[{'OK ' if ok else 'FAIL'}] 血缘链：{' ← '.join(str(n['id']) for n in chain)}")
    else:
        print("[跳过] 库里还没有数据集")

    print("\n===== 结果 =====")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for item in failures:
            print("  -", item)
        return 1
    print(f"全部通过：{len(PAGES)} 个页面 + {len(PLATFORM_APIS)} 个平台接口 + "
          f"3 个日志接口 + 登录/注销/改密流程")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
