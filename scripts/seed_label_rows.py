#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, pathlib, shutil, time

EN_ROWS = [
  {"id":"en-0001","text":"The report is due tomorrow.","lang":"en","gold_act":"statement"},
  {"id":"en-0002","text":"This dataset contains 10,000 entries.","lang":"en","gold_act":"statement"},
  {"id":"en-0003","text":"The server is back online.","lang":"en","gold_act":"statement"},
  {"id":"en-0004","text":"Our meeting starts at 9 a.m.","lang":"en","gold_act":"statement"},
  {"id":"en-0005","text":"This approach reduces memory usage.","lang":"en","gold_act":"statement"},
  {"id":"en-0006","text":"The weather looks clear today.","lang":"en","gold_act":"statement"},
  {"id":"en-0007","text":"The paper draft needs editing.","lang":"en","gold_act":"statement"},
  {"id":"en-0008","text":"Latency increased after the update.","lang":"en","gold_act":"statement"},
  {"id":"en-0009","text":"Your package arrived this morning.","lang":"en","gold_act":"statement"},

  {"id":"en-0010","text":"What time is the submission deadline?","lang":"en","gold_act":"question"},
  {"id":"en-0011","text":"Where did you save the file?","lang":"en","gold_act":"question"},
  {"id":"en-0012","text":"How does the router decide to defer?","lang":"en","gold_act":"question"},
  {"id":"en-0013","text":"Who approved the budget?","lang":"en","gold_act":"question"},
  {"id":"en-0014","text":"Why is the build failing?","lang":"en","gold_act":"question"},
  {"id":"en-0015","text":"Which version should we install?","lang":"en","gold_act":"question"},
  {"id":"en-0016","text":"When will the results be ready?","lang":"en","gold_act":"question"},
  {"id":"en-0017","text":"Is the API rate-limited?","lang":"en","gold_act":"question"},
  {"id":"en-0018","text":"Can we reproduce the error?","lang":"en","gold_act":"question"},

  {"id":"en-0019","text":"Please send me the latest logs.","lang":"en","gold_act":"request"},
  {"id":"en-0020","text":"Could you review my PR?","lang":"en","gold_act":"request"},
  {"id":"en-0021","text":"Add a citation to section three.","lang":"en","gold_act":"request"},
  {"id":"en-0022","text":"Update the README before release.","lang":"en","gold_act":"request"},
  {"id":"en-0023","text":"Share the slides after the talk.","lang":"en","gold_act":"request"},
  {"id":"en-0024","text":"Please restart the service.","lang":"en","gold_act":"request"},
  {"id":"en-0025","text":"Draft a summary in one paragraph.","lang":"en","gold_act":"request"},
  {"id":"en-0026","text":"Upload the dataset to Drive.","lang":"en","gold_act":"request"},
  {"id":"en-0027","text":"Let me know if you need help.","lang":"en","gold_act":"request"},

  {"id":"en-0028","text":"I will fix the bug tonight.","lang":"en","gold_act":"promise"},
  {"id":"en-0029","text":"I'll share the slides by noon.","lang":"en","gold_act":"promise"},
  {"id":"en-0030","text":"We will deliver the revision tomorrow.","lang":"en","gold_act":"promise"},
  {"id":"en-0031","text":"I promise to write the tests.","lang":"en","gold_act":"promise"},
  {"id":"en-0032","text":"We'll follow up after the meeting.","lang":"en","gold_act":"promise"},
  {"id":"en-0033","text":"I will send the invoice today.","lang":"en","gold_act":"promise"},
  {"id":"en-0034","text":"I'll handle the deployment.","lang":"en","gold_act":"promise"},
  {"id":"en-0035","text":"We will not change the API.","lang":"en","gold_act":"promise"},

  {"id":"en-0036","text":"Thanks a ton for your help!","lang":"en","gold_act":"expressive"},
  {"id":"en-0037","text":"I'm sorry for the delay.","lang":"en","gold_act":"expressive"},
  {"id":"en-0038","text":"Great job on the experiment!","lang":"en","gold_act":"expressive"},
  {"id":"en-0039","text":"This is so frustrating.","lang":"en","gold_act":"expressive"},
  {"id":"en-0040","text":"I appreciate your quick response.","lang":"en","gold_act":"expressive"},
  {"id":"en-0041","text":"That result is amazing!","lang":"en","gold_act":"expressive"},
  {"id":"en-0042","text":"What a relief!","lang":"en","gold_act":"expressive"},

  {"id":"en-0043","text":"I hereby submit the final report.","lang":"en","gold_act":"declaration"},
  {"id":"en-0044","text":"We declare the meeting open.","lang":"en","gold_act":"declaration"},
  {"id":"en-0045","text":"I pronounce the task complete.","lang":"en","gold_act":"declaration"},
  {"id":"en-0046","text":"Access is granted to the beta group.","lang":"en","gold_act":"declaration"},
  {"id":"en-0047","text":"Your account is now suspended.","lang":"en","gold_act":"declaration"},
  {"id":"en-0048","text":"I appoint you as note-taker.","lang":"en","gold_act":"declaration"},
  {"id":"en-0049","text":"The session is adjourned.","lang":"en","gold_act":"declaration"},
  {"id":"en-0050","text":"This document is now obsolete.","lang":"en","gold_act":"declaration"}
]

ZH_ROWS = [
  {"id":"zh-0001","text":"报告明天截止。","lang":"zh","gold_act":"statement"},
  {"id":"zh-0002","text":"服务器已经恢复正常。","lang":"zh","gold_act":"statement"},
  {"id":"zh-0003","text":"这个数据集包含一万条记录。","lang":"zh","gold_act":"statement"},
  {"id":"zh-0004","text":"今天的天气很好。","lang":"zh","gold_act":"statement"},

  {"id":"zh-0005","text":"会议几点开始？","lang":"zh","gold_act":"question"},
  {"id":"zh-0006","text":"你把文件存在哪里了？","lang":"zh","gold_act":"question"},
  {"id":"zh-0007","text":"为什么构建一直失败？","lang":"zh","gold_act":"question"},
  {"id":"zh-0008","text":"我们什么时候可以看到结果？","lang":"zh","gold_act":"question"},
  {"id":"zh-0009","text":"这个接口有限流吗？","lang":"zh","gold_act":"question"},

  {"id":"zh-0010","text":"请把最新的日志发给我。","lang":"zh","gold_act":"request"},
  {"id":"zh-0011","text":"麻烦你审核一下我的PR。","lang":"zh","gold_act":"request"},
  {"id":"zh-0012","text":"发布前请更新README。","lang":"zh","gold_act":"request"},
  {"id":"zh-0013","text":"请重启一下服务。","lang":"zh","gold_act":"request"},

  {"id":"zh-0014","text":"我今晚会修这个bug。","lang":"zh","gold_act":"promise"},
  {"id":"zh-0015","text":"我们明天交付修订版，保证。","lang":"zh","gold_act":"promise"},
  {"id":"zh-0016","text":"我一定把测试补上。","lang":"zh","gold_act":"promise"},
  {"id":"zh-0017","text":"我会在今天发出发票。","lang":"zh","gold_act":"promise"},

  {"id":"zh-0018","text":"非常感谢你的帮助！","lang":"zh","gold_act":"expressive"},
  {"id":"zh-0019","text":"抱歉让你久等了。","lang":"zh","gold_act":"expressive"},
  {"id":"zh-0020","text":"这个结果太棒了！","lang":"zh","gold_act":"expressive"},
  {"id":"zh-0021","text":"真让人头大。","lang":"zh","gold_act":"expressive"},

  {"id":"zh-0022","text":"特此提交最终报告。","lang":"zh","gold_act":"declaration"},
  {"id":"zh-0023","text":"现在宣布会议开始。","lang":"zh","gold_act":"declaration"},
  {"id":"zh-0024","text":"你已获得测试资格。","lang":"zh","gold_act":"declaration"},
  {"id":"zh-0025","text":"本文件自即日起作废。","lang":"zh","gold_act":"declaration"}
]

def backup(path: str):
    p = pathlib.Path(path)
    if p.exists():
        b = p.with_suffix(p.suffix + f".bak.{int(time.time())}")
        shutil.copy2(p, b)
        print(f"[backup] {p} -> {b}")

def write_tsv(path: str, rows):
    header = ["id","text","lang","gold_act"]
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ok] wrote {p} ({len(rows)} rows)")

if __name__ == "__main__":
    backup("data/acts/label_en.tsv")
    backup("data/acts/label_zh.tsv")
    write_tsv("data/acts/label_en.tsv", EN_ROWS)
    write_tsv("data/acts/label_zh.tsv", ZH_ROWS)
