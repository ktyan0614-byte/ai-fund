# -*- coding: utf-8 -*-
"""把最新週報寄到信箱(Gmail)。

需要兩個環境變數(在 GitHub repo 的 Settings → Secrets 設定):
  MAIL_USERNAME      Gmail 帳號(也是收件人,寄給自己)
  MAIL_APP_PASSWORD  Gmail 應用程式密碼(不是登入密碼)
未設定時直接跳過,不影響主流程。
"""
import glob
import os
import smtplib
import sys
from email.mime.text import MIMEText

user = os.environ.get("MAIL_USERNAME", "").strip()
pw = os.environ.get("MAIL_APP_PASSWORD", "").strip()
if not user or not pw:
    print("未設定 MAIL_USERNAME / MAIL_APP_PASSWORD,跳過寄送")
    sys.exit(0)

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reports = sorted(glob.glob(os.path.join(base, "reports", "*.md")))
if not reports:
    print("找不到週報,跳過寄送")
    sys.exit(0)
latest = reports[-1]
date = os.path.basename(latest).replace(".md", "")
text = open(latest, encoding="utf-8").read()

import markdown
body = markdown.markdown(text, extensions=["tables"])
html = f"""<html><head><style>
body {{ font-family: -apple-system, 'Microsoft JhengHei', sans-serif;
       max-width: 720px; margin: auto; padding: 12px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
th, td {{ border: 1px solid #ccc; padding: 5px 8px; font-size: 14px; }}
th {{ background: #eaf0f8; }}
h1 {{ font-size: 20px; border-bottom: 2px solid #1a3c6e; padding-bottom: 4px; }}
h2, h3 {{ font-size: 16px; color: #1a3c6e; }}
blockquote {{ color: #777; font-size: 12px; border-left: 3px solid #ccc;
              margin: 8px 0; padding-left: 10px; }}
</style></head><body>{body}</body></html>"""

msg = MIMEText(html, "html", "utf-8")
msg["Subject"] = f"AI 投資公司週報 {date}"
msg["From"] = user
msg["To"] = user

with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
    s.login(user, pw)
    s.send_message(msg)
print(f"週報 {date} 已寄出至 {user}")
