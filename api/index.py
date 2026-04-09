import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse


def _html(target_url: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>On-Top Redirect</title>
  </head>
  <body style="font-family: sans-serif; padding: 24px; line-height: 1.5;">
    <h1>On-Top 서비스 주소 안내</h1>
    <p>현재 Vercel에서는 Streamlit 앱을 직접 호스팅하지 않습니다.</p>
    <p>아래 링크로 이동해 앱을 사용하세요:</p>
    <p><a href="{target_url}">{target_url}</a></p>
    <p>Vercel 환경변수 <code>TARGET_APP_URL</code>를 설정하면 자동 리다이렉트됩니다.</p>
  </body>
</html>"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        target = os.getenv("TARGET_APP_URL", "").strip()
        if target:
            parsed = urlparse(target)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                self.send_response(307)
                self.send_header("Location", target)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return

        fallback = "https://github.com/publicnawon/on-top_20260409"
        body = _html(fallback).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
