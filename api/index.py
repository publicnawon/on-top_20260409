import os


def _html(target_url: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>On-Top Redirect</title>
    <style>
      body {{
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f5f7fb;
        color: #111827;
        display: grid;
        place-items: center;
        min-height: 100vh;
      }}
      .card {{
        max-width: 680px;
        background: white;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        padding: 28px;
      }}
      a {{
        color: #0b57d0;
        text-decoration: none;
        font-weight: 600;
      }}
      code {{
        background: #f3f4f6;
        padding: 2px 6px;
        border-radius: 6px;
      }}
    </style>
  </head>
  <body>
    <main class="card">
      <h1>On-Top 서비스 주소 안내</h1>
      <p>현재 Vercel 환경에서는 Streamlit 앱을 직접 실행하지 않습니다.</p>
      <p>아래 링크로 이동해 앱을 사용하세요:</p>
      <p><a href="{target_url}">{target_url}</a></p>
      <p>Vercel 프로젝트 환경변수 <code>TARGET_APP_URL</code>를 설정하면 자동 리다이렉트됩니다.</p>
    </main>
  </body>
</html>"""


def handler(request):
    target_url = os.getenv("TARGET_APP_URL", "").strip()
    if target_url:
        return {
            "statusCode": 307,
            "headers": {"Location": target_url, "Cache-Control": "no-store"},
            "body": "",
        }

    fallback = "https://github.com/publicnawon/on-top_20260409"
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"},
        "body": _html(fallback),
    }
