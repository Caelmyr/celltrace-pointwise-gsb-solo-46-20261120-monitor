import os

from waitress import serve

from celltrace.web import build_app


if __name__ == "__main__":
    serve(
        build_app(os.environ.get("CELLTRACE_DATA", ".celltrace")),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        threads=4,
        channel_timeout=35,
        max_request_body_size=68 * 1024 * 1024,
    )
