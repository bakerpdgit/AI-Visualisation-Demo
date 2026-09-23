"""Tiny local web server for the demo (no internet needed).

Browsers only allow the webcam and ES modules on pages served over
http://localhost or https://, so double-clicking index.html won't work.
This serves the project folder at http://localhost:8000/ and opens it.

Usage:  python tools/serve.py [port]
"""
import http.server
import os
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Force the right content types. (On some Windows PCs the registry says .js is
# "text/plain", which makes browsers refuse to run JavaScript modules.)
TYPES = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.json': 'application/json', '.bin': 'application/octet-stream', '.md': 'text/plain; charset=utf-8',
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def guess_type(self, path):
        return TYPES.get(os.path.splitext(path)[1].lower()) or super().guess_type(path)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, *args):
        pass  # keep the console quiet


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    for p in range(port, port + 20):
        try:
            httpd = http.server.ThreadingHTTPServer(('127.0.0.1', p), Handler)
            break
        except OSError:
            continue
    else:
        sys.exit('Could not find a free port.')
    url = f'http://localhost:{httpd.server_address[1]}/'
    print(f'\n  Cat or Alligator demo is running at {url}')
    print('  Leave this window open while you use the demo. Close it (or press Ctrl+C) to stop.\n')
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
