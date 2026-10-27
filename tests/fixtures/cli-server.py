#!/usr/bin/env python3
import http.server, json, os, pathlib, sys, time
root = pathlib.Path(os.environ['AUTARKY_FIXTURE_ROOT'])
if '--version' in sys.argv:
    print('version: 1234 (fixture)')
    sys.exit(0)
if '--list-devices' in sys.argv:
    print('Available devices:')
    sys.exit(0)
(root / 'launch-args.json').write_text(json.dumps(sys.argv[1:]))
(root / 'server-pid').write_text(str(os.getpid()))
if os.environ.get('AUTARKY_FIXTURE_MODE') == 'startup': time.sleep(60)
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def reply(self, status, body):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
    def do_GET(self): self.reply(200, {'status': 'ok'})
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with (root / 'requests.jsonl').open('a') as out:
            out.write(json.dumps({'cache_prompt': request['cache_prompt'], 'length': len(request['prompt'])}) + '\n')
        mode = os.environ.get('AUTARKY_FIXTURE_MODE', 'success')
        count = len((root / 'requests.jsonl').read_text().splitlines())
        if mode == 'cancel': time.sleep(60)
        if mode == 'gated_success':
            while not (root / 'release-response').exists(): time.sleep(0.01)
        if mode == 'error' or (mode == 'partial' and count > 1):
            self.reply(500, {'error': {'type': 'internal_error', 'message': 'private prompt and server error'}})
        elif mode == 'oom':
            self.reply(500, {'error': {'type': 'out_of_memory'}})
        else:
            self.reply(200, {'stop': True, 'truncated': False, 'tokens_predicted': 100, 'tokens_evaluated': 900,
                'timings': {'prompt_n': 900, 'predicted_n': 100, 'cache_n': 0, 'prompt_ms': 1000,
                    'predicted_ms': 2000, 'prompt_per_second': 900.0, 'predicted_per_second': 50.0}})
server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
print('server is listening on http://127.0.0.1:' + str(server.server_port), flush=True)
server.serve_forever()
