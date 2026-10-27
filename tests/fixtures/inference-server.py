#!/usr/bin/env python3
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
args=sys.argv[1:]
def arg(name, default=''):
    for i, token in enumerate(args):
        key, sep, value = token.partition('=')
        if key.startswith('--'): key = key.replace('_', '-')
        if key == name: return value if sep else args[i+1]
    return default
folder=arg('--tags')
mode=arg('--reasoning-budget-message')
with open(folder+'/pid','w') as f: f.write(str(os.getpid()))
if mode=='startup-hang':
    time.sleep(120)
    sys.exit(0)
if mode=='exit': sys.exit(3)
count=0
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def do_GET(self):
        if mode=='health-hang': time.sleep(120)
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        global count
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert self.path=='/completion'
        assert body['cache_prompt'] is False
        assert body['stream'] is False
        key=arg('--api-key')
        if not key and arg('--api-key-file'):
            with open(arg('--api-key-file')) as f: key=f.readline().strip()
        if key and self.headers.get('Authorization') != 'Bearer '+key:
            self.send_response(401); self.end_headers(); return
        count += 1
        if mode=='request-hang': time.sleep(120)
        with open(folder+'/requests','w') as f: f.write(str(count))
        response={'stop':True,'truncated':False,'tokens_predicted':100,'tokens_evaluated':25,
          'timings':{'cache_n':0,'prompt_n':25,'prompt_ms':2.0,'prompt_per_second':float(count*100),
          'predicted_n':100,'predicted_ms':4.0,'predicted_per_second':float(count*10)}}
        status=200
        if mode=='partial-second' and count==2: del response['timings']['prompt_per_second']
        if mode=='stale-count': response['tokens_predicted']=99
        if mode=='error': status=500; response={'error':{'type':'server_error','message':'unrelated failure'}}
        if mode=='oom': status=500; response={'error':{'type':'out_of_memory','message':'allocation'}}
        self.send_response(status); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(json.dumps(response).encode())
server=HTTPServer(('127.0.0.1',0),Handler)
print('main: server is listening on http://127.0.0.1:'+str(server.server_port),flush=True)
server.serve_forever()
