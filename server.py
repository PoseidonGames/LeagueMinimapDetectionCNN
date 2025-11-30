from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS so iPad can access the JSON
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

    def log_message(self, format, *args):
        # Suppress console spam (optional)
        pass

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    PORT = 8000
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("🚀 League Gank Detector Server")
    print("=" * 60)
    print(f"\n📱 On your iPad, open Safari and go to:")
    print(f"\n   http://{local_ip}:{PORT}/gank_detector.html")
    print(f"\n💻 Local access: http://localhost:{PORT}/gank_detector.html")
    print(f"\n🛑 Press Ctrl+C to stop the server\n")
    print("=" * 60)
    
    server = HTTPServer(('0.0.0.0', PORT), CORSRequestHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        server.shutdown()
