#!/usr/bin/env python3
"""
Simple HTTP server to serve the reports directory (penalties and costs).
Run this script from the reports directory to make the data accessible to the React app.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    # Change to the reports directory (parent of the script location)
    os.chdir(Path(__file__).parent)

    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Serving reports data at http://localhost:{PORT}")
        print(f"Serving from directory: {os.getcwd()}")
        print(f"  - Penalties: /penalties/penalties_summary.json")
        print(f"  - Costs: /costs/costs_by_day.csv")
        print("\nPress Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
