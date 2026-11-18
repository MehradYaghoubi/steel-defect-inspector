import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from steel_inspector.data import CLASSES
from steel_inspector.inference import annotate, load_onnx, predict

MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def inspect_image(runtime, content, *, threshold):
    with Image.open(BytesIO(content)) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("Images must be 16 megapixels or smaller.")
        image = ImageOps.exif_transpose(source).convert("RGB")
    boxes, scores, labels = predict(runtime, image, threshold=threshold)
    detections = [
        {"type": CLASSES[label].replace("_", " ").capitalize(), "score": float(score), "box": box.tolist()}
        for label, score, box in zip(labels, scores, boxes)
    ]
    captions = [f'{row["type"]} {row["score"]:.0%}' for row in detections]
    preview = BytesIO()
    annotate(image, boxes, labels, captions).save(preview, format="PNG")
    return {
        "detections": detections,
        "overlay": f"data:image/png;base64,{base64.b64encode(preview.getvalue()).decode('ascii')}",
    }


class DemoHandler(BaseHTTPRequestHandler):
    timeout = 15

    def respond(self, status, media_type, body):
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def respond_json(self, status, payload):
        self.respond(status, "application/json", json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.respond(HTTPStatus.OK, "text/html; charset=utf-8", self.server.page)

    def do_POST(self):
        host = self.headers.get("Host")
        port = self.server.server_port
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.respond_json(HTTPStatus.FORBIDDEN, {"error": "Open the demo from its local URL."})
            return
        if self.headers.get("Origin") not in (None, f"http://{host}"):
            self.respond_json(HTTPStatus.FORBIDDEN, {"error": "Open the demo from its local URL."})
            return
        target = urlsplit(self.path)
        if target.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            threshold = float(parse_qs(target.query).get("threshold", ["0.25"])[0])
            if not 0 < threshold <= 1:
                raise ValueError
        except ValueError:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid image length or confidence threshold."})
            return
        if not 0 < length <= MAX_FILE_BYTES:
            self.respond_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Choose an image under 15 MB."})
            return
        content = self.rfile.read(length)
        try:
            result = inspect_image(self.server.runtime, content, threshold=threshold)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "Choose a readable image up to 16 megapixels."})
            return
        self.respond_json(HTTPStatus.OK, result)


def main():
    parser = argparse.ArgumentParser(description="Run the local CPU demo")
    parser.add_argument("--model", type=Path, default=Path("runs/detector.onnx"))
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Detector {args.model} is missing. Export a checkpoint and pass it to --model.")
    runtime = load_onnx(args.model)
    with ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler) as server:
        server.page = Path(__file__).with_name("demo.html").read_bytes()
        server.runtime = runtime
        print(f"Open http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
