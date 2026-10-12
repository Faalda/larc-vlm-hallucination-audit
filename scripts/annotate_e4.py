"""Serve the blinded web interface for the 200-item label audit.

Blinding protocol: raters see only the image and the
flagged category. No caption, no original CHAIR label, no computed feature.
Role is A, B, or C (C = third-rater adjudication, normally only given the
subset where A and B disagree -- pass --ids-file to restrict to those).

Usage:
    python scripts/annotate_e4.py --role A --port 8765
    python scripts/annotate_e4.py --role B --port 8766
    python scripts/annotate_e4.py --role C --port 8767 --ids-file results/e4_label_audit/disagreements.txt

Saves one JSON object per line to results/e4_label_audit/judgments_<role>.jsonl,
atomically, as each item is labeled. analyze_e4_label_audit.py reads these.
"""
import argparse
import json
import mimetypes
import os
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parents[1]
OUT_DIR = HERE / "results" / "e4_label_audit"
MANIFEST = OUT_DIR / "manifest_private.json"
IMAGES_DIR = HERE / "data" / "images"
VALID_ROLES = {"A", "B", "C"}
VALID_LABELS = {"VISIBLE", "NOT_VISIBLE", "AMBIGUOUS"}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E4 Label Audit</title>
<style>
  :root { --ink:#18201d; --paper:#f4f1e8; --panel:#fffdf7; --line:#c9c4b5; --green:#166534; --red:#a12b2b; --amber:#8a5a00; }
  * { box-sizing:border-box; }
  body { margin:0; color:var(--ink); background:var(--paper); font-family:Georgia,"Times New Roman",serif; }
  header { min-height:54px; padding:12px 24px; color:white; background:#22352e; display:flex; align-items:center; justify-content:space-between; gap:16px; }
  header strong { font-size:19px; }
  #progress { font-family:"Courier New",monospace; }
  main { width:min(1100px,100%); margin:auto; padding:20px; display:grid; grid-template-columns:minmax(0,1.4fr) minmax(260px,.6fr); gap:20px; }
  .image-wrap { height:calc(100vh - 96px); min-height:480px; display:flex; align-items:center; justify-content:center; background:#171a18; border:1px solid #111; }
  #image { width:100%; height:100%; object-fit:contain; }
  .panel { padding:20px; background:var(--panel); border:1px solid var(--line); box-shadow:4px 4px 0 #d8d2c3; }
  .target { margin:8px 0 16px; padding:12px; border-left:6px solid #b9382f; background:#f4e3d9; font-size:26px; font-weight:bold; }
  .rubric { padding:10px; background:#eeece4; font-size:13px; line-height:1.4; margin-bottom:10px; }
  .labels { display:grid; gap:8px; }
  button { min-height:46px; border:1px solid #6e746f; background:white; color:var(--ink); font:700 15px Georgia,serif; cursor:pointer; }
  button[data-label="VISIBLE"].selected { color:white; background:var(--green); }
  button[data-label="NOT_VISIBLE"].selected { color:white; background:var(--red); }
  button[data-label="AMBIGUOUS"].selected { color:white; background:var(--amber); }
  .actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:16px; }
  #save { color:white; background:#22352e; }
  #status { min-height:22px; margin-top:10px; font-family:"Courier New",monospace; font-size:13px; }
  @media (max-width:760px) { main { grid-template-columns:1fr; } .image-wrap { height:50vh; min-height:280px; } }
</style>
</head>
<body>
<header><strong>E4 Label Audit &middot; Rater <span id="role"></span></strong><span id="progress"></span></header>
<main>
  <div class="image-wrap"><img id="image" alt="image under annotation"></div>
  <section class="panel">
    <div class="rubric"><b>Use the image only.</b> Does an object matching the category below appear anywhere a reasonable viewer would recognize? Judge presence, not COCO-box quality.</div>
    <h2>Flagged category</h2><div class="target" id="target"></div>
    <div class="labels">
      <button data-label="VISIBLE">1 &middot; Visible</button>
      <button data-label="NOT_VISIBLE">2 &middot; Not visible</button>
      <button data-label="AMBIGUOUS">3 &middot; Ambiguous</button>
    </div>
    <div class="actions"><button id="prev">&larr; Previous</button><button id="save">Save &amp; next &rarr;</button></div>
    <div id="status"></div>
  </section>
</main>
<script>
let state, index=0, selectedLabel=null;
const $=id=>document.getElementById(id);
function paint(){document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('selected',b.dataset.label===selectedLabel));}
function render(){
  const item=state.items[index], saved=state.annotations[item.annotation_id]||{};
  $('role').textContent=state.role;
  $('progress').textContent=`${index+1} / ${state.items.length} · ${state.completed} saved`;
  $('image').src=`/image/${encodeURIComponent(item.annotation_id)}`;
  $('target').textContent=item.flagged_category;
  selectedLabel=saved.label||null; paint();
  $('status').textContent=saved.label?'Previously saved; saving again will replace it.':'';
}
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{selectedLabel=b.dataset.label;paint();});
$('prev').onclick=()=>{if(index>0){index--;render();}};
$('save').onclick=async()=>{
  if(!selectedLabel){$('status').textContent='Pick a label first.';return;}
  $('save').disabled=true; $('status').textContent='Saving…';
  const r=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({annotation_id:state.items[index].annotation_id,label:selectedLabel})});
  const result=await r.json(); $('save').disabled=false;
  if(!r.ok){$('status').textContent=result.error||'Save failed';return;}
  state.annotations[result.annotation.annotation_id]=result.annotation; state.completed=result.completed;
  if(index<state.items.length-1) index++; render();
};
document.addEventListener('keydown',e=>{
  if(['1','2','3'].includes(e.key)) document.querySelector(`[data-label="${['VISIBLE','NOT_VISIBLE','AMBIGUOUS'][Number(e.key)-1]}"]`).click();
  if(e.key==='ArrowLeft') $('prev').click();
});
fetch('/api/state').then(r=>r.json()).then(data=>{
  state=data; $('role').textContent=data.role;
  const first=data.items.findIndex(x=>!data.annotations[x.annotation_id]);
  index=first<0?0:first; render();
});
</script>
</body>
</html>"""


class AnnotationApp:
    def __init__(self, role, items, output):
        self.role = role
        self.items = items
        self.by_id = {item["annotation_id"]: item for item in items}
        self.output = output
        existing = read_jsonl(output)
        self.annotations = {row["annotation_id"]: row for row in existing}
        self.output_order = [row["annotation_id"] for row in existing]

    def completed_count(self):
        return sum(item["annotation_id"] in self.annotations for item in self.items)

    def save(self, payload):
        annotation_id = payload.get("annotation_id")
        if annotation_id not in self.by_id:
            raise ValueError("Unknown annotation_id")
        label = payload.get("label")
        if label not in VALID_LABELS:
            raise ValueError("Invalid label")
        row = {
            "annotation_id": annotation_id,
            "rater": self.role,
            "label": label,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if annotation_id not in self.annotations:
            self.output_order.append(annotation_id)
        self.annotations[annotation_id] = row
        ordered = [self.annotations[aid] for aid in self.output_order]
        write_jsonl_atomic(self.output, ordered)
        return row


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self.send_json({"role": app.role, "items": app.items,
                                 "annotations": app.annotations,
                                 "completed": app.completed_count()})
            elif path.startswith("/image/"):
                annotation_id = path.removeprefix("/image/")
                item = app.by_id.get(annotation_id)
                if not item:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                image_path = IMAGES_DIR / item["image_filename"]
                if not image_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(str(image_path))[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 8192:
                    raise ValueError("Request too large")
                row = app.save(json.loads(self.rfile.read(length)))
                self.send_json({"annotation": row, "completed": app.completed_count()})
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format_string, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(VALID_ROLES))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ids-file", type=Path, default=None,
                         help="restrict to these annotation_ids, one per line "
                              "(used for rater C on the A/B disagreement subset)")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    items = [{"annotation_id": r["annotation_id"], "image_filename":
              f"COCO_val2014_{r['image_id']:012d}.jpg", "flagged_category":
              r["category"]} for r in manifest]
    if args.ids_file:
        keep = {line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()}
        items = [it for it in items if it["annotation_id"] in keep]
        if not items:
            raise SystemExit(f"No items matched ids in {args.ids_file}")

    output = OUT_DIR / f"judgments_{args.role}.jsonl"
    app = AnnotationApp(args.role, items, output)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Rater {args.role}: http://{args.host}:{args.port}  ({len(items)} items)")
    print(f"Saving decisions to {output}")
    server.serve_forever()


if __name__ == "__main__":
    main()
