import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for item in json.loads((ROOT/"provenance/datasets.json").read_text(encoding="utf-8"))["datasets"]:
    p=ROOT/item["relative_path"]
    if p.exists():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"], p
        continue
    content=urllib.request.urlopen(item["source_url"],timeout=120).read()
    assert hashlib.sha256(content).hexdigest()==item["sha256"], item["dataset"]
    assert len(content)==item["bytes"]
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(content)
    print(item["dataset"],item["sha256"])
