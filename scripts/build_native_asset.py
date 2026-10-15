"""Package inventoried native sources with neutral headers and actual build time."""
import argparse,hashlib,io,json,tarfile,time
from pathlib import Path,PurePosixPath


def build(source_root,manifest,output):
    root=Path(source_root).resolve();manifest=Path(manifest);output=Path(output)
    raw=manifest.read_bytes();files=json.loads(raw)['files'];payload={}
    for name,expected in files.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or any(x in rel.parts for x in ['.git','__pycache__']):
            raise ValueError('Unsafe native inventory path')
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('Native inventory escapes source root')
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Native inventory mismatch: '+name)
        payload[name]=data
    if manifest.name in payload:raise ValueError('Inventory filename conflicts with source')
    payload[manifest.name]=raw;created=int(time.time());output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:
        output.chmod(0o600)
        with tarfile.open(fileobj=f,mode='w:gz') as archive:
            for name,data in sorted(payload.items()):
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
                info.mtime=created;info.uid=info.gid=0;info.uname=info.gname=''
                archive.addfile(info,io.BytesIO(data))
    return dict(files=len(files),packaged_at_unix=created,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,source_bytes_unchanged=True,owner_headers_neutral=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.source_root,a.manifest,a.output)))


if __name__=='__main__':main()
