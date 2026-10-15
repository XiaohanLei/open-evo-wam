"""Audit a source-bound native installation inventory using private privacy rules."""
import argparse,hashlib,json,re
from pathlib import Path


def audit(root,manifest,rules,exceptions):
    inventory=json.loads(Path(manifest).read_text())['files']
    allowed={(x['path'],x['category'],x['line']) for x in exceptions}
    findings=[];notices=[]
    for name,expected in inventory.items():
        path=Path(root)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Native inventory mismatch: '+name)
        text=path.read_bytes().decode('utf8',errors='replace')
        for number,line in enumerate(text.splitlines(),1):
            for rule in rules:
                if not re.search(rule['pattern'],line,re.I):continue
                key=(name,rule['category'],line)
                legal=(rule['category']=='infrastructure' and
                       re.match(r'^\s*(?:#\s*)?Copyright\b',line) is not None)
                if legal and key in allowed:
                    notices.append(dict(path=name,line=number,classification='mandatory_upstream_copyright',
                                        line_sha256=hashlib.sha256(line.encode()).hexdigest()))
                else:findings.append(dict(path=name,line=number,category=rule['category']))
    return dict(status='passed' if not findings else 'failed',files=len(inventory),
                manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                private_findings=findings,preserved_upstream_notices=notices,
                scope='inventoried native text and file hashes; not model tensor or runtime validation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--exceptions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.root,a.manifest,json.loads(a.rules.read_text()),json.loads(a.exceptions.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],files=report['files'],findings=len(report['private_findings']))))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
