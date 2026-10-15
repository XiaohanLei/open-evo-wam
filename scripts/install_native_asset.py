"""Install a private native source bundle with a verified, source-bound inventory."""
import argparse,hashlib,json,os,tarfile,tempfile,urllib.request
from pathlib import Path,PurePosixPath


def install(bundle,destination,manifest_name):
    destination=Path(destination).resolve()
    if PurePosixPath(manifest_name).name != manifest_name or manifest_name in ('.','..'):
        raise ValueError('Inventory must be a root-level filename')
    if destination.exists():raise ValueError('Use a new destination; existing native sources are preserved')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-staging-',dir=destination.parent) as directory:
        staging=Path(directory)
        with tarfile.open(bundle,'r:*') as archive:
            for member in archive.getmembers():
                relative=PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
                    raise ValueError('Unsafe native bundle path/link')
                if not member.isfile() and not member.isdir():raise ValueError('Unsupported native bundle member')
                if '.git' in relative.parts or '__pycache__' in relative.parts:raise ValueError('History/cache is not an installation asset')
                target=staging/relative
                if member.isdir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as source,target.open('xb') as output:
                        while chunk:=source.read(1024*1024):output.write(chunk)
                    target.chmod(0o600)
        manifest=json.loads((staging/manifest_name).read_text())
        files=manifest['files']
        if set(files)!={str(p.relative_to(staging)) for p in staging.rglob('*') if p.is_file() and p.name!=manifest_name}:
            raise ValueError('Native inventory must cover exactly the extracted files')
        for name,expected in files.items():
            actual=hashlib.sha256((staging/name).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('Native source inventory mismatch: '+name)
        if not (staging/'LICENSE').is_file():raise ValueError('Native license must accompany its sources')
        os.replace(staging,destination)
    return dict(installed=True,files=len(files),manifest_sha256=hashlib.sha256((destination/manifest_name).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path);p.add_argument('--url')
    p.add_argument('--sha256',required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--manifest',required=True);a=p.parse_args()
    if bool(a.bundle)==bool(a.url):raise ValueError('Choose exactly one local bundle or HTTPS URL')
    if len(a.sha256)!=64 or any(c not in '0123456789abcdef' for c in a.sha256):raise ValueError('Expected SHA256 must be explicit lowercase hex')
    a.destination.resolve().parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-download-',dir=a.destination.resolve().parent) as directory:
        bundle=a.bundle
        if a.url:
            if not a.url.startswith('https://'):raise ValueError('Native downloads require HTTPS')
            bundle=Path(directory)/'asset.tar.gz'
            with urllib.request.urlopen(a.url,timeout=60) as response,bundle.open('wb') as out:
                while chunk:=response.read(1024*1024):out.write(chunk)
        h=hashlib.sha256()
        with bundle.open('rb') as source:
            while chunk:=source.read(1024*1024):h.update(chunk)
        if h.hexdigest()!=a.sha256:raise ValueError('Native bundle SHA256 mismatch')
        print(json.dumps(install(bundle,a.destination,a.manifest)))


if __name__=='__main__':main()
