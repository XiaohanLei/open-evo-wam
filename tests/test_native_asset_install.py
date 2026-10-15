import hashlib,importlib.util,io,json,tarfile,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_installer',ROOT/'scripts/install_native_asset.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


class NativeAssetInstallTests(unittest.TestCase):
    def bundle(self,path,extra=None):
        content={'module.py':b'VALUE=1\n','LICENSE':b'Fixture license\n'}
        content['MANIFEST.json']=json.dumps({'files':{k:hashlib.sha256(v).hexdigest() for k,v in content.items()}}).encode()
        if extra:content.update(extra)
        with tarfile.open(path,'w:gz') as archive:
            for name,data in content.items():
                member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    def test_verified_install_and_preserve_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle.tar.gz';self.bundle(bundle)
            result=installer.install(bundle,root/'native','MANIFEST.json');self.assertEqual(result['files'],2)
            with self.assertRaisesRegex(ValueError,'existing'):installer.install(bundle,root/'native','MANIFEST.json')
            self.assertEqual((root/'native/module.py').read_bytes(),b'VALUE=1\n')
    def test_reject_traversal_and_uninventoried_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle.tar.gz';self.bundle(bundle,{'../escape':b'x'})
            with self.assertRaisesRegex(ValueError,'Unsafe'):installer.install(bundle,root/'native','MANIFEST.json')
            self.assertFalse((root/'escape').exists())
            self.bundle(bundle,{'unlisted.py':b'x'})
            with self.assertRaisesRegex(ValueError,'exactly'):installer.install(bundle,root/'native','MANIFEST.json')
            self.assertFalse((root/'native').exists())
