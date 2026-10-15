import hashlib,importlib.util,json,os,tarfile,tempfile,time,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('native_builder',Path(__file__).parents[1]/'scripts/build_native_asset.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NativeHeaders(unittest.TestCase):
    def test_old_source_time_does_not_leak_into_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=b'preserved upstream source\n';p=root/'source.py';p.write_bytes(data);os.utime(p,(1000,1000))
            manifest=root/'INVENTORY.json';manifest.write_text(json.dumps({'files':{'source.py':hashlib.sha256(data).hexdigest()}}));archive=root/'native.tar.gz';start=int(time.time())
            module.build(root,manifest,archive)
            with tarfile.open(archive) as t:
                for member in t.getmembers():
                    self.assertGreaterEqual(member.mtime,start);self.assertEqual((member.uid,member.gid),(0,0));self.assertEqual((member.uname,member.gname),('',''))
                self.assertEqual(t.extractfile('source.py').read(),data)
            self.assertEqual(p.stat().st_mtime,1000)

    def test_bad_inventory_does_not_create_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'source.py').write_text('actual');m=root/'INVENTORY.json';m.write_text(json.dumps({'files':{'source.py':'0'*64}}));out=root/'native.tar.gz'
            with self.assertRaises(ValueError):module.build(root,m,out)
            self.assertFalse(out.exists())
