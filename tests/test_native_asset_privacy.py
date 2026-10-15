import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('native_asset_audit',Path(__file__).parents[1]/'scripts/audit_native_asset.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NativeAssetPrivacy(unittest.TestCase):
    def check(self,text,category,exceptions):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'source.py').write_text(text)
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'files':{'source.py':hashlib.sha256(text.encode()).hexdigest()}}))
            return module.audit(root,manifest,[{'category':category,'pattern':'protected_marker'}],exceptions)

    def test_only_exact_infrastructure_copyright_is_exempt(self):
        line='# Copyright protected_marker. All rights reserved.'
        exceptions=[{'path':'source.py','category':'infrastructure','line':line}]
        self.assertEqual(self.check(line+'\n','infrastructure',exceptions)['status'],'passed')
        self.assertEqual(self.check(line+'\nendpoint="protected_marker"\n','infrastructure',exceptions)['status'],'failed')
        self.assertEqual(self.check(line+' changed\n','infrastructure',exceptions)['status'],'failed')

    def test_personal_identity_cannot_be_exempted(self):
        line='# Copyright protected_marker'
        result=self.check(line+'\n','personal_identity',[{'path':'source.py','category':'personal_identity','line':line}])
        self.assertEqual(result['status'],'failed')
