"""Read every indexed DCP tensor safely; this is not an architecture load."""
import argparse
from datetime import datetime, timezone
import importlib
import io
import json
from pathlib import Path
import pickle
import time


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            'torch.distributed.checkpoint.metadata': {
                'Metadata', 'TensorStorageMetadata', 'TensorProperties',
                '_MEM_FORMAT_ENCODING', 'ChunkStorageMetadata', 'MetadataIndex', 'StorageMeta'},
            'torch.distributed.checkpoint.filesystem': {'_StorageInfo'},
            'torch.serialization': {'_get_layout'},
            'torch': {'Size', 'bfloat16', 'float16', 'float32', 'float64', 'int64', 'int32', 'bool'},
        }
        # Python 3.13 moved pathlib classes; normalize metadata on Python 3.11.
        if module in {'pathlib', 'pathlib._local'} and name == 'PosixPath':
            return Path
        if name not in allowed.get(module, set()):
            raise pickle.UnpicklingError('Unexpected DCP metadata class')
        return getattr(importlib.import_module(module), name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    root = args.checkpoint.resolve()
    report = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                  tensors_read=0, logical_tensor_bytes=0,
                  architecture_load=False, inference=False)
    def save():
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        args.report.chmod(0o600)
    save()
    start = time.monotonic()
    try:
        metadata = MetadataReader(io.BytesIO((root / '.metadata').read_bytes())).load()
        seen = set()
        for index, storage in metadata.storage_data.items():
            spec = metadata.state_dict_metadata[index.fqn]
            chunks = [c for c in spec.chunks if tuple(c.offsets) == tuple(index.offset)]
            if len(chunks) != 1:
                raise ValueError('DCP chunk mapping is ambiguous')
            chunk = chunks[0]
            path = root / storage.relative_path
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError('DCP storage escapes checkpoint')
            if storage.offset < 0 or storage.length <= 0 or storage.offset + storage.length > path.stat().st_size:
                raise ValueError('DCP storage range is invalid')
            with path.open('rb') as source:
                source.seek(storage.offset)
                tensor = torch.load(io.BytesIO(source.read(storage.length)),
                                    map_location='cpu', weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(chunk.sizes) or tensor.dtype != spec.properties.dtype:
                raise ValueError('DCP tensor shape or dtype differs from metadata')
            if not torch.isfinite(tensor).all().item():
                raise ValueError('DCP tensor contains nonfinite values')
            key = (index.fqn, tuple(index.offset))
            if key in seen:
                raise ValueError('Duplicate DCP chunk')
            seen.add(key)
            report['tensors_read'] += 1
            report['logical_tensor_bytes'] += tensor.numel() * tensor.element_size()
            del tensor
            if report['tensors_read'] % 100 == 0:
                save()
        expected = {(name, tuple(c.offsets)) for name, spec in metadata.state_dict_metadata.items() for c in spec.chunks}
        if seen != expected:
            raise ValueError('Not all DCP chunks were read')
        report.update(status='passed', all_chunks_read=True, shapes_and_dtypes_match=True,
                      all_values_finite=True, elapsed_seconds=time.monotonic()-start)
        save()
        print(json.dumps(report))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__,
                      elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__ == '__main__':
    main()
