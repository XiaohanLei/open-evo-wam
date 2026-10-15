"""Rewrite metadata strings without executing pickle globals or altering tensors."""
import pickletools
import struct


def rewrite_pickle_strings(data, replacements):
    operations = list(pickletools.genops(data))
    if not operations or operations[-1][0].name != 'STOP' or operations[-1][2] + 1 != len(data):
        raise ValueError('Expected one complete metadata pickle')
    pieces = []
    for i, (opcode, value, position) in enumerate(operations):
        end = operations[i + 1][2] if i + 1 < len(operations) else len(data)
        piece = data[position:end]
        if isinstance(value, str) and value in replacements:
            encoded = replacements[value].encode('utf-8')
            if opcode.name == 'SHORT_BINUNICODE' and len(encoded) <= 255:
                piece = b'\x8c' + bytes([len(encoded)]) + encoded
            elif opcode.name in {'SHORT_BINUNICODE', 'BINUNICODE'}:
                piece = b'X' + struct.pack('<I', len(encoded)) + encoded
            elif opcode.name == 'BINUNICODE8':
                piece = b'\x8d' + struct.pack('<Q', len(encoded)) + encoded
            else:
                raise ValueError('Only Unicode values may be rewritten; globals remain intact')
        pieces.append(piece)
    for i, (opcode, value, position) in enumerate(operations):
        if opcode.name != 'FRAME':
            continue
        frame_start, frame_end = position + 9, position + 9 + value
        indices = [j for j, (_, _, start) in enumerate(operations) if frame_start <= start < frame_end]
        for j in indices:
            end = operations[j + 1][2] if j + 1 < len(operations) else len(data)
            if end > frame_end:
                raise ValueError('Metadata opcode crosses frame boundary')
        pieces[i] = b'\x95' + struct.pack('<Q', sum(len(pieces[j]) for j in indices))
    result = b''.join(pieces)
    # This parser checks syntax without importing or constructing pickle objects.
    decoded = list(pickletools.genops(result))
    if len(decoded) != len(operations):
        raise ValueError('Metadata operation count changed')
    for (before, value, _), (after, updated, _) in zip(operations, decoded):
        if before.name == 'FRAME':
            if after.name != 'FRAME':
                raise ValueError('Frame changed type')
        elif isinstance(value, str) and value in replacements:
            if updated != replacements[value]:
                raise ValueError('String rewrite mismatch')
        elif before.name != after.name or value != updated:
            raise ValueError('Non-path metadata changed')
    return result
