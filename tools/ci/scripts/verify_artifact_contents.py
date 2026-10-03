"""Bounded, read-only archive inventory and source-byte checks; never extract files."""
import argparse
from email.parser import BytesParser
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
import tomllib
import zipfile
import zlib
from verify_source import CONTRACT, product_snapshot, read_json, require, safe_path


# Deliberately sized for this repository's small source packages, not arbitrary
# third-party archives. Current artifacts are < 0.2 MiB compressed / 0.8 MiB TAR.
# The compressed cap also bounds ZIP directory parsing and gzip header/padding.
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_MEMBERS = 512
MAX_NAME_BYTES = 1024
MAX_COMPRESSION_RATIO = 200
MAX_ZIP_DIRECTORY_BYTES = 512 * 1024
MAX_TAR_BYTES = 20 * 1024 * 1024
MAX_TAR_HEADERS = 1024
MAX_TAR_METADATA_BYTES = 4096
MAX_TAR_TOTAL_METADATA_BYTES = 64 * 1024
MAX_TAR_PADDING_BYTES = 10240
TAR_RATIO_FLOOR_BYTES = 64 * 1024  # Small TARs still contain record padding.
_BLOCK = 512
_ZIP_LOCAL = struct.Struct('<4s5H3L2H')
_ZIP_END = struct.Struct('<4s4H2LH')
_PAX_KEYS = {'path', 'mtime', 'atime', 'ctime', 'uid', 'gid', 'uname', 'gname'}
_WINDOWS_DEVICES = {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(n) for n in range(1, 10)),
                    *('LPT' + str(n) for n in range(1, 10))}


def _archive_name(name, directory=False):
    """Reject aliases before path normalization (including Windows aliases)."""
    require(isinstance(name, str) and bool(name), f'Unsafe archive member: {name!r}')
    if directory and name.endswith('/'):
        name = name[:-1]
    parts = name.split('/')
    require(len(name.encode('utf-8', 'surrogatepass')) <= MAX_NAME_BYTES
            and not any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in name)
            and '\\' not in name and ':' not in name
            and all(p not in ('', '.', '..') and not p.endswith((' ', '.'))
                    and p.split('.')[0].upper() not in _WINDOWS_DEVICES for p in parts),
            f'Unsafe archive member: {name!r}')
    return name


def _expected_inventory(expected):
    expected = set(expected)
    require(len(expected) <= MAX_MEMBERS, 'Expected archive member count exceeds limit')
    for name in expected:
        _archive_name(name)
    directories = {str(parent) for name in expected for parent in PurePosixPath(name).parents
                   if str(parent) != '.'}
    return expected, directories


def _archive_bytes(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, f'Compressed archive size exceeds limit: {path}')
    # Bound the read as well as stat, and retain one immutable input for both
    # inventory and payload verification (no path reopen / TOCTOU between them).
    with path.open('rb') as stream:
        data = stream.read(MAX_ARCHIVE_BYTES + 1)
    require(len(data) <= MAX_ARCHIVE_BYTES, f'Compressed archive size exceeds limit: {path}')
    return data


def _member_size(size, total):
    require(0 <= size <= MAX_MEMBER_BYTES, 'Archive member size exceeds limit')
    total += size
    require(total <= MAX_TOTAL_BYTES, 'Archive aggregate size exceeds limit')
    return total


class _TarStream:
    """Bound every seek/read before gzip can expand data, including hidden data."""
    def __init__(self, stream, compressed_size):
        self.stream = stream
        self.limit = min(MAX_TAR_BYTES, max(TAR_RATIO_FLOOR_BYTES, compressed_size * MAX_COMPRESSION_RATIO))

    def read(self, size):
        remaining = self.limit - self.stream.tell()
        data = self.stream.read(min(size, remaining + 1))
        require(len(data) <= remaining, 'TAR expansion/compression ratio exceeds limit')
        return data

    def exact(self, size):
        require(self.stream.tell() + size <= self.limit, 'TAR expansion/compression ratio exceeds limit')
        data = self.read(size)
        require(len(data) == size, 'Truncated TAR archive')
        return data

    def seek(self, offset):
        require(0 <= offset <= self.limit, 'TAR expansion/compression ratio exceeds limit')
        require(self.stream.seek(offset) == offset, 'Truncated TAR archive')


def _pax_metadata(data):
    """Parse only bounded local PAX records; never use tarfile's recursive reader."""
    result = {}
    while data:
        length, separator, _ = data.partition(b' ')
        require(separator and length.isdigit() and len(length) <= 5, 'Invalid PAX record')
        size = int(length)
        require(len(length) + 4 <= size <= len(data) and data[size - 1:size] == b'\n', 'Invalid PAX record')
        key, separator, value = data[len(length) + 1:size - 1].partition(b'=')
        require(separator, 'Invalid PAX record')
        key, value = key.decode('utf-8', 'strict'), value.decode('utf-8', 'strict')
        # In particular, reject sparse/size/linkpath overrides, global headers,
        # and unknown vendor extensions with potentially different semantics.
        require(key in _PAX_KEYS and key not in result, 'Unsupported or duplicate PAX metadata')
        if key in ('mtime', 'atime', 'ctime'):
            # Bounded fixed-point decimal syntax is finite by construction; no
            # NaN, infinities, exponents or unbounded numeric conversions.
            require(re.fullmatch(r'-?[0-9]{1,20}(?:\.[0-9]{1,20})?', value) is not None,
                    'Invalid PAX timestamp')
        elif key in ('uid', 'gid'):
            require(re.fullmatch(r'[0-9]{1,10}', value) is not None and int(value) <= 0xFFFFFFFF,
                    'Invalid PAX owner ID')
        elif key in ('uname', 'gname'):
            require(len(value.encode('utf-8')) <= 256
                    and not any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value),
                    'Invalid PAX owner name')
        result[key] = value
        data = data[size:]
    return result


def _tar_text(field):
    value, separator, padding = field.partition(b'\0')
    require(not separator or not any(padding), 'Unsafe archive member: embedded NUL')
    return value.decode('utf-8', 'strict')


def _tar_inventory(stream, expected, directories, label):
    members, seen, pending = [], set(), None
    total = metadata_total = headers = 0
    while True:
        header = stream.exact(_BLOCK)
        if not any(header):
            require(pending is None, 'Orphan PAX metadata')
            # Consume the trailer before materializing any file payload. A small valid
            # inventory must not hide huge zero padding / concatenated streams.
            tail = stream.read(MAX_TAR_PADDING_BYTES)
            require(len(tail) >= _BLOCK and len(tail) < MAX_TAR_PADDING_BYTES
                    and len(tail) % _BLOCK == 0 and not any(tail), 'Invalid or excessive TAR padding')
            break
        headers += 1
        require(headers <= MAX_TAR_HEADERS, 'TAR header count exceeds limit')
        member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        kind = header[156:157]
        if kind == tarfile.XHDTYPE:
            require(pending is None, 'Chained PAX metadata is unsupported')
            require(0 <= member.size <= MAX_TAR_METADATA_BYTES, 'TAR metadata size exceeds limit')
            metadata_total += member.size
            require(metadata_total <= MAX_TAR_TOTAL_METADATA_BYTES, 'TAR aggregate metadata exceeds limit')
            pending = _pax_metadata(stream.exact(member.size))
            stream.seek(((stream.stream.tell() + _BLOCK - 1) // _BLOCK) * _BLOCK)
            continue
        # Do not call getmembers(), next(), or extractfile(): those implicitly
        # consume GNU/PAX/sparse extension bodies before callers can bound them.
        require(kind in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE), 'Unsupported archive member type')
        directory = kind == tarfile.DIRTYPE
        raw_name = _tar_text(header[:100])
        prefix = _tar_text(header[345:500])
        if prefix:
            raw_name = prefix + '/' + raw_name
        _archive_name(raw_name, directory)
        name = _archive_name((pending or {}).get('path', raw_name), directory)
        pending = None
        require(not member.linkname and not (member.mode & 0o7000), 'Unsupported archive member attributes')
        require(name not in seen, f'Duplicate archive member: {name}')
        seen.add(name)
        require(len(seen) <= MAX_MEMBERS, 'Archive member count exceeds limit')
        require(name in (directories if directory else expected), f'{label} inventory differs: {name}')
        if directory:
            require(member.size == 0, 'Nonempty TAR directory')
        else:
            total = _member_size(member.size, total)
            members.append((name, stream.stream.tell(), member.size))
        stream.seek(stream.stream.tell() + ((member.size + _BLOCK - 1) // _BLOCK) * _BLOCK)
    actual = {name for name, _, _ in members}
    require(actual == expected, f'{label} inventory differs: {sorted(actual ^ expected)}')
    return members


def _read_tar_payload(stream, offset, size):
    stream.seek(offset)
    return stream.exact(size)


def tar_members(path, expected, label='TAR'):
    expected, directories = _expected_inventory(expected)
    data = _archive_bytes(path)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as archive:
            stream = _TarStream(archive, len(data))
            members = _tar_inventory(stream, expected, directories, label)
            return {name: _read_tar_payload(stream, offset, size) for name, offset, size in members}
    except (EOFError, OSError, tarfile.HeaderError, UnicodeError, zlib.error) as error:
        raise ValueError(f'Invalid TAR archive: {path}: {error}') from error


def _zip_extra(data, location):
    """Validate TLV framing, then reject extensions outside this wheel contract."""
    position = 0
    while position < len(data):
        require(position + 4 <= len(data), f'Malformed ZIP {location} extra field')
        _, size = struct.unpack_from('<2H', data, position)
        position += 4 + size
        require(position <= len(data), f'Malformed ZIP {location} extra field')
    # Reviewed wheels use no extras. In particular, local-only Unicode Path,
    # Unix attributes and ZIP64 fields must not give another reader a different
    # name, type or size from the checked central-directory entry.
    require(not data, f'Unsupported ZIP {location} extra fields')


def _zip_directory(data, expected, label):
    # Bound central-directory objects before ZipFile constructs its inventory.
    offset = data.rfind(b'PK\x05\x06', max(0, len(data) - 65557))
    require(offset >= 0 and offset + _ZIP_END.size <= len(data), 'Invalid ZIP end record')
    _, disk, directory_disk, disk_count, count, size, start, comment = _ZIP_END.unpack_from(data, offset)
    require(offset + _ZIP_END.size + comment == len(data), 'Invalid ZIP trailing data')
    require(disk == directory_disk == 0 and disk_count == count, 'Multipart ZIP is unsupported')
    require(count <= MAX_MEMBERS, 'Archive member count exceeds limit')
    require(count == len(expected), f'{label} inventory differs: member count')
    require(size <= MAX_ZIP_DIRECTORY_BYTES and start + size == offset, 'Invalid or excessive ZIP directory')
    position = start
    actual_count = 0
    while position < offset:
        require(position + 46 <= offset and data[position:position + 4] == b'PK\x01\x02', 'Invalid ZIP directory record')
        require(struct.unpack_from('<H', data, position + 34)[0] == 0, 'Multipart ZIP member is unsupported')
        namesize, extrasize, commentsize = struct.unpack_from('<3H', data, position + 28)
        extra_start = position + 46 + namesize
        record_end = extra_start + extrasize + commentsize
        require(record_end <= offset, 'Invalid ZIP directory record bounds')
        _zip_extra(data[extra_start:extra_start + extrasize], 'central')
        position = record_end
        actual_count += 1
        require(actual_count <= MAX_MEMBERS, 'Archive member count exceeds limit')
    require(position == offset and actual_count == count, 'ZIP directory count mismatch')
    return start


def _zip_payload_range(data, member, next_offset):
    offset = member.header_offset
    require(0 <= offset <= next_offset - _ZIP_LOCAL.size, 'Invalid ZIP local header')
    signature, version, flags, method, _, _, crc, compressed, size, namesize, extra = _ZIP_LOCAL.unpack_from(data, offset)
    require(signature == b'PK\x03\x04' and version <= 20, 'Unsupported ZIP local header')
    require(flags == member.flag_bits and method == member.compress_type, 'ZIP local metadata mismatch')
    start = offset + _ZIP_LOCAL.size + namesize + extra
    end = start + member.compress_size
    require(0 < namesize <= MAX_NAME_BYTES and start <= end <= next_offset, 'Invalid ZIP member bounds')
    _zip_extra(data[offset + _ZIP_LOCAL.size + namesize:start], 'local')
    name = data[offset + _ZIP_LOCAL.size:offset + _ZIP_LOCAL.size + namesize].decode('utf-8' if flags & 0x800 else 'cp437')
    require(name == member.orig_filename, 'ZIP local name mismatch')
    expected = (member.CRC, member.compress_size, member.file_size)
    if flags & 8:
        require((crc, compressed, size) in ((0, 0, 0), expected), 'ZIP local size mismatch')
        descriptor = data[end:next_offset]
        if len(descriptor) == 16 and descriptor[:4] == b'PK\x07\x08':
            descriptor = descriptor[4:]
        require(len(descriptor) == 12 and struct.unpack('<3L', descriptor) == expected, 'Invalid ZIP data descriptor')
    else:
        require((crc, compressed, size) == expected and end == next_offset, 'ZIP local size mismatch')
    return start, end


def _read_zip_payload(data, member, start, end):
    payload = data[start:end]
    if member.compress_type == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(payload, member.file_size + 1)
        # Do not trust declared sizes to mask excess output or trailing streams.
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                'Invalid or excessive ZIP compressed payload')
    require(len(payload) == member.file_size, 'ZIP payload size mismatch')
    require(zlib.crc32(payload) == member.CRC, 'ZIP payload CRC mismatch')
    return payload


def zip_members(path, expected, label='Wheel'):
    expected, _ = _expected_inventory(expected)
    data = _archive_bytes(path)
    directory_start = _zip_directory(data, expected, label)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            require(len(members) == len(expected) and len(members) <= MAX_MEMBERS, f'{label} inventory differs: member count')
            seen, total = set(), 0
            for member in members:
                name = _archive_name(member.orig_filename)
                require(name == member.filename, f'Unsafe archive member: {name!r}')
                mode = member.external_attr >> 16
                require(not member.is_dir() and not (member.external_attr & 0x10)
                        and stat.S_IFMT(mode) in (0, stat.S_IFREG) and not (mode & 0o7000),
                        'Unsupported ZIP member type or attributes')
                require(name not in seen, f'Duplicate archive member: {name}')
                seen.add(name)
                require(name in expected, f'{label} inventory differs: {name}')
                require(member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and not (member.flag_bits & ~0x80E) and member.extract_version <= 20,
                        'Unsupported ZIP compression or flags')
                total = _member_size(member.file_size, total)
                require(0 <= member.compress_size <= MAX_ARCHIVE_BYTES
                        and member.file_size <= member.compress_size * MAX_COMPRESSION_RATIO,
                        'ZIP compression ratio exceeds limit')
            require(seen == expected, f'{label} inventory differs: {sorted(seen ^ expected)}')
            ordered = sorted(members, key=lambda m: m.header_offset)
            require((ordered[0].header_offset if ordered else directory_start) == 0, 'Unexpected ZIP prefix')
            ranges = []
            for index, member in enumerate(ordered):
                end = ordered[index + 1].header_offset if index + 1 < len(ordered) else directory_start
                ranges.append((member, *_zip_payload_range(data, member, end)))
            return {member.filename: _read_zip_payload(data, member, start, end) for member, start, end in ranges}
    except (zipfile.BadZipFile, UnicodeError, zlib.error, struct.error) as error:
        raise ValueError(f'Invalid ZIP archive: {path}: {error}') from error


def one(folder, pattern):
    matches = list(folder.glob(pattern))
    require(len(matches) == 1, f'Expected exactly one {folder}/{pattern}')
    return matches[0]


def verify_artifacts(source, artifacts):
    source, artifacts = Path(source).resolve(), Path(artifacts).resolve()
    snapshot = product_snapshot(source)
    contract = read_json(source, CONTRACT)
    py = contract['languages']['python']
    project = tomllib.loads((source / py['metadata']).read_text())['project']
    normalized = py['distribution'].replace('-', '_')
    version = snapshot['versions']['python']
    pyroot = source / py['root']
    runtime = {e['path'][len(py['root']) + 1:] for e in snapshot['runtime_files'] if e['language'] == 'python'}
    prefix = f'{normalized}-{version}/'
    egg = f'src/{normalized}.egg-info/'
    expected_sdist = runtime | set(py['package_metadata']) | set(py['sdist_generated_files']) | {egg + n for n in py['egg_info_files']}
    sdist = tar_members(one(artifacts / 'python', '*.tar.gz'), {prefix + n for n in expected_sdist}, 'Sdist')
    sdist = {n.removeprefix(prefix): b for n, b in sdist.items()}
    for name in runtime | set(py['package_metadata']):
        require(sdist[name] == safe_path(pyroot, name).read_bytes(), f'Sdist source-byte mismatch: {name}')
    require(BytesParser().parsebytes(sdist['PKG-INFO'])['Version'] == version, 'Sdist version mismatch')
    info = f'{normalized}-{version}.dist-info/'
    licenses = {info + 'licenses/' + n for n in project['license-files']}
    expected_wheel = {n.removeprefix('src/') for n in runtime} | {info + n for n in py['wheel_metadata_files']} | licenses
    wheel = zip_members(one(artifacts / 'python', '*.whl'), expected_wheel)
    for name in runtime:
        require(wheel[name.removeprefix('src/')] == safe_path(pyroot, name).read_bytes(), f'Wheel source-byte mismatch: {name}')
    for name in project['license-files']:
        require(wheel[info + 'licenses/' + name] == safe_path(pyroot, name).read_bytes(), f'Wheel notice mismatch: {name}')
    require(BytesParser().parsebytes(wheel[info + 'METADATA'])['Version'] == version, 'Wheel version mismatch')
    js = contract['languages']['javascript']
    metadata = read_json(source, js['metadata'])
    allowlist = {'package/' + n for n in ['package.json', *metadata['files']]}
    npm = tar_members(one(artifacts / 'javascript', '*.tgz'), allowlist, 'npm')
    for name, data in npm.items():
        require(data == safe_path(source / js['root'], name.removeprefix('package/')).read_bytes(), f'npm source-byte mismatch: {name}')
    return {'status': 'passed', 'wheel_files': len(wheel), 'sdist_files': len(sdist), 'npm_files': len(npm)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_artifacts(args.source, args.artifacts), indent=2))
