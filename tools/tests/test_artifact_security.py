"""Bounded hostile-archive fixtures. Offline only; no extraction or installation."""
import gzip
import io
from pathlib import Path
import stat
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci/scripts'))
import verify_artifact_contents as verifier


def tar_header(name='package/a', size=0, kind=tarfile.REGTYPE, link=''):
    member = tarfile.TarInfo(name)
    member.size, member.type, member.linkname = size, kind, link
    return member.tobuf(format=tarfile.USTAR_FORMAT)


def tar_record(name, data=b'x', kind=tarfile.REGTYPE):
    return tar_header(name, len(data), kind) + data + b'\0' * (-len(data) % 512)


def pax_record(key, value):
    data = f' {key}={value}\n'.encode()
    size = len(data) + 1
    while len(str(size)) + len(data) != size:
        size = len(str(size)) + len(data)
    return str(size).encode() + data


class ArchiveSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def tar(self, *records, padding=1024):
        path = self.root / 'test.tar.gz'
        path.write_bytes(gzip.compress(b''.join(records) + b'\0' * padding))
        return path

    def zip(self, entries, compression=zipfile.ZIP_STORED, nonseekable=False):
        class NonSeekable(io.BytesIO):
            def seek(self, *args):
                raise OSError('nonseekable fixture')
        output = NonSeekable() if nonseekable else io.BytesIO()
        with warnings.catch_warnings(), zipfile.ZipFile(output, 'w', compression=compression) as archive:
            warnings.simplefilter('ignore', UserWarning)  # Intentional duplicate members.
            for name, data in entries:
                archive.writestr(name, data)
        path = self.root / 'test.whl'
        path.write_bytes(output.getvalue())
        return path

    def zip_extra(self, extra, location):
        path = self.zip([('package/a', b'x')])
        data = bytearray(path.read_bytes())
        if location in ('local', 'both'):
            start = 30 + len('package/a')
            data[start:start] = extra
            struct.pack_into('<H', data, 28, len(extra))
        central = data.index(b'PK\x01\x02')
        if location in ('central', 'both'):
            start = central + 46 + len('package/a')
            data[start:start] = extra
            struct.pack_into('<H', data, central + 30, len(extra))
        end = data.rfind(b'PK\x05\x06')
        struct.pack_into('<2L', data, end + 12, end - central, central)
        path.write_bytes(data)
        return path

    def rejected(self, path, expected, pattern):
        is_zip = path.suffix == '.whl'
        reader = verifier.zip_members if is_zip else verifier.tar_members
        payload = '_read_zip_payload' if is_zip else '_read_tar_payload'
        with mock.patch.object(verifier, payload, side_effect=AssertionError('payload was read')) as body:
            with self.assertRaisesRegex(ValueError, pattern):
                reader(path, expected)
            body.assert_not_called()

    def test_valid_stored_deflated_and_descriptor_wheels(self):
        files = {'package/a': b'hello\n', 'package/b': b'world\n', 'package/empty': b''}
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            for nonseekable in (False, True):
                with self.subTest(compression=compression, descriptor=nonseekable):
                    path = self.zip(files.items(), compression, nonseekable)
                    self.assertEqual(verifier.zip_members(path, files), files)

    def test_valid_tar_directories_pax_and_ustar_prefix(self):
        output = io.BytesIO()
        long_name = 'package/' + 'a' * 80 + '/' + 'b' * 40
        with tarfile.open(fileobj=output, mode='w:gz', format=tarfile.PAX_FORMAT) as archive:
            directory = tarfile.TarInfo('package/')
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            member = tarfile.TarInfo(long_name)
            member.pax_headers = {'mtime': '123.456', 'uname': 'builder'}
            member.size = 5
            archive.addfile(member, io.BytesIO(b'hello'))
        path = self.root / 'test.tar.gz'
        path.write_bytes(output.getvalue())
        self.assertEqual(verifier.tar_members(path, {long_name}), {long_name: b'hello'})
        path = self.tar(tar_record(long_name, b'hello'))
        self.assertEqual(verifier.tar_members(path, {long_name}), {long_name: b'hello'})

    def test_extra_and_missing_names_rejected_before_any_payload(self):
        cases = [([('package/a', b'a'), ('package/extra', b'b')], {'package/a', 'package/b'}),
                 ([('package/a', b'a')], {'package/a', 'package/b'}),
                 ([('package/a', b'a'), ('package/extra', b'b')], {'package/a'})]
        for entries, expected in cases:
            with self.subTest(entries=entries, expected=expected):
                self.rejected(self.zip(entries), expected, 'inventory differs')
                self.rejected(self.tar(*(tar_record(n, b) for n, b in entries)), expected, 'inventory differs')

    def test_duplicate_names_rejected_before_any_payload(self):
        entries = [('package/a', b'a'), ('package/a', b'b')]
        self.rejected(self.zip(entries), {'package/a', 'package/b'}, 'Duplicate')
        self.rejected(self.tar(*(tar_record(n, b) for n, b in entries)), {'package/a'}, 'Duplicate')

    def test_nonzero_zip_member_disk_rejected_before_payload(self):
        for volume in (1, 65535):
            with self.subTest(volume=volume):
                path = self.zip([('package/a', b'x')])
                data = bytearray(path.read_bytes())
                central = data.index(b'PK\x01\x02')
                struct.pack_into('<H', data, central + 34, volume)
                path.write_bytes(data)
                self.rejected(path, {'package/a'}, 'Multipart ZIP member')

    def test_unsafe_zip_and_tar_names_rejected_without_normalizing(self):
        names = ['/absolute', '../outside', 'package/../a', './package/a', 'package//a',
                 'package/./a', 'package/a/', 'C:/outside', 'C:outside', 'package/a:b',
                 'package\\a', 'package/a\n', 'package/a\x7f', 'package/a.', 'package/a ',
                 'package/CON', 'package/lpt1.txt']
        for name in names:
            with self.subTest(name=name):
                self.rejected(self.zip([(name, b'x')]), {'package/a'}, 'Unsafe archive member')
                self.rejected(self.tar(tar_record(name)), {'package/a'}, 'Unsafe archive member')

    def test_embedded_nul_zip_and_tar_paths_fail_before_payload(self):
        path = self.zip([('package/a', b'x')])
        path.write_bytes(path.read_bytes().replace(b'package/a', b'package\0a'))
        self.rejected(path, {'package/a'}, 'Unsafe archive member')
        self.rejected(self.tar(tar_record('package\0a')), {'package/a'}, 'embedded NUL')

    def test_zip_nonregular_types_and_dos_directories_fail(self):
        for system in (0, 3):
            for kind in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFDIR):
                with self.subTest(system=system, kind=kind):
                    member = zipfile.ZipInfo('package/a')
                    member.create_system = system
                    member.external_attr = (kind | 0o644) << 16
                    self.rejected(self.zip([(member, b'x')]), {'package/a'}, 'Unsupported ZIP member')
        member = zipfile.ZipInfo('package/a')
        member.external_attr = (stat.S_IFREG | 0o644) << 16 | 0x10
        self.rejected(self.zip([(member, b'x')]), {'package/a'}, 'Unsupported ZIP member')

    def test_tar_nonregular_and_extension_types_rejected_before_extension_reads(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE,
                     tarfile.BLKTYPE, tarfile.GNUTYPE_SPARSE, tarfile.GNUTYPE_LONGNAME,
                     tarfile.GNUTYPE_LONGLINK, tarfile.XGLTYPE, b'?'):
            with self.subTest(kind=kind):
                path = self.tar(tar_record('package/a'), tar_header('package/b', 10**9, kind))
                self.rejected(path, {'package/a', 'package/b'}, 'Unsupported archive member type')

    def test_privileged_attributes_rejected(self):
        member = zipfile.ZipInfo('package/a')
        member.external_attr = (stat.S_IFREG | 0o4644) << 16
        self.rejected(self.zip([(member, b'x')]), {'package/a'}, 'Unsupported ZIP member')
        member = tarfile.TarInfo('package/a')
        member.mode = 0o4644
        self.rejected(self.tar(member.tobuf()), {'package/a'}, 'Unsupported archive member attributes')

    def test_tar_directories_must_be_zero_sized_unique_expected_ancestors(self):
        self.rejected(self.tar(tar_record('other', b'', tarfile.DIRTYPE)), {'package/a'}, 'inventory differs')
        self.rejected(self.tar(tar_record('package/', b'x', tarfile.DIRTYPE)), {'package/a'}, 'Nonempty TAR directory')
        self.rejected(self.tar(tar_record('package/', b'', tarfile.DIRTYPE),
                               tar_record('package', b'', tarfile.DIRTYPE)), {'package/a'}, 'Duplicate')
        self.rejected(self.tar(tar_record('package//', b'', tarfile.DIRTYPE)), {'package/a'}, 'Unsafe archive member')

    def test_compressed_size_limit_before_archive_parser(self):
        for suffix, parser, read in (('.whl', 'ZipFile', verifier.zip_members),
                                     ('.tar.gz', 'GzipFile', verifier.tar_members)):
            with self.subTest(suffix=suffix):
                path = self.root / ('oversized' + suffix)
                with path.open('wb') as stream:
                    stream.truncate(verifier.MAX_ARCHIVE_BYTES + 1)
                module = verifier.zipfile if suffix == '.whl' else verifier.gzip
                with mock.patch.object(module, parser) as opened:
                    with self.assertRaisesRegex(ValueError, 'Compressed archive size exceeds limit'):
                        read(path, {'package/a'})
                    opened.assert_not_called()

    def test_archive_read_is_bounded_even_if_stat_lies(self):
        path = self.root / 'oversized.tar.gz'
        with path.open('wb') as stream:
            stream.truncate(verifier.MAX_ARCHIVE_BYTES + 1)
        with mock.patch.object(Path, 'stat', return_value=mock.Mock(st_size=0)):
            self.rejected(path, {'package/a'}, 'Compressed archive size exceeds limit')

    def test_member_size_limit_before_payload(self):
        path = self.zip([('package/a', b'x')])
        data = bytearray(path.read_bytes())
        central = data.index(b'PK\x01\x02')
        struct.pack_into('<L', data, 22, verifier.MAX_MEMBER_BYTES + 1)
        struct.pack_into('<L', data, central + 24, verifier.MAX_MEMBER_BYTES + 1)
        path.write_bytes(data)
        self.rejected(path, {'package/a'}, 'Archive member size exceeds limit')
        self.rejected(self.tar(tar_header(size=verifier.MAX_MEMBER_BYTES + 1)),
                      {'package/a'}, 'Archive member size exceeds limit')

    def test_aggregate_size_limit_before_payload(self):
        files = {'package/a': b'aaa', 'package/b': b'bbb'}
        with mock.patch.object(verifier, 'MAX_TOTAL_BYTES', 5):
            self.rejected(self.zip(files.items()), files, 'aggregate size')
            self.rejected(self.tar(*(tar_record(n, b) for n, b in files.items())), files, 'aggregate size')

    def test_member_count_limit_before_payload(self):
        path = self.zip([('package/a', b'a'), ('package/b', b'b')])
        with mock.patch.object(verifier, 'MAX_MEMBERS', 1):
            self.rejected(path, {'package/a'}, 'member count exceeds limit')
            self.rejected(self.tar(tar_record('package/', b'', tarfile.DIRTYPE), tar_record('package/a')),
                          {'package/a'}, 'member count exceeds limit')

    def test_compression_ratio_rejects_real_zip_and_tar_bombs_before_payload(self):
        bomb = b'0' * (1024 * 1024)
        self.rejected(self.zip([('package/a', bomb)], zipfile.ZIP_DEFLATED), {'package/a'}, 'compression ratio')
        self.rejected(self.tar(tar_record('package/a', bomb)), {'package/a'}, 'expansion/compression ratio')

    def test_tar_absolute_expansion_limit_before_payload(self):
        path = self.tar(tar_record('package/a', b'x' * 1024))
        with mock.patch.object(verifier, 'MAX_TAR_BYTES', 1024):
            self.rejected(path, {'package/a'}, 'expansion/compression ratio')

    def test_tar_hidden_pax_size_is_checked_before_metadata_read(self):
        path = self.tar(tar_header('././@PaxHeader', verifier.MAX_TAR_METADATA_BYTES + 1, tarfile.XHDTYPE))
        with mock.patch.object(verifier, '_pax_metadata') as parser:
            self.rejected(path, {'package/a'}, 'TAR metadata size exceeds limit')
            parser.assert_not_called()

    def test_tar_pax_aggregate_and_header_count_are_bounded(self):
        metadata = tar_record('././@PaxHeader', pax_record('mtime', '123.45'), tarfile.XHDTYPE)
        path = self.tar(metadata, tar_record('package/a'), metadata, tar_record('package/b'))
        with mock.patch.object(verifier, 'MAX_TAR_TOTAL_METADATA_BYTES', 20):
            self.rejected(path, {'package/a', 'package/b'}, 'TAR aggregate metadata')
        with mock.patch.object(verifier, 'MAX_TAR_HEADERS', 2):
            self.rejected(path, {'package/a', 'package/b'}, 'TAR header count')

    def test_tar_chained_or_orphan_metadata_rejected_without_recursion(self):
        metadata = tar_record('././@PaxHeader', pax_record('mtime', '1'), tarfile.XHDTYPE)
        self.rejected(self.tar(metadata, metadata, tar_record('package/a')), {'package/a'}, 'Chained PAX')
        self.rejected(self.tar(metadata), {'package/a'}, 'Orphan PAX')

    def test_tar_pax_size_sparse_link_and_unknown_overrides_rejected(self):
        for key in ('size', 'GNU.sparse.size', 'GNU.sparse.map', 'GNU.sparse.major', 'linkpath', 'vendor.unknown'):
            with self.subTest(key=key):
                metadata = tar_record('././@PaxHeader', pax_record(key, '1000000000'), tarfile.XHDTYPE)
                self.rejected(self.tar(metadata, tar_record('package/a')), {'package/a'}, 'Unsupported or duplicate PAX')

    def test_tar_pax_timestamp_owner_ids_and_names_have_strict_values(self):
        cases = [
            ('mtime', ['not-a-number', 'NaN', 'Infinity', '1e999', '1.', '--1', '1' * 21, '0.' + '1' * 21]),
            ('atime', ['inf', '1\x00']), ('ctime', ['nan', '']),
            ('uid', ['not-a-number', '-1', '+1', '1.5', '4294967296', '1' * 11]),
            ('gid', ['not-a-number', '-1', '']),
            ('uname', ['root\x00trailing', 'a\n', 'a\t', 'a\x7f', 'a\x85', 'a' * 257]),
            ('gname', ['root\x00trailing', 'a\r'])]
        for key, values in cases:
            for value in values:
                with self.subTest(key=key, value=value):
                    metadata = tar_record('././@PaxHeader', pax_record(key, value), tarfile.XHDTYPE)
                    self.rejected(self.tar(metadata, tar_record('package/a')), {'package/a'}, 'Invalid PAX')

    def test_tar_valid_finite_pax_times_and_owner_values_pass(self):
        fields = {'mtime': '1234567890.123456789', 'atime': '-0.125', 'ctime': '0',
                  'uid': '0', 'gid': '4294967295', 'uname': 'builder', 'gname': ''}
        metadata = tar_record('././@PaxHeader', b''.join(pax_record(k, v) for k, v in fields.items()), tarfile.XHDTYPE)
        path = self.tar(metadata, tar_record('package/a'))
        self.assertEqual(verifier.tar_members(path, {'package/a'}), {'package/a': b'x'})

    def test_tar_pax_paths_and_records_are_validated(self):
        for value, message in [('../outside', 'Unsafe archive member'), ('package/extra', 'inventory differs')]:
            metadata = tar_record('././@PaxHeader', pax_record('path', value), tarfile.XHDTYPE)
            self.rejected(self.tar(metadata, tar_record('package/a')), {'package/a'}, message)
        records = [b'0 x=\n', b'999999999999 x=\n', b'99 mtime=1\n', b'11 mtime=1!',
                   pax_record('mtime', '1') + pax_record('mtime', '2')]
        for data in records:
            with self.subTest(data=data):
                self.rejected(self.tar(tar_record('././@PaxHeader', data, tarfile.XHDTYPE), tar_record('package/a')),
                              {'package/a'}, 'PAX')

    def test_tar_trailing_padding_and_concatenated_archives_rejected_before_payload(self):
        self.rejected(self.tar(tar_record('package/a'), padding=1024 * 1024), {'package/a'}, 'excessive TAR padding')
        path = self.tar(tar_record('package/a'))
        path.write_bytes(path.read_bytes() + gzip.compress(tar_record('package/hidden') + b'\0' * 1024))
        self.rejected(path, {'package/a'}, 'TAR padding')
        self.rejected(self.tar(tar_record('package/a'), padding=512), {'package/a'}, 'TAR padding')

    def test_tar_bad_checksum_truncation_and_gzip_crc_fail_before_payload(self):
        header = bytearray(tar_header())
        header[0] ^= 1
        self.rejected(self.tar(bytes(header)), {'package/a'}, 'Invalid TAR archive')
        self.rejected(self.tar(tar_header(size=20)), {'package/a'}, 'TAR')
        path = self.tar(tar_record('package/a'))
        data = bytearray(path.read_bytes())
        data[-8] ^= 1
        path.write_bytes(data)
        self.rejected(path, {'package/a'}, 'Invalid TAR archive')

    def test_zip_forged_directory_count_is_rejected_before_zipfile_parsing(self):
        path = self.zip([('package/a', b'x'), ('package/b', b'y')])
        data = bytearray(path.read_bytes())
        end = data.rfind(b'PK\x05\x06')
        struct.pack_into('<2H', data, end + 8, 1, 1)
        path.write_bytes(data)
        with mock.patch.object(verifier.zipfile, 'ZipFile') as parser:
            self.rejected(path, {'package/a'}, 'ZIP directory count mismatch')
            parser.assert_not_called()

    def test_zip_directory_size_bound_precedes_zipfile_parsing(self):
        path = self.zip([('package/a', b'x')])
        with mock.patch.object(verifier, 'MAX_ZIP_DIRECTORY_BYTES', 1), \
                mock.patch.object(verifier.zipfile, 'ZipFile') as parser:
            self.rejected(path, {'package/a'}, 'excessive ZIP directory')
            parser.assert_not_called()

    def test_zip_late_local_name_mismatch_rejected_before_all_payloads(self):
        path = self.zip([('package/a', b'x'), ('package/b', b'y')])
        data = bytearray(path.read_bytes())
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            offset = archive.infolist()[1].header_offset + 30
        data[offset:offset + 9] = b'../escape'
        path.write_bytes(data)
        self.rejected(path, {'package/a', 'package/b'}, 'ZIP local name mismatch')

    def test_zip_local_size_offsets_and_descriptors_validated_before_payload(self):
        for case in ('size', 'offset', 'descriptor', 'encryption', 'method'):
            with self.subTest(case=case):
                path = self.zip([('package/a', b'x')], nonseekable=case == 'descriptor')
                data = bytearray(path.read_bytes())
                central = data.index(b'PK\x01\x02')
                if case == 'size':
                    struct.pack_into('<L', data, 22, 1000)
                elif case == 'offset':
                    struct.pack_into('<L', data, central + 42, 1)
                elif case == 'descriptor':
                    data[central - 1] ^= 1
                elif case == 'encryption':
                    struct.pack_into('<H', data, central + 8, 1)
                else:
                    struct.pack_into('<H', data, central + 10, zipfile.ZIP_BZIP2)
                path.write_bytes(data)
                self.rejected(path, {'package/a'}, 'ZIP')

    def test_zip_extra_name_type_size_and_unknown_extensions_rejected_before_payload(self):
        unicode_path = b'\x01' + struct.pack('<L', zlib.crc32(b'package/a')) + b'../outside'
        fields = [(0x7075, unicode_path), (0x0001, struct.pack('<Q', 10**9)),
                  (0x000D, b'unix'), (0x7855, b'unix'), (0x000A, b'ntfs'),
                  (0x5455, b'timestamp'), (0xFFFF, b'unknown')]
        for location in ('local', 'central', 'both'):
            for identifier, payload in fields:
                with self.subTest(location=location, identifier=identifier):
                    extra = struct.pack('<2H', identifier, len(payload)) + payload
                    self.rejected(self.zip_extra(extra, location), {'package/a'},
                                  'Unsupported ZIP .* extra fields')

    def test_zip_malformed_and_trailing_extra_bytes_rejected_before_payload(self):
        for location in ('local', 'central', 'both'):
            for extra in (b'x', b'\x75\x70\xff\xff', b'\xff\xff\x00\x00x'):
                with self.subTest(location=location, extra=extra):
                    self.rejected(self.zip_extra(extra, location), {'package/a'},
                                  'Malformed ZIP .* extra field')

    def test_zip_central_extra_validated_before_zipfile_parses_extensions(self):
        extra = struct.pack('<2H', 0x7075, 1) + b'\x01'
        path = self.zip_extra(extra, 'central')
        with mock.patch.object(verifier.zipfile, 'ZipFile') as parser:
            self.rejected(path, {'package/a'}, 'Unsupported ZIP central extra fields')
            parser.assert_not_called()

    def test_zip_declared_size_cannot_hide_actual_deflate_expansion(self):
        path = self.zip([('package/a', b'0' * 100000)], zipfile.ZIP_DEFLATED)
        data = bytearray(path.read_bytes())
        central = data.index(b'PK\x01\x02')
        # Forge a tiny, internally consistent declared size and prefix CRC.
        # The real DEFLATE stream still expands to 100000 bytes.
        struct.pack_into('<L', data, 14, zlib.crc32(b'0'))
        struct.pack_into('<L', data, 22, 1)
        struct.pack_into('<L', data, central + 16, zlib.crc32(b'0'))
        struct.pack_into('<L', data, central + 24, 1)
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'excessive ZIP compressed payload'):
            verifier.zip_members(path, {'package/a'})

    def test_zip_crc_mismatch_is_rejected(self):
        path = self.zip([('package/a', b'x')])
        data = bytearray(path.read_bytes())
        data[30 + len('package/a')] = ord('y')
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'ZIP payload CRC mismatch'):
            verifier.zip_members(path, {'package/a'})

    def test_no_extraction_or_recursive_tar_reader_is_used(self):
        path = self.tar(tar_record('package/a'))
        with mock.patch.object(tarfile, 'open', side_effect=AssertionError('unbounded TAR parser')), \
                mock.patch.object(tarfile.TarFile, 'extractall', side_effect=AssertionError('extraction')), \
                mock.patch.object(tarfile.TarFile, 'extractfile', side_effect=AssertionError('extraction')):
            self.assertEqual(verifier.tar_members(path, {'package/a'}), {'package/a': b'x'})
        path = self.zip([('package/a', b'x')])
        with mock.patch.object(zipfile.ZipFile, 'extractall', side_effect=AssertionError('extraction')), \
                mock.patch.object(zipfile.ZipFile, 'read', side_effect=AssertionError('unbounded payload read')):
            self.assertEqual(verifier.zip_members(path, {'package/a'}), {'package/a': b'x'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
