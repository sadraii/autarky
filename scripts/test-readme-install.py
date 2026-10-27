#!/usr/bin/env python3
"""Run the README release-install block with owned files and no external tools."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


STUB = r'''
import os, pathlib, shutil, sys, tarfile
root = pathlib.Path(os.environ['INSTALL_TEST_ROOT']).resolve()
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with (root / 'calls').open('a') as log:
    log.write(name + '\n')
def owned(path):
    path = pathlib.Path(path).resolve()
    assert path.is_relative_to(root), path
    return path
if name == 'gh':
    archive = 'autarky-x86_64-unknown-linux-gnu.tar.gz'
    shutil.copyfile(root / 'payload', archive)
    mode = os.environ['INSTALL_TEST_CASE']
    if mode != 'missing-checksum':
        checksum = (root / 'checksum').read_text()
        if mode == 'name-mismatch': checksum = checksum.replace(archive, 'other.tar.gz')
        pathlib.Path(archive + '.sha256').write_text(checksum)
    if mode == 'corrupt':
        with open(archive, 'ab') as out: out.write(b'corruption')
    if mode == 'missing-archive': pathlib.Path(archive).unlink()
    if mode == 'download-failure': sys.exit(1)
elif name == 'tar':
    assert args == ['-xzf', 'autarky-x86_64-unknown-linux-gnu.tar.gz'], args
    with tarfile.open(args[1]) as archive:
        assert archive.getnames() == ['autarky', 'LICENSE']
        member = archive.getmember('autarky')
        owned('autarky').write_bytes(archive.extractfile(member).read())
        owned('autarky').chmod(0o755)
        owned('LICENSE').write_bytes(archive.extractfile('LICENSE').read())
elif name == 'mkdir':
    assert args[0] == '-p'
    owned(args[1]).mkdir(parents=True, exist_ok=True)
elif name == 'install':
    assert args[:2] == ['-m', '755'], args
    shutil.copyfile(owned(args[2]), owned(args[3]))
    owned(args[3]).chmod(0o755)
elif name == 'autarky':
    assert args == ['--version'], args
    print('autarky fixture')
else:
    raise AssertionError(name)
'''


def release_block():
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    section = readme.split('### Release archives\n', 1)[1]
    return section.split('```bash\n', 1)[1].split('\n```', 1)[0]


class ReadmeInstallTest(unittest.TestCase):
    def test_both_crates_distribute_the_same_mit_license(self):
        # Separate Cargo packages must retain the same copyright and permission notice.
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((root / 'server/LICENSE').read_bytes(), (root / 'LICENSE').read_bytes())
        self.assertTrue((root / 'LICENSE').read_text().startswith('MIT License\n'))

    def test_download_and_verification_gate_all_install_steps(self):
        block = release_block()
        shell = shutil.which('bash')
        assert shell is not None, 'bash is required to test the documented block'
        for case in ('valid', 'corrupt', 'download-failure', 'missing-checksum',
                     'name-mismatch', 'missing-archive'):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                    prefix='autarky-readme-install-') as directory:
                root = Path(directory)
                tools = root / 'tools'
                tools.mkdir()
                wrapper = '#!' + sys.executable + '\n' + STUB
                for name in ('gh', 'tar', 'mkdir', 'install'):
                    tool = tools / name
                    tool.write_text(wrapper)
                    tool.chmod(0o755)
                (tools / 'python3').symlink_to(sys.executable)
                with tarfile.open(root / 'payload', 'w:gz') as archive:
                    payload = wrapper.encode()
                    member = tarfile.TarInfo('autarky')
                    member.size = len(payload)
                    member.mode = 0o755
                    archive.addfile(member, io.BytesIO(payload))
                    license_text = (Path(__file__).resolve().parents[1] / 'LICENSE').read_bytes()
                    member = tarfile.TarInfo('LICENSE')
                    member.size = len(license_text)
                    member.mode = 0o644
                    archive.addfile(member, io.BytesIO(license_text))
                digest = hashlib.sha256((root / 'payload').read_bytes()).hexdigest()
                (root / 'checksum').write_text(
                    digest + '  autarky-x86_64-unknown-linux-gnu.tar.gz\n')
                work = root / 'work'
                work.mkdir()
                env = dict(os.environ, HOME=str(root / 'home'), PATH=str(tools),
                           INSTALL_TEST_ROOT=str(root), INSTALL_TEST_CASE=case)
                result = subprocess.run([shell, '--noprofile', '--norc', '-c', block],
                                        cwd=work, env=env, capture_output=True,
                                        text=True, timeout=10)
                calls = (root / 'calls').read_text().splitlines()
                if case == 'valid':
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(calls, ['gh', 'tar', 'mkdir', 'install', 'autarky'])
                    self.assertIn('Checksum verified', result.stdout)
                    self.assertEqual((root / 'home/.local/bin/autarky').read_text(), wrapper)
                    self.assertEqual((work / 'LICENSE').read_bytes(), license_text)
                else:
                    self.assertNotEqual(result.returncode, 0,
                                        f'{case}: shell accepted failure; calls={calls}')
                    self.assertEqual(calls, ['gh'],
                                     f'{case}: extraction/install/version ran after failure')
                    self.assertFalse((root / 'home/.local/bin/autarky').exists())


if __name__ == '__main__':
    unittest.main()
