import os
import fnmatch
from datetime import datetime, timezone
import paramiko

class mySFTP:
    def __init__(self, host, username, password=None, port=22, key_filename=None):
        self.transport = paramiko.Transport((host, port))
        if key_filename:
            private_key = paramiko.RSAKey.from_private_key_file(key_filename)
            self.transport.connect(username=username, pkey=private_key)
        else:
            self.transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def upload(self, filename, path=None):
        """
        Uploads a file to the SFTP server, creating directories if necessary.
        """
        original_local_dir = os.getcwd()
        local_dir, basename = os.path.split(filename)
        os.chdir(local_dir)

        if path:
            self._ensure_remote_path(path)
            remote_file = os.path.join(path, basename)
        else:
            remote_file = basename

        try:
            self.sftp.put(basename, remote_file)
        finally:
            os.chdir(original_local_dir)

    def download(self, filename, path='./'):
        """
        Downloads a file from the SFTP server to a local path.
        """
        local_path = os.path.join(path, os.path.basename(filename))
        self.sftp.get(filename, local_path)

    def list(self, path='.', pattern=''):
        """
        Lists files in a directory on the SFTP server matching a pattern.
        """
        files = self.sftp.listdir(path)
        if pattern:
            files = fnmatch.filter(files, f"*{pattern}*")
        return files

    def remove_files_time(self, path='.', time=30, force=False):
        """
        Removes files older than `time` days or all if `force=True`.
        """
        now = datetime.now(timezone.utc)
        for filename in self.sftp.listdir_attr(path):
            file_mtime = datetime.fromtimestamp(filename.st_mtime, timezone.utc)
            if (now - file_mtime).days > time or force:
                self.sftp.remove(os.path.join(path, filename.filename))

    def _ensure_remote_path(self, path):
        """
        Ensures that the specified remote path exists by creating missing directories.
        """
        dirs = path.strip('/').split('/')
        current_path = ''
        for directory in dirs:
            current_path = os.path.join(current_path, directory)
            try:
                self.sftp.stat(current_path)
            except FileNotFoundError:
                self.sftp.mkdir(current_path)

    def _path_exists(self, path):
        """
        Checks if a remote path exists.
        """
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def close(self):
        """
        Closes the SFTP session.
        """
        self.sftp.close()
        self.transport.close()
