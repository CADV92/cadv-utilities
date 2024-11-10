import os
from ftplib import FTP, error_perm
from datetime import datetime

class myFTP(FTP):
    def upload(self, filename, callback=None, path=None):
        """
        Uploads a file to the FTP server. If the specified path does not exist, it creates the necessary directories.

        Parameters:
        - filename (str): Path to the local file to upload.
        - callback (function, optional): Function to call on each block of data uploaded.
        - path (str, optional): Destination path on the server. Creates directories if they do not exist.
        """
        original_local_dir = os.getcwd()
        original_ftp_dir = self.pwd()
        
        # Ensure remote path exists and switch to it
        if path:
            if not self._path_exists(path):
                if not self._ensure_remote_path(path):
                    print(f"\tError: Could not create or access directory {path}. Insufficient permissions.")
                    return
            self.cwd(path)
        else:
            path = './'
        
        # Change to local directory of the file
        local_dir, basename = os.path.split(filename)
        os.chdir(local_dir)

        try:
            # Upload file
            with open(basename, 'rb') as f:
                self.storbinary(f'STOR {basename}', f, callback=callback)
        finally:
            # Restore the original directories for both FTP and local
            self.cwd(original_ftp_dir)
            os.chdir(original_local_dir)

    def download(self, filename, callback=None, path='./'):
        """
        Downloads a file from the FTP server.

        Parameters:
        - filename (str): Name of the file on the server to download.
        - callback (function, optional): Function to call on each block of data downloaded.
        - path (str): Local path where the file will be saved. 
        """
        with open(os.path.join(path, filename), "wb") as f:
            self.retrbinary(f'RETR {filename}', f.write if callback is None else callback)

    def remove_files_time(self, files, time=30, force=False):
        """
        Removes files from the FTP server if they are older than 30 days, or forcefully if specified.

        Parameters:
        - files (list of str): List of filenames to remove.
        - time  (int): Number of days since file creation.
        - force (bool): If True, removes the files regardless of age.
        """
        now = datetime.now()
        for file in files:
            response = self.voidcmd(f"MDTM {file}")
            mtime = datetime.strptime(response[4:18], '%Y%m%d%H%M%S')
            diff = now - mtime
            if diff.days > time or force:
                self.delete(file)
    
    def list(self, path='./', pattern=''):
        """
        Lists files on the FTP server matching the specified pattern.

        Parameters:
        - path (str): Directory path on the server to list files from.
        - pattern (str): Filename pattern to match.

        Returns:
        - list of str: List of filenames that match the pattern.
        """
        ls = []
        glob_path = os.path.join(path, f'*{pattern}*')
        self.retrlines(f'NLST {glob_path}', ls.append)
        return ls

    def _ensure_remote_path(self, path):
        """
        Ensures that the specified path exists on the FTP server, creating directories as necessary.

        Parameters:
        - path (str): Path on the server to ensure existence.
        """
        directories = path.strip('/').split('/')
        current_ftp_path = self.pwd()
        current_path = ''
        for directory in directories:
            current_path = f"{current_path}/{directory}" if current_path else directory
            try:
                self.cwd(current_path)
            except error_perm:
                # Create the directory if it does not exist
                self.mkd(current_path)
                self.cwd(current_path)
        # Return to the root directory after ensuring the path exists
        self.cwd(current_ftp_path)
        return True

    def _path_exists(self, path):
        """
        Checks if a path exists on the FTP server.

        Parameters:
        - path (str): Path on the server to check.

        Returns:
        - bool: True if path exists, False otherwise.
        """
        try:
            self.cwd(path)
            self.cwd('/')  # Return to the root if path exists
            return True
        except error_perm:
            return False