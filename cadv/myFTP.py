import os
from ftplib import FTP
from datetime import datetime

class myFTP(FTP):
    def upload(self, filename, callback=None, path=None):
        current_path = self.pwd()
        if not path:
            self.cwd(path)
        with open(filename, 'rb') as f:
            self.storbinary('STOR '+filename, f, callback=callback)
        self.cwd(current_path)
    
    def download(self, filename, callback=None, path='./'):
        if callback is None:
            with open(os.path.join(path,filename), "wb") as f:
                self.retrbinary('RETR '+filename, f.write)
        else:
            self.retrbinary('RETR '+filename, callback)
    
    def remove(self, files, force=False):
        now = datetime.now()
        for file in files:
            print(file)
            response = self.voidcmd(f"MDTM {file}")
            mtime = response[4:18]
            mtime = datetime.strptime(mtime, '%Y%m%d%H%M%S')
            diff = now - mtime
            if diff.days > 30 or force:
                self.delete(file)
    
    def list(self, **kwargs):
        path = kwargs['path'] if 'path' in kwargs else './'
        pattern = kwargs['pattern'] if 'pattern' in kwargs else ''
        ls = []
        glb_path = os.path.join(path,f'*{pattern}*')
        self.retrlines(f'NLST {glb_path}', ls.append)
        return ls
