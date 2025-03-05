import os
import logging
from typing import List, Optional, Callable
import paramiko
from pathlib import Path
from datetime import datetime

class mySSH:
    def __init__(self, hostname: str, username: str, port: int = 22):
        """
        Inicializa una conexión SSH.

        Parameters:
        - hostname (str): Nombre o IP del servidor
        - username (str): Nombre de usuario
        - port (int): Puerto SSH (default: 22)
        """
        self.hostname = hostname
        self.username = username
        self.port = port
        self.client = None
        self.sftp = None
        self.logger = logging.getLogger(__name__)

    def connect(self, password: str = None, key_filename: str = None):
        """
        Establece la conexión SSH usando contraseña o clave SSH.

        Parameters:
        - password (str, optional): Contraseña para autenticación
        - key_filename (str, optional): Ruta al archivo de clave privada SSH
        """
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            if key_filename:
                self.client.connect(
                    hostname=self.hostname,
                    username=self.username,
                    port=self.port,
                    key_filename=key_filename
                )
            else:
                self.client.connect(
                    hostname=self.hostname,
                    username=self.username,
                    port=self.port,
                    password=password
                )
            self.sftp = self.client.open_sftp()
            self.logger.info(f"Conectado exitosamente a {self.hostname}")
        except Exception as e:
            self.logger.error(f"Error de conexión: {str(e)}")
            raise

    def upload(self, filename: str, remote_path: str = './', callback: Callable = None) -> bool:
        """
        Sube un archivo al servidor SSH.

        Parameters:
        - filename (str): Ruta al archivo local
        - remote_path (str): Ruta remota donde se guardará el archivo
        - callback (callable): Función para monitorear el progreso
        """
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Archivo no encontrado: {filename}")

            if not self._path_exists(remote_path):
                self._ensure_remote_path(remote_path)

            local_path = Path(filename)
            remote_file = os.path.join(remote_path, local_path.name)
            
            self.sftp.put(str(local_path), remote_file, callback=callback)
            self.logger.info(f"Archivo subido exitosamente: {remote_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al subir archivo: {str(e)}")
            return False

    def download(self, remote_file: str, local_path: str = './', callback: Callable = None) -> bool:
        """
        Descarga un archivo del servidor SSH.

        Parameters:
        - remote_file (str): Ruta al archivo remoto
        - local_path (str): Ruta local donde se guardará el archivo
        - callback (callable): Función para monitorear el progreso
        """
        try:
            local_file = os.path.join(local_path, os.path.basename(remote_file))
            self.sftp.get(remote_file, local_file, callback=callback)
            self.logger.info(f"Archivo descargado exitosamente: {local_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al descargar archivo: {str(e)}")
            return False

    def list(self, path: str = './', pattern: str = '') -> List[str]:
        """
        Lista archivos en el servidor que coinciden con el patrón.

        Parameters:
        - path (str): Directorio a listar
        - pattern (str): Patrón para filtrar archivos

        Returns:
        - List[str]: Lista de archivos que coinciden
        """
        try:
            files = self.sftp.listdir(path)
            if pattern:
                return [f for f in files if pattern in f]
            return files
        except Exception as e:
            self.logger.error(f"Error al listar archivos: {str(e)}")
            return []

    def remove_files_time(self, files: List[str], time: int = 30, force: bool = False) -> None:
        """
        Elimina archivos más antiguos que cierto número de días.

        Parameters:
        - files (List[str]): Lista de archivos a evaluar
        - time (int): Días de antigüedad límite
        - force (bool): Si True, elimina sin importar la antigüedad
        """
        now = datetime.now()
        for file in files:
            try:
                stats = self.sftp.stat(file)
                mtime = datetime.fromtimestamp(stats.st_mtime)
                if (now - mtime).days > time or force:
                    self.sftp.remove(file)
                    self.logger.info(f"Archivo eliminado: {file}")
            except Exception as e:
                self.logger.error(f"Error al procesar {file}: {str(e)}")

    def execute_command(self, command: str) -> tuple:
        """
        Ejecuta un comando en el servidor remoto.

        Parameters:
        - command (str): Comando a ejecutar

        Returns:
        - tuple: (stdout, stderr)
        """
        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            return stdout.read().decode(), stderr.read().decode()
        except Exception as e:
            self.logger.error(f"Error al ejecutar comando: {str(e)}")
            return '', str(e)

    def _path_exists(self, path: str) -> bool:
        """
        Verifica si existe una ruta en el servidor.

        Parameters:
        - path (str): Ruta a verificar
        """
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def _ensure_remote_path(self, path: str) -> bool:
        """
        Asegura que existe una ruta en el servidor, creándola si es necesario.

        Parameters:
        - path (str): Ruta a crear
        """
        try:
            current_path = ''
            for part in path.strip('/').split('/'):
                current_path = f"{current_path}/{part}" if current_path else part
                if not self._path_exists(current_path):
                    self.sftp.mkdir(current_path)
            return True
        except Exception as e:
            self.logger.error(f"Error al crear ruta: {str(e)}")
            return False

    def close(self):
        """Cierra las conexiones SSH y SFTP"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        self.logger.info("Conexiones cerradas")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()