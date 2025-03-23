from ftplib import FTP
import os
from datetime import datetime

ftp = FTP('192.168.1.1')
ftp.login(user='admin', passwd='admin')
ftp.sendcmd('TYPE I')  # Установка бинарного режима

local_folder = 'C:/Users/pniki/Pictures'
remote_folder = '/NO NAME/usb_share'

def get_remote_files(ftp):
    files = {}
    try:
        for name, meta in ftp.mlsd():
            if name not in ('.', '..'):  # Игнорируем системные папки
                files[name] = meta
    except Exception as e:
        print(f"Ошибка при получении списка файлов: {e}")
    return files


def sanitize_filename(name):
    try:
        return name.encode('utf-8', 'ignore').decode('utf-8')
    except Exception as e:
        print(f"Ошибка при обработке имени файла {name}: {e}")
        return "invalid_filename"


def sync_from_ftp(ftp, remote_folder, local_folder):
    os.makedirs(local_folder, exist_ok=True)
    try:
        ftp.cwd(remote_folder)
        remote_files = get_remote_files(ftp)

        for name, meta in remote_files.items():
            name = sanitize_filename(name)
            local_path = os.path.join(local_folder, name)

            if meta['type'] == 'file':
                remote_mtime = datetime.strptime(meta.get('modify', '19700101000000'), '%Y%m%d%H%M%S')
                if not os.path.exists(local_path) or datetime.fromtimestamp(os.path.getmtime(local_path)) < remote_mtime:
                    print(f"Скачиваю файл: {name}")
                    try:
                        with open(local_path, 'wb') as f:
                            ftp.retrbinary(f'RETR {name}', f.write)
                    except Exception as e:
                        print(f"Ошибка при скачивании файла {name}: {e}")
            elif meta['type'] == 'dir':
                try:
                    sync_from_ftp(ftp, f"{remote_folder}/{name}", local_path)
                except Exception as e:
                    print(f"Ошибка при навигации по папке {remote_folder}/{name}: {e}")

        for name in os.listdir(local_folder):
            if name not in remote_files:
                local_path = os.path.join(local_folder, name)
                if os.path.isdir(local_path):
                    try:
                        os.rmdir(local_path)
                        print(f"Удалена папка: {name}")
                    except Exception as e:
                        print(f"Ошибка при удалении папки {name}: {e}")
                elif os.path.isfile(local_path):
                    try:
                        os.remove(local_path)
                        print(f"Удалён файл: {name}")
                    except Exception as e:
                        print(f"Ошибка при удалении файла {name}: {e}")
    except Exception as e:
        print(f"Ошибка при навигации по папке {remote_folder}: {e}")


def sync_to_ftp(ftp, local_folder, remote_folder):
    try:
        try:
            ftp.cwd(remote_folder)
        except Exception:
            ftp.mkd(remote_folder)
            ftp.cwd(remote_folder)

        remote_files = get_remote_files(ftp)

        for name in os.listdir(local_folder):
            name = sanitize_filename(name)
            local_path = os.path.join(local_folder, name)

            if os.path.isdir(local_path):
                if name not in remote_files:
                    try:
                        ftp.mkd(name)
                    except Exception as e:
                        print(f"Ошибка при создании папки {name}: {e}")
                try:
                    sync_to_ftp(ftp, local_path, f"{remote_folder}/{name}")
                except Exception as e:
                    print(f"Ошибка при навигации по папке {remote_folder}/{name}: {e}")
            elif os.path.isfile(local_path):
                local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path))
                remote_mtime = datetime.strptime(remote_files.get(name, {}).get('modify', '19700101000000'), '%Y%m%d%H%M%S')
                if name not in remote_files or local_mtime > remote_mtime:
                    print(f"Загружаю файл: {name}")
                    try:
                        with open(local_path, 'rb') as f:
                            ftp.storbinary(f'STOR {name}', f)
                    except Exception as e:
                        print(f"Ошибка при загрузке файла {name}: {e}")

        for name in remote_files:
            if name not in os.listdir(local_folder) and name not in ('.', '..'):
                try:
                    ftp.delete(name)
                    print(f"Удалён файл с FTP: {name}")
                except Exception as e:
                    print(f"Ошибка при удалении файла с FTP {name}: {e}")
    except Exception as e:
        print(f"Ошибка при навигации по папке {remote_folder}: {e}")

try:
    sync_to_ftp(ftp, local_folder, remote_folder)
    sync_from_ftp(ftp, remote_folder, local_folder)
    
except Exception as e:
    print(f"Общая ошибка при синхронизации: {e}")

ftp.quit()
print('Полная синхронизация завершена!')
