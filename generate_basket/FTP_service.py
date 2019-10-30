from ftplib import FTP
import os.path

class FTP_service:
    def __init__(self, host, username, password, port=21) -> None:
        self.ftp = FTP()
        # ftp.set_debuglevel(2)         #打开调试级别2，显示详细信息
        self.ftp.connect(host, port)
        self.ftp.login(username, password)  # 登录，如果匿名登录则用空串代替即可
        self.ftp.encoding = 'utf-8'

    def download_file(self, remote_path, local_path):
        buf_size = 4096  # 设置缓冲块大小
        print("downloading: %s -> %s" % (remote_path, local_path))
        fp = open(local_path, 'wb')  # 以写模式在本地打开文件
        self.ftp.retrbinary('RETR ' + remote_path, fp.write, buf_size)  # 接收服务器上文件并写入本地文件
        self.ftp.set_debuglevel(0)  # 关闭调试
        fp.close()  # 关闭文件

    def upload_file(self, remote_path, local_path):
        buf_size = 4096
        fp = open(local_path, 'rb')
        print("uploading: %s -> %s" % (local_path, remote_path))
        self.ftp.storbinary('STOR ' + remote_path, fp, buf_size)
        self.ftp.set_debuglevel(0)
        fp.close()