from ftplib import FTP
from datetime import date, timedelta, datetime
import zipfile

import os.path

'''
使用方法：

下载今天的成分股权重文件到DEFAULT_LOCAL_DIR：      执行python csindex_ftp_down.py 
下载20190305的成分股权重文件到DEFAULT_LOCAL_DIR：  执行python csindex_ftp_down.py --date=20190305
'''

DEFAULT_LOCAL_DIR = "d:/dat1/basket0/downloaded/"


class FtpClient:

    def __init__(self, host, username, password, port=21) -> None:
        self.ftp = FTP()
        self.ftp.encoding = 'utf-8'
        # ftp.set_debuglevel(2)         #打开调试级别2，显示详细信息
        self.ftp.connect(host, port)
        self.ftp.login(username, password)  # 登录，如果匿名登录则用空串代替即可

    def download_file(self, remote_path, local_path):
        buf_size = 4096  # 设置缓冲块大小
        if os.path.exists(local_path):
            print("downloading skipped. %s exists." % local_path)
            return
        print("downloading: %s -> %s" % (remote_path, local_path))
        fp = open(local_path, 'wb')  # 以写模式在本地打开文件
        self.ftp.retrbinary('RETR ' + remote_path, fp.write, buf_size)  # 接收服务器上文件并写入本地文件
        self.ftp.set_debuglevel(0)  # 关闭调试

        fp.close()  # 关闭文件

    def upload_file(self, remote_path, local_path):
        bufsize = 4096  # 设置缓冲块大小
        print("uploading: %s -> %s" % (local_path, remote_path))
        fp = open(local_path, 'rb')
        self.ftp.storbinary('STOR ' + remote_path, fp, bufsize)
        self.ftp.set_debuglevel(0)
        fp.close()

    def exists(self, dir_path: str, file_name: str) -> bool:
        self.ftp.cwd(dir_path)
        return file_name in self.ftp.nlst()


def get_weight_xls_file_name(idx_code: str, date1: date) -> str:
    return idx_code + 'weightnextday' + date1.strftime("%Y%m%d") + ".xls"


class WeightnextdayManager:

    @staticmethod
    def _unzip_all_files(dest_dir: str, zip_file_path: str):
        zfile = zipfile.ZipFile(zip_file_path, 'r')
        print("unzipping %s in %s" % (zip_file_path, dest_dir))
        for filename in zfile.namelist():
            data = zfile.read(filename)
            file = open(dest_dir + filename, 'w+b')
            file.write(data)
            file.close()

    @staticmethod
    def _download_weightnextday_and_unzip(ftp_client: FtpClient, idx_code: str, date1: date, local_dir: str):
        weig_fn = get_weight_xls_file_name(idx_code, date1)
        weig_path_xls_dst = local_dir + '%sweightnextday%s.xls' % (idx_code, date1.strftime("%Y%m%d"))
        weig_fn_zip = '%sweightnextday%s.zip' % (idx_code, date1.strftime("%Y%m%d"))
        weig_path_zip_dst = local_dir + weig_fn_zip
        weig_dir_path = '/idxdata/data/asharedata/%s/weight_for_next_trading_day/' % idx_code
        weig_path_zip_src = weig_dir_path + weig_fn_zip
        if os.path.exists(weig_path_xls_dst):
            print("%s already exists. skipped download zip file and unzip." % weig_path_xls_dst)
            return
        if not ftp_client.exists(weig_dir_path, weig_fn_zip):
            print("[error] %s not exists yet. cannot download!" % weig_path_zip_src)
            return
        ftp_client.download_file(weig_path_zip_src, weig_path_zip_dst)
        WeightnextdayManager._unzip_all_files(local_dir, weig_path_zip_dst)

    @staticmethod
    def _upload_weightnextday_xls(ftp_client: FtpClient, idx_code: str, date1: date, local_dir: str):
        weig_fn = get_weight_xls_file_name(idx_code, date1)
        weig_path_local = local_dir + weig_fn
        weig_path_remote = 'hs300/%sweightnextday%s.xls' % (idx_code, date1.strftime("%Y%m%d"))
        ftp_client.upload_file(weig_path_remote, weig_path_local)

    @staticmethod
    def down_and_up(date_str: str, out_dir: str, need_upload: bool):
        ftp_csindex = FtpClient('ftp.csindex.com.cn', 'csidfzq', '61731595')
        ftp_lan213 = FtpClient('192.168.38.213', 'index', 'dfzq1234')
        os.makedirs(out_dir, exist_ok=True)
        date1 = date.today() + timedelta(days=0)  # default is today
        if date_str:
            dt1 = datetime.strptime(date_str, "%Y%m%d")
            date1 = dt1.date()

        WeightnextdayManager._download_weightnextday_and_unzip(ftp_csindex, '000016', date1, out_dir)
        WeightnextdayManager._download_weightnextday_and_unzip(ftp_csindex, '000300', date1, out_dir)
        WeightnextdayManager._download_weightnextday_and_unzip(ftp_csindex, '000905', date1, out_dir)

        if need_upload:
            WeightnextdayManager._upload_weightnextday_xls(ftp_lan213, '000016', date1, out_dir)
            WeightnextdayManager._upload_weightnextday_xls(ftp_lan213, '000300', date1, out_dir)
            WeightnextdayManager._upload_weightnextday_xls(ftp_lan213, '000905', date1, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='~')
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--upload', type=int, default=0)
    parser.add_argument('--out', type=str, default=DEFAULT_LOCAL_DIR)
    args = parser.parse_args()
    date_s0: str = args.date
    need_upload: bool = args.upload != 0
    out_dir: str = args.out

    WeightnextdayManager.down_and_up(date_s0, out_dir, need_upload)
