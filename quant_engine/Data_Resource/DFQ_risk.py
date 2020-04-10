import pymysql

class dfq_risk:
    def __init__(self):
        self.host = '139.196.77.199'
        self.port = 81
        self.usr = 'guli'
        self.pwd = 'dfquant'
        self.database = 'dfrisk'
        conn = pymysql.connect(host=self.host, user=self.usr, password=self.pwd, port=self.port, db=self.database)
        self.cur = conn.cursor()