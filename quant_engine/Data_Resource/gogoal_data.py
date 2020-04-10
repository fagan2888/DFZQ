import pymssql

class GoGoal_data:
    def __init__(self):
        self.server = '192.168.1.228'
        self.port = '1433'
        self.usr = 'yspzyyx'
        self.pwd = 'y2iaciej'
        self.database = 'FUNDRISKCONTROL'
        conn = pymssql.connect(server=self.server, user=self.usr, password=self.pwd,
                               database=self.database, port=self.port)
        self.cur = conn.cursor()
