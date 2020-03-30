import pymssql
import pandas as pd

class IndexCompSQL:
    def __init__(self):
        self.server = '192.168.1.165'
        self.usr = 'dfreader'
        self.pwd = 'dfreader'
        self.database = 'PriceData'

    def get_IndexComp(self,index,start=None,end=None):
        conn = pymssql.connect(server=self.server, user=self.usr, password=self.pwd, database=self.database, charset='utf8')
        cur = conn.cursor()
        sql_sentense = "SELECT [stkcd], [ntime], [weight] " \
                       "FROM [{0}].[dbo].[indexcmpwt{1}] ".format(self.database, index)
        if start and not end:
            sql_sentense += "WHERE [ntime] >= {0} order by [ntime],[stkcd]".format(start)
        elif not start and end:
            sql_sentense += "WHERE [ntime] <= {0} order by [ntime],[stkcd]".format(end)
        elif start and end:
            sql_sentense += "WHERE [ntime] >= {0} and [ntime] <= {1} order by [ntime],[stkcd]".format(start,end)
        else:
            sql_sentense += "order by [ntime],[stkcd]"
        cur.execute(sql_sentense)
        df = pd.DataFrame(cur.fetchall(),columns=['code', 'date', 'weight'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date',inplace=True)
        df['code'] = df.apply(lambda row: row['code']+'.SH' if row['code'][0]=='6' else row['code']+'.SZ',axis=1)

        return df