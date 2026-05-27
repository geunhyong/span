import FinanceDataReader as fdr

df_krx = fdr.StockListing('KRX')

# 삼성전자, 2025년
df = fdr.DataReader("005930", "2025")
df.head(10)
