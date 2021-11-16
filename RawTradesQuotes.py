# starting from SpreadBars.py & TradeDayBarsSelectTime.py

import pyarrow.feather as feather
import fastparquet as fp

from google.cloud import bigquery
import os
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = None

from datetime import datetime,timedelta

Local_Project = 'qrd1-295412'
PROJECT_ID = 'dbd-sdlc-prod'

# Point GOOGLE_APPLICATION_CREDENTIALS to the credentials key file and initiate the big query client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/big_query/qrd1-295412-8215f94ff78d.json'

client = bigquery.Client(project = Local_Project)
exchange = 'NYQ' # NSQ, NMQ, NAQ, NYQ, PCQ, ASQ, BTQ

# Table schema retrieval, set the config for the job
job_config= bigquery.QueryJobConfig(use_query_cache=False)
query = f'SELECT * FROM {PROJECT_ID}.{exchange}_NORMALISED.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS'
result = client.query(query).to_dataframe(create_bqstorage_client=False)

# Before runing a query it is advisable to do a dry run to check the query cost
# Create the Table that will be queried
table = '{}.{}_NORMALISED.{}_NORMALISED'.format(PROJECT_ID, exchange, exchange)

# Set the big query job config, important to set dry_run=True for dry run to view query size and cost
job_config= bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

startDateTime = datetime(2021, 4, 22, 19, 59, 58) # datetime(2021, 4, 23, 0, 0, 0) # datetime(2021, 4, 22, 23, 59, 59) # datetime(2021, 1, 18, 0, 0, 0) # startDateTime = datetime(2021, 1, 20, 15, 0, 0)
startDateTime = startDateTime.strftime("%Y-%m-%d %H:%M:%S.%f")
endDateTime = datetime(2021, 4, 22, 20, 0, 0) # datetime(2021, 1, 20, 23, 59, 59) # TIMESTAMP('2019-09-04 23:59:59.999999')
endDateTime = endDateTime.strftime("%Y-%m-%d %H:%M:%S.%f")

TRTH_QUERY = f'''
SELECT
  *
FROM
  {table}
WHERE
  # Set this to limit results to a set of RICs
  (RIC LIKE 'AAIC.K')
  # (RIC LIKE 'BB' OR RIC LIKE 'CNQ' OR RIC LIKE 'XOM' OR RIC LIKE 'WFC' OR RIC LIKE 'RDSa' OR RIC LIKE 'TD' OR RIC LIKE 'SLF' OR RIC LIKE 'OXY' OR RIC LIKE 'DVN' OR RIC LIKE 'GM')
  # Set this to determine the partitions
  AND (Date_Time BETWEEN "{startDateTime}" AND "{endDateTime}")
  # (Date_Time BETWEEN "{startDateTime}" AND "{endDateTime}")
  AND Type="Quote"
  AND ((Bid_Price IS NOT NULL AND Bid_Size IS NOT NULL) AND
       (Ask_Price IS NOT NULL AND Ask_Size IS NOT NULL) AND
       (Ask_Price > Bid_Price))
  AND NOT (Qualifiers LIKE 'M[IMB_SIDE]')
  AND NOT (Qualifiers LIKE 'M[ASK_TONE]')
  AND NOT (Qualifiers LIKE 'M[BID_TONE]')
  AND NOT (Qualifiers LIKE '[BID_TONE];[ASK_TONE]')
  AND NOT (Qualifiers LIKE 'A[BID_TONE];A[ASK_TONE]')
'''

query_job = client.query(TRTH_QUERY, job_config=job_config)
print('\nDry run - query size {:,d} MB, query cost {:.2f} USD\n'.format(int(round(query_job.total_bytes_processed / 1024**2)),
                                                                   5.0 * (query_job.total_bytes_processed)/1099511627776))
job_config= bigquery.QueryJobConfig(use_query_cache=False)

##Running the query and streaming to a datafrme
t1 = datetime.now()
result = client.query(TRTH_QUERY, job_config=job_config).to_dataframe(create_bqstorage_client=False)
t2 = datetime.now()

print('\n+++++ run time = ', t2 - t1)

# core_name = './results/QuoteBarsDay_' + exchange + '_.'
core_name = './results/RawQuotes_' + exchange + '_' + str(result['RIC'].shape[0]) + '_.'

result.to_csv(core_name + 'csv', index=False)
feather.write_feather(result, core_name + 'fthr')
fp.write(core_name + 'parq', result)

print('\nwait')
print('about to exit')
