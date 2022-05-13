# CREATING THE DATABASE
Since scraping long historical data required very long time and for the objective of the ***reproducibility*** we decided to make the database into two steps, the `FIRST STEP` is shown below just to show how did we get the data, but in order to reproduce the work, one can only start from `SECOND STEP` immediately which takes just several minutes.

## FIRST STEP:

***Here you need to use "eco395_final_project/code/sql_setup/sql_code_one.ipynb" and "eco395_final_project/code/data_one"***

**Reminder**:No need to do it! we already did it for you and you can start from the `SECOND STEP` to get your database ready!

In this part you will get a list of all the S&P500 stock tickers using the Pandas `read_html` function to scrape Wikipedia as shown below:
```python
table=pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
df = table[0]
stockticker = df["Symbol"].to_list()
```
By running the following cell you will write the tickers into a txt file and save that in the`data_one` folder with name `tickerlist.txt`. The objective of this step is to allow updating the S&P500 list since some of the stocks have been removed. Also the table in Wikipedia uses "." instead "-" that is used in Yahoo finance tickers which we are going to scrape too. 
```python
with open("../data_one/tickerlist.txt", "w+") as out_file:
    for ticker in stockticker:
        out_file.write("%s\n" %ticker)
```
The folder `data_two` has the updated txt tickers file which is the one used to scrape Yahoo finance here. There are four ticker have been fixed in this new list those are: "BF.B", "BRK.B", "CEG", and "OGN".
```python
with open("../data_two/tickerlist.txt", "r") as in_file:
    clean_tickers = in_file.read()
```
The following part will download all the historical data from 2019-04-29 to 2022-04-29 for each variable alone. **Note**: This step might take very long time!
```python
adj_close = yf.download(clean_tickers, "2019-04-29", "2022-04-29")["Adj Close"]
close_price = yf.download(clean_tickers, "2019-04-29", "2022-04-29")["Close"]
high_price = yf.download(clean_tickers, "2019-04-29", "2022-04-29")["High"]
low_price = yf.download(clean_tickers, "2019-04-29", "2022-04-29")["Low"]
open_price = yf.download(clean_tickers, "2019-04-29", "2022-04-29")["Open"]
volume = yf.download(clean_tickers, "2019-04-29", "2022-04-29")["Volume"]
```
This part will write the imported dataframe to csv files to be used in the `SECOND STEP`.
```python
adj_close.to_csv("../data_one/adj_close.csv")
close_price.to_csv("../data_one/close_price.csv")
high_price.to_csv("../data_one/high_price.csv")
low_price.to_csv("../data_one/low_price.csv")
open_price.to_csv("../data_one/open_price.csv")
volume.to_csv("../data_one/volume.csv")
```
## SECOND STEP:

***Here you need to use "eco395_final_project/code/sql_setup/sql_code_two.ipynb" and "eco395_final_project/code/data_two"***

You might want to use GCP SQL to create a database called `Project` and then connect the database with DBeaver (or another SQL client of your choice). After that, There is a file called `demo.env` in "eco395_final_project/code/demo.env" which you need to connect to the database. Modify the file `demo.env` by providing the right credentials and then change its name to`.env`.

Now, go to the file `sql_code_two`, "code/sql_setup/sql_code_two.ipynb". After runnig the first block of the code to import required libraries, you will get the data produced in `FIRST STEP` through reading the csv files from the folder`data_two`by running the second block shown below:
```python
def import_csv_as_dataframe(variable):
    ''' Takes the variable name from [adj_close, close_price, high_price, low_price, open_price, volume] and return a dataframe for all S&P500 '''
    file_path = os.path.join("../data_two", variable+".csv")
    df = pd.read_csv(file_path)       
    return df
```
Then, you may call the previous function on the required csv files as follows:
```python
adj_close = import_csv_as_dataframe("adj_close")
close_price = import_csv_as_dataframe("close_price")
high_price = import_csv_as_dataframe('high_price')
low_price = import_csv_as_dataframe("low_price")
open_price = import_csv_as_dataframe("open_price")
volume = import_csv_as_dataframe("volume")
```
After that, you will run the block that has the schema primarily to make the date type. The complete schema is shown in the original code.
```python
schema = {
    "Date": Date,
    "A": Numeric,
    "AAL": Numeric,
    "AAP": Numeric,
    "AAPL": Numeric,
    .
    .
    .
```
Finally, you will use the following commands to send the dataframes to the database using `SQLAlchemy`.
**Warning**: The best practice here is to run the code line by line, and after each line you need to update your SQL database. Running the following lines all at once might cause a failure that might turn to be hard to fix.

```python
open_price.to_sql("open_price", engine, if_exists="replace", dtype=schema, index=False)
high_price.to_sql("high_price", engine, if_exists="replace", dtype=schema, index=False)
low_price.to_sql("low_price", engine, if_exists="replace", dtype=schema, index=False)
close_price.to_sql("close_price", engine, if_exists="replace", dtype=schema, index=False)
adj_close.to_sql("adj_close", engine, if_exists="replace", dtype=schema, index=False)
volume.to_sql("volume", engine, if_exists="replace", dtype=schema, index=False)
```
