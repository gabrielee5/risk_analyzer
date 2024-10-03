# Risk Analysis

The goal with this project is to create a program that analysis the risk of a crypto trading account. I will focus on creating it for the futures market given that to me it needs to monitor my algos' risk parameters.

The ratios and other calculations are not complete and actually pretty wrong.

Implemented a database integration to have the historical equity data be analyzed for the ratios and risk parameters. In this case the database is created by another repo and downloaded when needed (I probably pushed it already).

## Emergency Selling

Created a program that cleans the bybit account from all open positions; to be used only if you want a tabula rasa.

## Account Info

Get basic info about an account. Useful for initializing a new client without knowing the specifics of the account.

## DB Reader

Converts a database file to a csv file.


### Account manager file

Add a .env file and its structure should be this:

    001_api_key = "abc"
    001_api_secret = "xyz"
    001_name = "pinco"

    002_api_key = "abc"
    002_api_secret = "xyz"
    002_name = "pallino"

### Bybit connection file

The file is being used only to test new functions.