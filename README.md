# Risk Analysis

The goal with this repo is to create a program that analysis the risk of a crypto trading account. I will focus on creating it for the futures market given that to me it needs to monitor my algos' risk parameters.

Implemented a database integration to have the historical equity data be analyzed for the ratios and risk parameters. In this case the database is created by another repo and downloaded when needed (I probably pushed it already).

### Account manager file

Add a .env file and its structure should be this:

    001_api_key = "abc"
    001_api_secret = "xyz"
    001_name = "pinco"

    002_api_key = "abc"
    002_api_secret = "xyz"
    002_name = "pallino"

## Live Tracker

This file uses only live informations about an account and is actually more useful consideirng it gives an overview on the account.

## Emergency Selling

Created a program that cleans the bybit account from all open positions; to be used only if you want a tabula rasa.

## Account Info

Get basic info about an account. Useful for initializing a new client without knowing the specifics of the account.

## Last Trades

Prints the last trades taken for a specific account. Used to double check executions.

## Manual Trade

Manually trade on a specific account to modify a postion that is not congruent with freqtrade. In case something wrong happens.
It is not meant to be used regularly so it is not very fancy.

## Bybit connection file

The file is being used only to test new functions.

## VaR Calculation

Calculated the Value at Risk of the portfolio with the following procedure: 
- fetches portfolio's assets historical data;
- calculates assets' weights and portfolio leverage;
- calculates portfolio's standard deviation;
- and the VaR in percentage and monetary value.

At first glance the value appers to be very conservative, it needs further tests.