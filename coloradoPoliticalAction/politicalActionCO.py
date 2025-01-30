#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt # pyplot is the state-based interface to matplotlib
import requests
pd.options.display.max_columns = 200


# this reads in all of the protest data and creates a dataframe from it
# data as of april 12th 2023
protestDF = pd.read_csv("ccc_compiled.csv")
protestDF


# changes fips codes to ints from floats
# for any of the ones that were empty I gave them 0s, this does not change the quality of the data because even
# if the data had 0s originally, it would still not be useful data since the fips codes have to be 5 digits
# so if it has a 0 it is unusable. if any code has fewer than 5 digits it is should just be missing a leading 0
protestDF["fips_code"] = protestDF["fips_code"].fillna(0).astype(int)

# adds a column of just the month and year so that dataframes can be grouped by their month of occurance
protestDF["date"] = pd.to_datetime(protestDF["date"])
# explain strftime
protestDF["monthPerYear"] = protestDF["date"].dt.strftime("%m-%Y")

# protest column is made so that we can count the number of protests accurately
# setting the value to 1 means each protest has a value of 1 protest
protestDF["protest"] = 1
protestDF


# In[4]:


# calculates the average size of protests in the entire US for every month
protestUSDateDF = protestDF.groupby("monthPerYear",sort=False).agg({"size_mean":np.mean}).reset_index()
protestUSDateDF


# average size of protests in US per month from 01/2017-04/2023
# I used a line plot because I am trying to see trends 
USMSPMPlot = sb.relplot(x="monthPerYear",y="size_mean",aspect=3,data=protestUSDateDF,kind="line",
                # I changed the labels to make it more readable 
        color="DarkGrey").set_xticklabels(rotation=90).set_ylabels("Mean Size of Protests").set_xlabels("Month"
                # this allows me to set the title of the plot
                    ).fig.suptitle("Average Size of Protests in the US per Month")
plt.show()


# calculates the average size of protests in CO for every month  
protestCOMeanSizeDF = protestDF.groupby("state").get_group("CO").groupby("monthPerYear",sort=False).agg(
    {"size_mean":np.mean}).reset_index()
protestCOMeanSizeDF


# average size of protests in CO per month from 01/2017-04/2023
# I used a line plot because I am trying to see trends 
COMSPMPlot = sb.relplot(x="monthPerYear",y="size_mean",aspect=3,
    data=protestCOMeanSizeDF,kind="line",color="DarkGrey").set_xticklabels(rotation=90
                # I changed the labels to make it more readable 
    ).set_ylabels("Mean Size of Protests").set_xlabels("Month"
                # this allows me to set the title of the plot
                    ).fig.suptitle("Average Size of Protests in Colorado per Month")
plt.show()


# calculates the number of protests in US for every month 
protestUSSumDF = protestDF.groupby("monthPerYear",sort=False).count().reset_index()
protestUSSumDF


# number of protests per month in US per month from 01/2017-04/2023
# I used a line plot because I am trying to see trends 
USNPMPlot = sb.relplot(x="monthPerYear",y="protest",data=protestUSSumDF,kind="line",aspect=3,
                # I changed the labels to make it more readable 
        color="DarkGrey").set_xticklabels(rotation=90).set_ylabels("Number of Protests").set_xlabels("Month"
                # this allows me to set the title of the plot
                    ).fig.suptitle("Number of Protests per Month in the US")
plt.show()


# calculates the number of protests in CO for every month 
# we remove the month of january from 2022 because it is statistically unusual in that it has a ton of activity 
# because of there being a lot of strikes for king soopers and not directly protest related
# it also makes it's difficult to read the data (further explanation below)
protestCOSumDF = protestDF.groupby("state").get_group("CO").groupby("monthPerYear",
        sort=False).count().reset_index().drop(labels=60,axis=0)
protestCOSumDF


# number of protests per month in CO per month from 01/2017-04/2023
# I used a line plot because I am trying to see trends 
CONPMPlot = sb.relplot(x="monthPerYear",y="protest",data=protestCOSumDF,kind="line",aspect=3,
                # I changed the labels to make it more readable 
        color="DarkGrey").set_xticklabels(rotation=90).set_ylabels("Number of Protests").set_xlabels("Month"
                # this allows me to set the title of the plot
                    ).fig.suptitle("Number of Protests per Month in Colorado")
plt.show()


# The number of protests per political offiliation per month in US
# 0 for unafilliated, 1 for left, and 2 for right 
protestUSPMPVDF = protestDF.groupby(["monthPerYear",
# we drop line 161 as it holds the data for the unusually large number of pritests for the unaffiliated group
# this was because there were a bunch of king soopers strikes, it is certainly an outlier as well as makes the
# rest of the data hard to see for the chart
        "valence"], sort = False).agg({"protest":sum}).reset_index().drop(labels=161,axis=0)
protestUSPMPVDF


# shows the number of protests for left leaning protests 1, right leaning 2, or neither 0, in US
USPPRBVPlot = sb.catplot(x="monthPerYear",y="protest",data=protestUSPMPVDF,hue="valence",
                # changed the labels to make it more readable
           kind="bar",aspect=3).set_xticklabels(rotation=90).set_ylabels("Number of Protests"
                ).set_xlabels("Month"
                # this allows me to set the title of the plot
                ).fig.suptitle("Number of Protests per Political Affiliation per Month in the US")
plt.show()


# The number of protests per political offiliation per month in CO
# 0 for unafilliated, 1 for left, and 2 for right 
protestCOPMPVDF = protestDF.groupby("state").get_group("CO").groupby(["monthPerYear",
# we drop line 161 as it holds the data for the unusually large number of pritests for the unaffiliated group
# this was because there were a bunch of king soopers strikes, it is certainly an outlier as well as makes the
# rest of the data hard to see for the chart
        "valence"], sort = False).agg({"protest":sum}).reset_index().drop(labels=161,axis=0)
protestCOPMPVDF


# shows the number of protests for left leaning protests 1, right leaning 2, or neither 0, in CO
COPPRBVPlot = sb.catplot(x="monthPerYear",y="protest",data=protestCOPMPVDF,hue="valence",
                # changed the labels to make it more readable
           kind="bar",aspect=3).set_xticklabels(rotation=90).set_ylabels("Number of Protests"
                ).set_xlabels("Month"
                # this allows me to set the title of the plot
                    ).fig.suptitle("Number of Protests per Political Affiliation per Month in Colorado")
plt.show()


# this was to figure out and explain why there was a spike in protests in january of 2022
protestDF.groupby("state").get_group("CO").groupby("monthPerYear"
# it was clear the spike was for unaffiliated so that was added to narrow it down to find the reason
                                    ).get_group("01-2022").query("valence == 0.0")
# from this it is clear that they were strikes of kind soopers


# this function takes two lists of lists and creates a dictionary from them
# the reason there is two is because one of the lists is supposed to be the under 18 population data and the 
# other is supposed to be the total population data and so by subtracting the under 18 from the total
# we get the number of people who who are 18 and older ie people of voting age
# then from the created dictionary the function creates a data frame

# the reason for making this function is so that I can use several different years data and do the same for each 
# of those years
def createCountyPopDF(under18Query,byAgeQuery):
    adjustedPop = {}
    # loops through the total population data
    for i in range(1,len(byAgeQuery)):
        # loops through the under18 population data 
        for j in range(1,len(under18Query)):
# loop through both to get this check, that makes sure we are pairing properly and getting acurate values
            if((byAgeQuery[i][2]+byAgeQuery[i][3])==(under18Query[j][2]+under18Query[j][3])):
                pop = int(byAgeQuery[i][1]) - int(under18Query[j][1])
                adjustedPop[byAgeQuery[i][0].split(" County")[0]] = [
                    int(byAgeQuery[i][2]+byAgeQuery[i][3]),pop]
                break
                    
    # now convert to data frame
    adjustedPop = pd.DataFrame.from_dict(adjustedPop,orient="index").reset_index()
    adjustedPop.columns=["County","FipsCode","EstOver18Pop"]
    return adjustedPop


# this function creates a data frame that is the merged data of a county info data frame and an election data 
# frame from the same year
def createCountyInfoDF(countyPopDF, electionDF):
    countyInfoDF = pd.merge(countyPopDF, electionDF, on="County")
    return countyInfoDF


# Any dropped rows in the election data were totals so they were removed and any dropped columns were 
# unnecisary data to what I was trying to answer. I renamed some columns so that they would be the same across
# the board and could then be therefore merged
# I added a year column to all final versions of that year's dataframe so that it could be used in the graphs
# If any year does not have either the voting data or the Census datat there was just not available data for 
# that year


# this read in and created a dataframe from election results of the 2010 general election
# was not given registered voter data
electionCO2010DF = pd.read_csv(
"State_of_Colorado_Elections_Database__2010_Nov_2_General_Election_Voting_Statistics_State_of_Colorado.csv").drop(
    labels=[0,65],axis=0).drop(labels=["Active Voters"], axis=1)
electionCO2010DF.columns=["County","Votes"]
electionCO2010DF["Year"] = 2010
# the lambda function takes the string and replaces the comma with nothin ie taking it out and then can convert
# that new string to an int as there are new commas and this gets applied to every line in the votes
electionCO2010DF["Votes"] = electionCO2010DF["Votes"].apply(lambda x:int(x.replace(",","")))
electionCO2010DF


# this read in and created a dataframe from election results of the 2012 general election
# Note: The election from the original source document contained data entry errors that have been preserved.
electionCO2012DF = pd.read_csv(
"State_of_Colorado_Elections_Database__2012_Nov_6_General_Election_Voting_Statistics_State_of_Colorado.csv").drop(
    labels=[0,65],axis=0).drop(labels=["Active Voters","Inactive Voters"], axis=1)
electionCO2012DF.columns=["County","RegisteredVoters","Votes"]
electionCO2012DF["Year"] = 2012
# the lambda function takes the string and replaces the comma with nothin ie taking it out and then can convert
# that new string to an int as there are new commas and this gets applied to every line in the votes
electionCO2012DF["Votes"] = electionCO2012DF["Votes"].apply(lambda x:int(x.replace(",","")))
# this is done the same as above but to the registed voters column
electionCO2012DF["RegisteredVoters"] = electionCO2012DF["RegisteredVoters"].apply(lambda x:int(x.replace(",","")))
electionCO2012DF


# the data ignores counties smaller than 20,000
key = "10f4bcd7cb690c629f423fac1a31ec61dce6b2ed"
year = "2014"
# K200104_001E is the total population of that county
infoToGet = "NAME,K200104_001E"
state = "08"
rootURL = "https://api.census.gov/data/{0}/acs/acsse?get={1}&for=county:*&in=state:{2}&key={3}"
censusCOCountyPopQuery14 = requests.get(rootURL.format(year,infoToGet,state,key)).json()
censusCOCountyPopQuery14


# K200104_002E is under 18
infoToGetQ2 = "NAME,K200104_002E"
year = "2014"
censusCOCountyUnder18Query14 = requests.get(rootURL.format(year,infoToGetQ2,state,key)).json()
censusCOCountyUnder18Query14


fipsCodesCO2014DF = createCountyPopDF(censusCOCountyUnder18Query14,censusCOCountyPopQuery14)
fipsCodesCO2014DF


# this read in and created a dataframe from election results of the 2014 general election
# Note: The election from the original source document contained data entry errors that have been preserved.
electionCO2014DF = pd.read_csv(
"State_of_Colorado_Elections_Database__2014_Nov_4_General_Election_Voting_Statistics_State_of_Colorado.csv").drop(
    labels=[0,65],axis=0).drop(labels=["Active Voters","Inactive Voters"], axis=1)
electionCO2014DF.columns=["County","RegisteredVoters","Votes"]
electionCO2014DF["Year"] = 2014
# the lambda function takes the string and replaces the comma with nothin ie taking it out and then can convert
# that new string to an int as there are new commas and this gets applied to every line in the votes
electionCO2014DF["Votes"] = electionCO2014DF["Votes"].apply(lambda x:int(x.replace(",","")))
# this is done the same as above but to the registed voters column
electionCO2014DF["RegisteredVoters"] = electionCO2014DF["RegisteredVoters"].apply(lambda x:int(x.replace(",","")))
electionCO2014DF


COCountyInfo2014DF = createCountyInfoDF(fipsCodesCO2014DF, electionCO2014DF)
COCountyInfo2014DF


# the data ignores counties smaller than 20,000
key = "10f4bcd7cb690c629f423fac1a31ec61dce6b2ed"
year = "2015"
# K200104_001E is the total population of that county
infoToGet = "NAME,K200104_001E"
state = "08"
rootURL = "https://api.census.gov/data/{0}/acs/acsse?get={1}&for=county:*&in=state:{2}&key={3}"
censusCOCountyPopQuery15 = requests.get(rootURL.format(year,infoToGet,state,key)).json()
censusCOCountyPopQuery15


# K200104_002E is under 18
infoToGetQ2 = "NAME,K200104_002E"
year = "2015"
censusCOCountyUnder18Query15 = requests.get(rootURL.format(year,infoToGetQ2,state,key)).json()
censusCOCountyUnder18Query15


fipsCodesCO2015DF = createCountyPopDF(censusCOCountyUnder18Query15,censusCOCountyPopQuery15)
fipsCodesCO2015DF["Year"] = 2015
fipsCodesCO2015DF


# the data ignores counties smaller than 20,000
key = "10f4bcd7cb690c629f423fac1a31ec61dce6b2ed"
year = "2016"
# K200104_001E is the total population of that county
infoToGet = "NAME,K200104_001E"
state = "08"
rootURL = "https://api.census.gov/data/{0}/acs/acsse?get={1}&for=county:*&in=state:{2}&key={3}"
censusCOCountyPopQuery16 = requests.get(rootURL.format(year,infoToGet,state,key)).json()
censusCOCountyPopQuery16


# K200104_002E is under 18
infoToGetQ2 = "NAME,K200104_002E"
year = "2016"
censusCOCountyUnder18Query16 = requests.get(rootURL.format(year,infoToGetQ2,state,key)).json()
censusCOCountyUnder18Query16


fipsCodesCO2016DF = createCountyPopDF(censusCOCountyUnder18Query16,censusCOCountyPopQuery16)
fipsCodesCO2016DF


# this read in and created a dataframe from election results of the 2016 general election
# Note: The election from the original source document contained data entry errors that have been preserved.
electionCO2016DF = pd.read_csv(
"State_of_Colorado_Elections_Database__2016_Nov_8_General_Election_Voting_Statistics_State_of_Colorado.csv").drop(
    labels=[0,65],axis=0).drop(labels=["Active Voters","Inactive Voters"], axis=1)
electionCO2016DF.columns=["County","RegisteredVoters","Votes"]
electionCO2016DF["Year"] = 2016
# the lambda function takes the string and replaces the comma with nothin ie taking it out and then can convert
# that new string to an int as there are new commas and this gets applied to every line in the votes
electionCO2016DF["Votes"] = electionCO2016DF["Votes"].apply(lambda x:int(x.replace(",","")))
# this is done the same as above but to the registed voters column
electionCO2016DF["RegisteredVoters"] = electionCO2016DF["RegisteredVoters"].apply(lambda x:int(x.replace(",","")))
electionCO2016DF


COCountyInfo2016DF = createCountyInfoDF(fipsCodesCO2016DF, electionCO2016DF)
COCountyInfo2016DF


# the data ignores counties smaller than 20,000
key = "10f4bcd7cb690c629f423fac1a31ec61dce6b2ed"
year = "2017"
# K200104_001E is the total population of that county
infoToGet = "NAME,K200104_001E"
state = "08"
rootURL = "https://api.census.gov/data/{0}/acs/acsse?get={1}&for=county:*&in=state:{2}&key={3}"
censusCOCountyPopQuery17 = requests.get(rootURL.format(year,infoToGet,state,key)).json()
censusCOCountyPopQuery17


# K200104_002E is under 18
infoToGetQ2 = "NAME,K200104_002E"
year = "2017"
censusCOCountyUnder18Query17 = requests.get(rootURL.format(year,infoToGetQ2,state,key)).json()
censusCOCountyUnder18Query17


fipsCodesCO2017DF = createCountyPopDF(censusCOCountyUnder18Query17,censusCOCountyPopQuery17)
fipsCodesCO2017DF["Year"] = 2017
fipsCodesCO2017DF


# the data ignores counties smaller than 20,000
key = "10f4bcd7cb690c629f423fac1a31ec61dce6b2ed"
year = "2018"
# K200104_001E is the total population of that county
infoToGet = "NAME,K200104_001E"
state = "08"
rootURL = "https://api.census.gov/data/{0}/acs/acsse?get={1}&for=county:*&in=state:{2}&key={3}"
censusCOCountyPopQuery18 = requests.get(rootURL.format(year,infoToGet,state,key)).json()
censusCOCountyPopQuery18


# K200104_002E is under 18
infoToGetQ2 = "NAME,K200104_002E"
year = "2018"
censusCOCountyUnder18Query18 = requests.get(rootURL.format(year,infoToGetQ2,state,key)).json()
censusCOCountyUnder18Query18


fipsCodesCO2018DF = createCountyPopDF(censusCOCountyUnder18Query18,censusCOCountyPopQuery18)
fipsCodesCO2018DF


# this read in and created a dataframe from election results of the 2018 general election
electionCO2018DF = pd.read_csv(
"State_of_Colorado_Elections_Database__2018_Nov_6_General_Election_Voting_Statistics_State_of_Colorado.csv").drop(
    labels=[0,65],axis=0).drop(labels=["Active Voters","Inactive Voters"], axis=1)
electionCO2018DF.columns=["County","RegisteredVoters","Votes"]
electionCO2018DF["Year"] = 2018
# the lambda function takes the string and replaces the comma with nothin ie taking it out and then can convert
# that new string to an int as there are new commas and this gets applied to every line in the votes
electionCO2018DF["Votes"] = electionCO2018DF["Votes"].apply(lambda x:int(x.replace(",","")))
# this is done the same as above but to the registed voters column
electionCO2018DF["RegisteredVoters"] = electionCO2018DF["RegisteredVoters"].apply(lambda x:int(x.replace(",","")))
electionCO2018DF


COCountyInfo2018DF = createCountyInfoDF(fipsCodesCO2018DF, electionCO2018DF)
COCountyInfo2018DF


# the data ignores counties smaller than 20,000
key = "10f4bcd7cb690c629f423fac1a31ec61dce6b2ed"
year = "2019"
# K200104_001E is the total population of that county
infoToGet = "NAME,K200104_001E"
state = "08"
rootURL = "https://api.census.gov/data/{0}/acs/acsse?get={1}&for=county:*&in=state:{2}&key={3}"
censusCOCountyPopQuery19 = requests.get(rootURL.format(year,infoToGet,state,key)).json()
censusCOCountyPopQuery19


# K200104_002E is under 18
infoToGetQ2 = "NAME,K200104_002E"
year = "2019"
censusCOCountyUnder18Query19 = requests.get(rootURL.format(year,infoToGetQ2,state,key)).json()
censusCOCountyUnder18Query19


fipsCodesCO2019DF = createCountyPopDF(censusCOCountyUnder18Query19,censusCOCountyPopQuery19)
fipsCodesCO2019DF["Year"] = 2019
fipsCodesCO2019DF


# The census does not have any population data for 2020, they say due to things aroud covid


# this read in and created a dataframe from election results of the 2020 general election
electionCO2020DF = pd.read_csv(
"State_of_Colorado_Elections_Database__2020_Nov_3_General_Election_Voting_Statistics_State_of_Colorado.csv").drop(
    labels=[0,65],axis=0).drop(labels=["Active Voters","Inactive Voters"], axis=1)
electionCO2020DF.columns=["County","RegisteredVoters","Votes"]
electionCO2020DF["Year"] = 2020
# the lambda function takes the string and replaces the comma with nothin ie taking it out and then can convert
# that new string to an int as there are new commas and this gets applied to every line in the votes
electionCO2020DF["Votes"] = electionCO2020DF["Votes"].apply(lambda x:int(x.replace(",","")))
# this is done the same as above but to the registed voters column
electionCO2020DF["RegisteredVoters"] = electionCO2020DF["RegisteredVoters"].apply(lambda x:int(x.replace(",","")))
electionCO2020DF


# the data ignores counties smaller than 20,000
key = "10f4bcd7cb690c629f423fac1a31ec61dce6b2ed"
year = "2021"
# K200104_001E is the total population of that county
infoToGet = "NAME,K200104_001E"
state = "08"
rootURL = "https://api.census.gov/data/{0}/acs/acsse?get={1}&for=county:*&in=state:{2}&key={3}"
censusCOCountyPopQuery21 = requests.get(rootURL.format(year,infoToGet,state,key)).json()
censusCOCountyPopQuery21


# K200104_002E is under 18
infoToGetQ2 = "NAME,K200104_002E"
year = "2021"
censusCOCountyUnder18Query21 = requests.get(rootURL.format(year,infoToGetQ2,state,key)).json()
censusCOCountyUnder18Query21


fipsCodesCO2021DF = createCountyPopDF(censusCOCountyUnder18Query21,censusCOCountyPopQuery21)
fipsCodesCO2021DF["Year"] = 2021
fipsCodesCO2021DF


# this read in and created a dataframe from election results of the 2022 general election
# was not given registered voter data
electionCO2022DF = pd.read_csv(
"State_of_Colorado_Elections_Database__2022_Nov_8_General_Election_Voting_Statistics_State_of_Colorado.csv").drop(
    labels=[0,65],axis=0).drop(labels="Active Voters", axis=1)
electionCO2022DF.columns=["County","Votes"]
electionCO2022DF["Year"] = 2022
# the lambda function takes the string and replaces the comma with nothin ie taking it out and then can convert
# that new string to an int as there are new commas and this gets applied to every line in the votes
electionCO2022DF["Votes"] = electionCO2022DF["Votes"].apply(lambda x:int(x.replace(",","")))
# this is done the same as above but to the registed voters column
electionCO2022DF


# this adds together all of the year dataframes and does it in cronological order 
allYearDF = pd.concat([electionCO2010DF,electionCO2012DF,COCountyInfo2014DF,fipsCodesCO2015DF,COCountyInfo2016DF,
                       fipsCodesCO2017DF,COCountyInfo2018DF,fipsCodesCO2019DF,electionCO2020DF,fipsCodesCO2021DF,
                       electionCO2022DF],ignore_index=True)
allYearDF


# shows the number of votes from 2010-2022 for general elections in CO
voting = allYearDF.groupby("Year").agg({"Votes":sum}).drop(labels=[2015,2017,2019,2021],axis=0).reset_index()
sb.relplot(x="Year",y="Votes",data=voting,kind="line",aspect=3).set_xticklabels(rotation=90
                # this allows me to set the title of the plot
                    ).fig.suptitle("Number of votes per Election in Colorado")
plt.show()


# shows the population estimate of people over 18 from 2014-2021 in CO
# 2010, 2012, 2020, and 2022 are all dropped because there is no data for those years
# 2010 and 2012 because it appears to be before the api started or started amassing data,
# 2020 was noted to be due to covid-19, and 2022 I would assume because it is not yet ready
population = allYearDF.groupby("Year").agg({"EstOver18Pop":sum}).drop(labels=[2010,2012,2020,2022],axis=0
            ).reset_index()
sb.relplot(x="Year",y="EstOver18Pop",data=population,kind="line",aspect=3).set_xticklabels(rotation=90
                # this allows me to set the title of the plot
                    ).fig.suptitle("Voting Age (Over 18) Population in Colorado")
plt.show()