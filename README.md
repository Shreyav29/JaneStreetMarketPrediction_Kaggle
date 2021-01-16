# JaneStreetMarketPrediction_Kaggle
I am participating in the Jane Street Market Prediction Kaggle challenge : https://www.kaggle.com/c/jane-street-market-prediction

# Description 
In this competiton we are given 500 days of historical stock market trading data and the objective is to build a quantitative trading model that maximises the utility. This is the trading data that is given, and each date is represented by numbers from 0 to 500, and each day has thousands of trades at different time stamps given by the ts_id.We also have a weight and return corresponding to each trade. Here resp is nothing but returns of the trade. We also have 130 features related to each trade but we donot know their meaning as their names are masked. 

And finally our aim is to predict the action, whch is to trade or not trade. 

# The returns of the portfolio is defined by this utility score given by Jane Street. 

# EDA 

<img src= "Images/Cumulative Returns and Weighted Returns.png">

Firstly lets look at some EDA plots. So as the data is huge, I have looked at several things during my EDA, but keeping the time in mind, I will be mentioning only some of the interesting ones now. As we go ahead with the model development part, I will discuss more things in detail. 

In the first plot the blue line is the cumulative returns and the orange one is the cumulative weighted returns. The blue line shows that there is an upwards trend in returns 

But when we look at the cumulative weighted returns, its very stable till 300 days mark but falls steadily afterwards till 450 and is increasing again. From this we can see that weights play an important role in calculating the total utility. We will again revisist the importance of weights during the model development. 
