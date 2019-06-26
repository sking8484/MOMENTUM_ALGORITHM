import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
plt.style.use('seaborn')



class RollingStdAlgo:



    def __init__(self, stock_list, train_test_split=False ):
        self.stock_list = stock_list
        self.train_test_split = train_test_split

        '''
           ROLLING STANDARD DEVIATION MAX WITH SHORT LONG MOVING AVERAGE TRADING STRATEGY.

           STOCK_LIST TAKES A LIST OF STOCKS. IF JUST ONE STOCK STILL PROVIDE LIST
           IF YOU WOULD LIKE TO RANDOMIZE AND TEST AGAINST DIFFERENT TIME, SET train_test_split = True



           1.) CALCULATE RETURNS OF THE STOCK, LOG of (RETURNS / RETURNS.SHIFT(1))
           2.) CALCULATE THE EXPONENTIAL MOVING STANDARD DEVIATION OF SPAN = 10 AdjClose PRICE
           3.) NORMALIZE THIS PREVIOUS NUMBER
           4.) CALCULATE THE SHORT MOVING AVERAGE OF AdjClose PRICES
           5.) CALCULATE THE LONG MOVING AVERAGE OF AdjClose PRICES
           6.) CALCULATE THE MAXIMUM OF THE WINDOW OF THE STANDARDIZED STANDARD DEVIATION
           7.) FIND THE CUMMULATIVE MAX STANDARD DEVIATION OF THE FIRST STANDARD DEVIATION ROLLING WINDOW
           8.) DEFINE THE TRADING SIGNALS
               a.) IF THE SHORT MOVING AVERAGE OF AdjClose PRICES IS ABOVE THE LONG MOVING AVERAGE OF THE AdjClose PRICES, SIGNAL IS 1
               ELSE SIGNAL IS -1

               b.) IF THE LONG ROLLING STD MAX IS EQUAL TO THE CUMULATIVE ROLLING STD MAX, SIGNAL = 1, ELSE SIGNAL = 0
           9.) FIND TRADE DIRECTION, MULTIPLY THE TRADE SIGNALS TOGETHOR
           10.) FIND THE STRATEGY RETURNS
               a.) MULTIPY THE TRADE DIRECTION BY THE DAY TO DAY RETURNS
               b.) CALCULATE CUMULATIVE SUM OF THE STRATEGY RETURNS
           11.) OPTIMIZE STRATEGY
               a.) RUN FOR LOOPS OF PRE SET INTERVALS FOR EACH POSSIBLE SIGNAL, OPTIMIZE BASED ON FINAL DAYS RETURNS
               b.) RUN TRAIN TEST SPLIT, OPTIMIZE ON DIFFERENT TIME PERIODS, SPLIT THIS DATA IN HALF,
               c.) USE OPTIMIZEDPARAMETERS ON SECOND HALF OF DATA

           12.) SAVE DATA!!!!!!!!!!!


       '''


    def run_algo(self):

        stocks = self.stock_list
        for stock in stocks:
            overall_returns = {}
            stock = stock
            aapl_df_beginning  = web.DataReader(stock.upper(), 'quandl', start = '2014-01-01')
            aapl_df_beginning.index = pd.to_datetime(aapl_df_beginning.index)
            aapl_df_beginning = aapl_df_beginning.sort_index()

            for x in range(1):
                returns_dict = {}

                """

                COMMENT ILOC HERE TO FIND OVERALL RETURNS

                """
                if not self.train_test_split:
                    aapl_df = aapl_df_beginning #.iloc[round((len(aapl_df_beginning)*((.5)*np.random.random()))):round((len(aapl_df_beginning)*((1-.51)*np.random.random()+.51)))]
                    train = round(len(aapl_df)*.5) #((.6-.4)*np.random.random()+.4))
                    test = len(aapl_df) - train
                elif self.train_test_split:
                    aapl_df = aapl_df_beginning.iloc[round((len(aapl_df_beginning)*((.5)*np.random.random()))):round((len(aapl_df_beginning)*((1-.51)*np.random.random()+.51)))]
                    train = round(len(aapl_df)*((.6-.4)*np.random.random()+.4))
                    test = len(aapl_df) - train

                ##SET PARAMETERS FOR THE STDS AND MOVING AVERAGES
                for short_roll in range(1,3):
                    for long_roll in range(108,112,1):
                        for first_rollingstd in range(10,16,2):
                            for std_rolling_max in range(130,160,5):
                                short_rolling = short_roll
                                long_rolling = long_roll
                                first_rolling_std = first_rollingstd
                                std_rolling_max = std_rolling_max


                                #GET APPLE RETURNS
                                """
                                COMMENT ILOC HERE TO FIND OVERALL RETURNS

                                """
                                if not self.train_test_split:
                                    aapl = aapl_df #.iloc[:train].copy()
                                elif self.train_test_split:
                                    aapl = aapl_df.iloc[:train].copy()

                                aapl = aapl.sort_index()

                                aapl['returns'] = np.log(aapl['AdjClose']/aapl['AdjClose'].shift(1))


                                aapl['FIRST_ROLLING_STD'] = aapl['AdjClose'].ewm(span = first_rolling_std).std()
                                aapl['STANDARD_STD'] = (aapl['FIRST_ROLLING_STD'] - aapl['FIRST_ROLLING_STD'].mean())/aapl['FIRST_ROLLING_STD'].std()
                                aapl['ROLLING_LONG'] = aapl['AdjClose'].rolling(window = long_rolling).mean()
                                aapl['ROLLING_SHORT'] = aapl['AdjClose'].rolling(window = short_rolling).mean()


                                aapl['AVG_STD_MAX'] = aapl['STANDARD_STD'].rolling(window = std_rolling_max).max()
                                aapl['CUMMAX'] = aapl['STANDARD_STD'].cummax()
                                #aapl['AVG_STD_MAX_10'] = aapl['STANDARD_STD'].rolling(window = 20).max()




                                #DEFINING TRADING STRATEGY
                                aapl['ma_above'] = np.where(aapl['ROLLING_SHORT'] > aapl['ROLLING_LONG'], 1, -1)


                                aapl['STD_MAX'] = np.where(aapl['CUMMAX'] == aapl['AVG_STD_MAX'], 1,0)

                                #CREATING TRADING SIGNALS

                                aapl['TRADE'] = aapl['STD_MAX'] * aapl['ma_above']


                                #FINDING RETURNS WITH STRATEGY AND RETURNS OF STOCK

                                aapl['STRAT_RETURNS'] = aapl['TRADE'] * aapl['returns'].shift(1)
                                aapl['STRAT_RETURNS'] = aapl['STRAT_RETURNS'].cumsum()
                                aapl['CUM_RETURNS'] = aapl['returns'].cumsum()



                                returns_dict[aapl['STRAT_RETURNS'].iloc[-1]] = [short_roll, long_roll,first_rollingstd,std_rolling_max, stock  ]
                                overall_returns[aapl['STRAT_RETURNS'].iloc[-1]] = [short_roll, long_roll, first_rollingstd, std_rolling_max, stock, aapl_df.iloc[0].name, aapl_df.iloc[-1].name,aapl_df.iloc[-1].name - aapl_df.iloc[0].name ]





                returns_df = pd.DataFrame.from_dict(returns_dict, orient = 'index')
                returns_df.columns = ['short_roll', 'long_roll','first_rollingstd', 'std_rolling_max', 'STOCK']
                max_returns = returns_df.sort_index(ascending=False).iloc[0]
                #max_returns


                short_rolling = max_returns[0]
                long_rolling = max_returns[1]
                first_rolling_std = max_returns[2]
                std_rolling_max = max_returns[3]


                #GET APPLE RETURNS
                """
                COMMENT ILOC HERE TO FIND OVERALL RETURNS

                """

                if not self.train_test_split:
                    aapl = aapl_df #.iloc[train:].copy()
                elif self.train_test_split:
                    aapl = aapl_df #.iloc[train:].copy()

                aapl['returns'] = np.log(aapl['AdjClose']/aapl['AdjClose'].shift(1))


                aapl['FIRST_ROLLING_STD'] = aapl['AdjClose'].ewm(span = first_rolling_std).std()
                aapl['STANDARD_STD'] = (aapl['FIRST_ROLLING_STD'] - aapl['FIRST_ROLLING_STD'].mean())/aapl['FIRST_ROLLING_STD'].std()
                aapl['ROLLING_LONG'] = aapl['AdjClose'].rolling(window = long_rolling).mean()
                aapl['ROLLING_SHORT'] = aapl['AdjClose'].rolling(window = short_rolling).mean()


                aapl['AVG_STD_MAX'] = aapl['STANDARD_STD'].rolling(window = std_rolling_max).max()
                aapl['CUMMAX'] = aapl['STANDARD_STD'].cummax()
                #aapl['AVG_STD_MAX_10'] = aapl['STANDARD_STD'].rolling(window = 20).max()




                #DEFINING TRADING STRATEGY
                aapl['ma_above'] = np.where(aapl['ROLLING_SHORT'] > aapl['ROLLING_LONG'], 1, -1)


                aapl['STD_MAX'] = np.where(aapl['CUMMAX'] == aapl['AVG_STD_MAX'], 1,0)

                #CREATING TRADING SIGNALS

                aapl['TRADE'] = aapl['STD_MAX'] * aapl['ma_above']


                #FINDING RETURNS WITH STRATEGY AND RETURNS OF STOCK

                aapl['STRAT_RETURNS'] = aapl['TRADE'] * aapl['returns'].shift(1)
                aapl['STRAT_RETURNS'] = aapl['STRAT_RETURNS'].cumsum()
                aapl['CUM_RETURNS'] = aapl['returns'].cumsum()


                difference = (aapl['STRAT_RETURNS'].iloc[-1] - float(max_returns.name))/float(max_returns.name)








                aapl[['STANDARD_STD', 'ROLLING_LONG', 'ROLLING_SHORT', 'AVG_STD_MAX', 'CUMMAX', 'TRADE']].plot(secondary_y = ['STANDARD_STD', 'AVG_STD_MAX', 'CUMMAX', 'TRADE'],
                                                                                                               style = ['-', '-', '--', '-^', '--'] ,figsize = (20,10),
                                                                                                               title = (stock.upper() + ' train percentage ' +
                                                                                                                        str(round(train/(train+test),3))+
                                                                                                                        ', DIFFERENCE BETWEEN TRAIN AND TEST = '
                                                                                                                        + str(round(difference*100, 3))+ '%'))

                plt.fill_between(aapl.index, aapl['CUMMAX'], aapl['AVG_STD_MAX'], color = 'black',alpha = 0.75 )

                #plt.figure(figsize = (20,5))
                #aapl[['ROLLING_LONG', 'ROLLING_SHORT']].plot(figsize = (20,5))

                aapl[['STRAT_RETURNS', 'CUM_RETURNS']].plot( figsize = (20,10), title = stock.upper())

                plt.fill_between(aapl.index, aapl['STRAT_RETURNS'], aapl['CUM_RETURNS'],
                                where = aapl['STRAT_RETURNS'] > aapl['CUM_RETURNS'], facecolor = 'g', interpolate = True,
                                alpha = 0.5)
                plt.fill_between(aapl.index, aapl['STRAT_RETURNS'], aapl['CUM_RETURNS'],
                                where = aapl['STRAT_RETURNS'] < aapl['CUM_RETURNS'], facecolor = 'r', interpolate = True,
                                alpha = 0.5)

                plt.show()





            overall_returns_df = pd.DataFrame.from_dict(overall_returns, orient = 'index')
            overall_returns_df.columns = ['short_roll', 'long_roll','first_rollingstd', 'std_rolling_max', 'stock', 'start', 'stop', 'time_difference']


            find_start_col = pd.read_excel('ALGO_STATISTICS.xlsx')
            start_col = len(find_start_col)
            writer = pd.ExcelWriter('ALGO_STATISTICS.xlsx', engine = 'openpyxl', startrow = start_col + 1)
            overall_returns_df.to_excel(writer, header = False)
            writer.save()
