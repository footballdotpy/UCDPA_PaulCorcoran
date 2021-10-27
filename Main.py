# Step 1 importing necessary libraries.
import zipfile
import kaggle as kg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import math

warnings.filterwarnings('ignore')


# Step 2 data extraction and cleaning.

def kaggle_download(dataset, kaggle_file_name):
    """This function initialises the kaggle api to download the csv files from a chosen source.
    A API key is required for use. This is stored on the users PC in users/username/.kaggle/kaggle.json.
    The function takes two inputs, the dataset argument must be the kaggle username/datasetname and the kaggle_file_name
    is the file csv name. Both inputs must be in string format. the output is the csv file in the working directory.
    This needs to be unzipped.
    """

    api = kg.KaggleApi()
    api.authenticate()
    api.dataset_download_file(dataset=dataset, file_name=kaggle_file_name)


# use function to retrieve datasets.

kaggle_download(dataset='stefanoleone992/fifa-22-complete-player-dataset', kaggle_file_name='players_22.csv')

kaggle_download(dataset='stefanoleone992/fifa-22-complete-player-dataset', kaggle_file_name='players_21.csv')

# use zipfile module to unzip files into directory.

with zipfile.ZipFile('players_22.csv.zip', 'r') as zipref:
    zipref.extractall()

with zipfile.ZipFile('players_21.csv.zip', 'r') as zipref:
    zipref.extractall()

# use pandas to read in csvs.

fifa21 = pd.read_csv('players_21.csv')
print(fifa21.shape)  # 18949 rows, 107 columns

fifa22 = pd.read_csv('players_22.csv')
print(fifa22.shape)  # 19239 rows, 107 columns

# merge both files
fifa_complete_data = pd.DataFrame.merge(fifa22, fifa21, how="outer")
print(fifa_complete_data.shape)  # 38188 rows, 107 columns

# preview the data
fifa_complete_data.head(10)

# calculate basic summary statistics
fifa_complete_data.describe()
fifa_complete_data.info()

# investigate for duplicates, we would expect there to be a large number as the majority of players were active in
# both years.

number_of_duplicated_players = fifa_complete_data.duplicated("short_name").sum()
print("The number of duplicated players is: ", number_of_duplicated_players)

fifa_complete_data = fifa_complete_data.drop_duplicates("short_name", keep="first")
# As we used the latest data set first in the merge,the keep param ensures we keep the 2022 up-to-date statistics for each player that was duplicated.

fifa_complete_data.shape  # we are left with 23265 player rows, 107 columns. Due to the fluctuation of players retiring
# and players graduating from academies into first team we have gained a couple of thousand players.

fifa_complete_data = fifa_complete_data[
    fifa_complete_data["overall"] >= 70]  # filtering dataset for players 70 and above
fifa_complete_data.shape  # (5978, 107)

# split the data on the player_position column,some players have multiple positions.

players_with_comma = fifa_complete_data[fifa_complete_data["player_positions"].str.contains(",")]
players_no_comma = fifa_complete_data[~fifa_complete_data["player_positions"].str.contains(",")]

# use regex to capture the players position, the first position before the ",".

regex = r'^(.+?),'

# apply the regex to the df containing players who have multiple positions outlined.
# the column club position could not be used as it contains "res" values indicating they are not first choice.

players_with_comma["player_positions"] = players_with_comma["player_positions"].str.extract(regex)

# merging the dataframes back with player_positions corrected.

fifa_complete_data = pd.merge(players_with_comma, players_no_comma, how="outer").sort_values("overall", ascending=False) \
    .reset_index().drop(columns=['index'])  # drop the index column as it is just a duplicate of ids.

print(fifa_complete_data.shape)  # check that we have not lost data.

# categorising the player positions to normalize the column.
# create a dictionary to iterate through and map player position to either defense,midfield or forward.

player_position_fifa = {"RB": "Defender", "RWB": "Defender", "RCB": "Defender", "CB": "Defender", "LB": "Defender",
                        "LWB": "Defender", "LCB": "Defender", "CDM": "Midfield", "CM": "Midfield", "LCM": "Midfield",
                        "RCM": "Midfield", "LM": "Midfield", "RM": "Midfield", "CAM": "Midfield", "LF": "Forward",
                        "ST": "Forward", "CF": "Forward", "RF": "Forward", "RW": "Forward", "LW": "Forward"}

# confirm key value pairs.
for key, value in player_position_fifa.items():
    print(key + ":" + str(value))

# iterate over columns and replace values with dictionary values.
for key, value in fifa_complete_data["player_positions"].iteritems():
    fifa_complete_data["player_positions"] = fifa_complete_data["player_positions"].apply(lambda x:
                                                                                          player_position_fifa.get(x,
                                                                                                                   x))

# check for null values
fifa_complete_data["player_positions"].isnull().sum()  # we get 0 values indicating no null values.

# check position count.
position_count = pd.crosstab(index=fifa_complete_data.player_positions, columns="Positions")
print(position_count)


def player_positions_plot():
    """ Make a visualisation of number of positions in df."""
    player_position_plot = ["GK", "Defender", "Midfield", "Forward"]
    plot1 = sns.catplot("player_positions", data=fifa_complete_data, kind="count",
                        legend=True, order=player_position_plot)
    plot1.fig.set_size_inches(10, 6)
    plot1.fig.subplots_adjust(top=0.81, right=0.86)
    # extract the matplotlib axes_subplot objects from the FacetGrid
    ax = plot1.facet_axis(0, 0)
    # iterate through the axes containers
    for i in ax.containers:
        labels = [f'{(v.get_height()):.0f}' for v in i]
        ax.bar_label(i, labels=labels, label_type='edge')
    plt.title("Player position totals FIFA 21/22")
    plt.xlabel("Positions")
    plt.ylabel("Frequency")
    plt.show()


player_positions_plot()

# investigating null values in overall df
null_values = fifa_complete_data.isnull().sum()
# pace,shooting,passing,dribbling,defending,physic have 570 missing values.
# equivalent to the GK stats which means goalkeepers do not have these attributes, we drop GK from dataset.
# dropping potential as it is closely linked to target variable.

df_to_use = fifa_complete_data.dropna(axis=0, how="any",
                                      subset=['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'])

# drop categorical/string variables.
df_to_use = df_to_use.drop(["nationality", "nation_position", "body_type", "real_face", "player_tags",
                            "player_traits", "dob", "player_positions", "short_name", "long_name", "potential",
                            "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
                            "goalkeeping_positioning",
                            "goalkeeping_reflexes", "goalkeeping_speed", "player_url", "club_joined", "club_name",
                            "league_name", "player_positions", "club_position", "player_url", "player_face_url",
                            "club_logo_url", "club_flag_url", "nation_logo_url", "nation_flag_url",
                            "club_loaned_from", "club_joined", "work_rate", "ls", "st", "rs", "lw",
                            "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm"
                               , "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb",
                            "cb", "rcb", "rb", "gk"], axis=1)

# Step 3: checking correlations with target variable with dataset.

# renaming player rating variable to more conventional name
fifa_complete_data = fifa_complete_data.rename(columns={"overall": "Skill_rating", "value_eur": "Player_value_euros"})
# recode player foot, 0 for left and 1 for right.
df_to_use["preferred_foot"] = df_to_use["preferred_foot"].replace("Left", 0).replace("Right", 1)
df_to_use = df_to_use.rename(columns={"overall": "Skill_rating", "value_eur": "Player_value_euros"})

pearson_corr = df_to_use.corr()["Player_value_euros"]

positive_corr_features = pearson_corr[pearson_corr > 0.5].round(decimals=2)
print("The variables that correlate with our target variable are:" "\n", positive_corr_features)

""" The variables that correlate the most with our target variable are:
 Skill_rating                0.80
value_eur                   1.00
wage_eur                    0.79
international_reputation    0.55
release_clause_eur          0.99
movement_reactions          0.62
Name: value_eur, dtype: float64 """


# creating a function for the pearson coeff plot.
def pearson_plot():
    plt.figure(figsize=(13, 11))
    # creating the corr plot
    df_to_use.corrwith(df_to_use['Player_value_euros']).plot(kind='barh', color="darkgreen")
    plt.title("Correlation between target variable player value and dataset")
    plt.ylabel(ylabel="variables", fontdict={"fontsize": .01})
    plt.show()


# call the correlation plot
pearson_plot()

# the results show 4 variables that negatively affect target variable and should be removed.
# they are variables that dont make alot of sense, eg nation_jersey_number, club_jersey_number
# league_level which is graded as 1 for highest has no affect on the skill rating.


# now selecting the most highly correlated features or variables from the data.
features_corr_positive = pearson_corr[pearson_corr > 0.5]
# printing the most postively and most negatively correlated factors:
print('Most Positively Correlated Features:''\n',
      features_corr_positive[features_corr_positive.values != 1.0].round(decimals=2))


# check distribution of data

def hist(var="plot"):
    """use histograms to check variable distribution"""
    plt.hist(x=var, data=df_to_use)
    plt.title("Histogram of variable distribution")
    plt.xlabel(var)
    plt.show()


# view the distributions of target variable and most correlated variables.

hist(var="Skill_rating")
hist(var="Player_value_euros")
hist(var="wage_eur")
hist(var="release_clause_eur")
hist(var="movement_reactions")
hist(var="mentality_composure")
hist(var="skill_ball_control")
hist(var="dribbling")
hist(var="attacking_short_passing")


# only the physical/mental stat metrics are normally distributed.

# investigate if linear relation is present in chosen variables.

def linear_relationship(variable="dummy", xlab="label", title="test"):
    """Scatter-plot to show the top correlating factors to skill level in fifa to illustrate the linear relationship"""
    sns.set()
    sns.scatterplot(x=variable, y="Player_value_euros", data=df_to_use)
    sns.set_theme("talk")
    plt.xlabel(xlab)
    plt.ylabel("Players valuation (Millions in Euro)")
    plt.title(title)
    plt.show()


# plot of linear relationship for the players movement reactions metric.
linear_relationship("movement_reactions", xlab="Players movement_reactions",
                    title="Linear Relationship plot")

# plot of linear relationship for the players movement composure metric.
linear_relationship("mentality_composure", xlab="Players mentality_composure value",
                    title="Linear Relationship plot")

# plot of linear relationship for the players offensive passing metric.
linear_relationship("attacking_short_passing", xlab="Players offensive short passing value",
                    title="Linear Relationship plot")

# plot of linear relationship for the players ball skill metric.
linear_relationship("skill_ball_control", xlab="Players ball skill value",
                    title="Linear Relationship plot")

# plot of linear relationship for the players ball skill metric.
linear_relationship("wage_eur", xlab="Players wage value (thousands of euro per week)",
                    title="Linear Relationship plot")

# Step 4: normalize the data using sklearn to scale the continuous high variance features.

# instantiating the scaler and assigning to a variable.
StandardScaler = StandardScaler()
# save our player unique id variable from scaling.
fifa_id = df_to_use["sofifa_id"]
# convert do df for later merge.
fifa_id = pd.DataFrame(fifa_id)
# drop it from the host dataframe.
df_to_use = df_to_use.drop(["sofifa_id"], axis=1)
# fitting the scaler to the newly created df to transform the data.
scaled_features = StandardScaler.fit_transform(df_to_use)

# turn it back to a pandas df. First retrieve the column names using a iteration
col_values = {df_to_use.columns.get_loc(c): c for idx, c in enumerate(df_to_use.columns)}
# then store this in a variable as a list of the values.
scaled_col_names = list(col_values.values())
# complete the dataframe process
scaled_features = pd.DataFrame(scaled_features, columns=scaled_col_names)

# check for any null values,release_clause_eur has a number of nan values from players not having a release clause.
scaled_features.isnull().sum()
# use mean to input NAN values
fill_mean = lambda col: col.fillna(col.mean())
# rewrite over scaled features and apply the mean of the column
scaled_features = scaled_features.apply(fill_mean, axis=0)
# check for any null values
scaled_features.isnull().sum()

# fill in nan values for non scaled dataset.
fill_mean = lambda col: col.fillna(col.mean())
# rewrite over scaled features and apply the mean of the column
df_to_use = df_to_use.apply(fill_mean, axis=0)


def scaled_hist(var="plot"):
    """use histograms to check variable distribution"""
    plt.hist(x=var, data=scaled_features)
    plt.title("Histogram of variable distribution")
    plt.xlabel(var)
    plt.show()


# check of each revealed no change to distribution.
scaled_hist(var="Player_value_euros")
scaled_hist(var="wage_eur")
scaled_hist(var="release_clause_eur")


# Step 5: drop the target variable from dataset to create modeling dataset.
modelling_data = df_to_use[
    ['Player_value_euros', 'mentality_composure', 'attacking_short_passing'
        , 'skill_ball_control', 'movement_reactions', 'Skill_rating', 'age', "dribbling", "shooting"]]
modelling_data = modelling_data.drop(["Player_value_euros"], axis=1)
# add back the fifa player id
# modelling_data = pd.merge(fifa_id, modelling_data, left_index=True, right_index=True)

# check modelling data for nan values
modelling_data.isna().sum()
# isolate the overall column which is the players rating.
target_variable = df_to_use["Player_value_euros"]

# Step 5b: Create a training/test split of 70/30 for non scaled dataset.

# use the train_test_split function to create the split, random_state allows for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(modelling_data, target_variable, random_state=42,
                                                    test_size=.10)
# print the length of train and test variables created using train_test_split using len.
print("X_train:", len(X_train), "y_train:", len(y_train))
print("X_test:", len(X_test), "y_test:", len(y_test))


# Step 6: Performing machine learning on target variable "Skill_level" using 2 different models.

# the first regression is on the non scaled dataset.

def linear_reg():
    """A function for performing linear regression"""

    # instantiate the model.
    regressor = LinearRegression()

    # fitting the model to the training data using fit() method.
    regressor.fit(X_train, y_train)

    # making a prediction on values in the testing data using the predict() method.
    y_predictions = regressor.predict(X_test)
    # model statistics.
    print("Intercept value: \n", regressor.intercept_)
    print("Slope value: \n", regressor.coef_)
    print("Training set score: \n", regressor.score(X_train, y_train).round(decimals=1))
    print("Test set: \n", regressor.score(X_test, y_test).round(decimals=1))
    print("mean squared error:", mean_squared_error(y_test, y_predictions))
    print("root mean squared error:", math.sqrt(mean_squared_error(y_test, y_predictions)))

    # Evaluate which features are the most important.
    importance = regressor.coef_

    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    X_train_Sm = sm.add_constant(X_train)
    X_train_Sm = sm.add_constant(X_train)
    ls = sm.OLS(y_train, X_train_Sm).fit()
    print(ls.summary())
    # create a regression plot
    sns.regplot(y_test, y_predictions)
    plt.title("Regression plot of model predictions")
    plt.show()


# use the function to perform linear regression on the players transfer value.
linear_reg()



# Perform ridge technique on model

from sklearn.linear_model import Ridge
# instantiate the model and fit.
ridge = Ridge().fit(X_train, y_train)

print("Training set score: {:.1f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.1f}".format(ridge.score(X_test, y_test)))


# perform hyperparam tuning on the ridge model
params_ridge = {'alpha': [1,0.1,0.01,0.001,0.0001,0] , "fit_intercept": [True, False],
                "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

Ridge_GS = GridSearchCV(estimator=ridge, param_grid=params_ridge, n_jobs=-1, cv= 10)
# fit the model
Ridge_GS.fit(X_train, y_train)
# print the best params
print(Ridge_GS.best_params_) # {'alpha': 0.0001, 'fit_intercept': True, 'solver': 'saga'}
# print the best R2
print(round((Ridge_GS.best_score_),1))
# fit the hyperparam tuned model
ridge_tuned = Ridge(alpha=0.0001,fit_intercept=True, solver='saga').fit(X_train, y_train)
ridge_y_pred = ridge_tuned.predict(X_test)
# print scores
print("Training set score: {:.1f}".format(ridge_tuned.score(X_train, y_train)))
print("Test set score: {:.1f}".format(ridge_tuned.score(X_test, y_test)))
print("mean squared error:", mean_squared_error(y_test, ridge_y_pred))
print("root mean squared error:", math.sqrt(mean_squared_error(y_test, ridge_y_pred)))

# -----------------------------------------------------------------------------------------------------------------------


# Step 7: clean new dataframe for classification model.

dt_df = fifa_complete_data.drop(["nationality", "nation_position", "body_type", "real_face", "player_tags",
                                 "player_traits", "dob", "short_name", "long_name", "potential",
                                 "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
                                 "goalkeeping_positioning",
                                 "goalkeeping_reflexes", "goalkeeping_speed", "player_url", "club_joined", "club_name",
                                 "league_name", "club_position", "player_url", "player_face_url",
                                 "club_logo_url", "club_flag_url", "nation_logo_url", "nation_flag_url",
                                 "club_loaned_from", "club_joined", "work_rate", "ls", "st", "rs", "lw",
                                 "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm"
                                    , "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb",
                                 "cb", "rcb", "rb", "gk"], axis=1)

# drop the GK rows again.

dt_df = dt_df.dropna(axis=0, how="any",
                     subset=['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'])

# drop columns with nan values
dt_df = dt_df.dropna(axis=1)

# recraft the preferred foot column for use in classification model.
dt_df["preferred_foot"] = dt_df["preferred_foot"].replace("Left", 0).replace("Right", 1)
# recraft target column player positions.
dt_df["player_positions"] = dt_df["player_positions"].replace("Forward", 0).replace("Midfield", 1) \
    .replace("Defender", 2)


# creating a function for the pearson coeff plot.
def pearson_plot_dt():
    plt.figure(figsize=(13, 11))
    # creating the corr plot
    df_to_use.corrwith(dt_df["player_positions"]).plot(kind='barh', color="darkgreen")
    plt.title("Correlation between target variable player positions and dataset")
    plt.ylabel(ylabel="variables", fontdict={"fontsize": .01})
    plt.show()


# call the correlation plot
pearson_plot_dt()

pearson_corr_dt = dt_df.corr()["player_positions"]

positive_corr_features_dt = pearson_corr_dt[pearson_corr_dt > 0.5].round(decimals=2)

print("The variables that correlate with the target variable Player postions are: \n", positive_corr_features_dt)

# create a new dataframe for classification model based on these features.

dt_df_model = dt_df[["player_positions", "defending", "mentality_interceptions", "defending_marking_awareness",
                     "defending_standing_tackle", "defending_sliding_tackle"]]

# split the target variable
dt_target_variable = dt_df_model["player_positions"]
# drop the target variable from modeling data.

dt_df = dt_df.drop(["player_positions"], axis=1)

# Step 8: Use decision tree analysis to model player position based on individual statistics.

# create training/test split

dt_X_train, dt_X_test, dt_y_train, dt_y_test = train_test_split(dt_df, dt_target_variable, test_size=.30,
                                                                random_state=42)
# Initialise the decision tree model.
dt = DecisionTreeClassifier()
# fit the model
dt.fit(dt_X_train, dt_y_train)
# use the test data to make predictions.
dt_y_pred = dt.predict(dt_X_test)
# create a classification to review predictions.
print(classification_report(dt_y_test, dt_y_pred))
# print the model accuracy which is 0.83 %
print("Accuracy:", metrics.accuracy_score(dt_y_test, dt_y_pred).round(decimals=2))


# Step 9: create a function to optimise model selection and performance.

# use ensemble Random Forest method to improve accuracy. First we must perform GridSearchCV to find out the best params.
def model_tuning_GS(model, parameter_dict):
    """Function to perform hyperparameter turning for the classification models using GridSearch."""
    # inspect the model params.
    model.get_params()
    # define the parameters using a dictionary that we want to test.
    model_grid = parameter_dict
    # initialise a GSCV object with the model as an argument. scoring is set to accuracy and CV set to 5.
    Grid_model = GridSearchCV(estimator=model, param_grid=model_grid, cv=5, scoring="accuracy")
    # fit the model to data.
    Grid_model.fit(dt_X_train, dt_y_train)
    # extract the best estimator, accuracy score and print them.
    print("GridSearchCV results:", model.__class__.__name__)
    # print best estimator
    print("Best Estimator:\n", Grid_model.best_estimator_)
    # printing the mean cross-validated score of the best_estimator:
    print("\n Best Score:\n", Grid_model.best_score_)
    # printing the parameter setting that gave the best results on the hold out testing data.:
    print("\n Best Hyperparameters:\n", Grid_model.best_params_)


# Step 9b: call the GridSearchCV function on the random forest and Gradient boosting ensemble methods.

parameter_dict = {'n_estimators': [100, 200, 300, 400, 500],
                  'max_depth': [1, 5, 8, 9, 10],
                  'min_samples_leaf': [0.1, 0.2]}

# model_tuning_GS(RandomForestClassifier(random_state=42), parameter_dict)

# Create a Gaussian Classifier using the best params found.
clf = RandomForestClassifier(n_estimators=100, max_depth=5)

# Train the model using the training sets y_pred=clf.predict(dt_X_test)
clf.fit(dt_X_train, dt_y_train)
# use random forest to make predictions
y_pred = clf.predict(dt_X_test)
# print the accuracy, an increase of 3% to 0.86 was recorded.
print("Accuracy:", metrics.accuracy_score(dt_y_test, y_pred).round(decimals=2))

# create a parameter dictionary to test for the gradient boosting algorithm.

parameter_dict_GB = {'n_estimators': [100, 200],
                     'subsample': [0.8, 0.4],
                     'max_depth': [1, 5],
                     'learning_rate': [0.01]}

# find the optimal params for GBC
model_tuning_GS(GradientBoostingClassifier(random_state=42), parameter_dict_GB)

# perform gradient boosting on optimum params found above.
ensemble = GradientBoostingClassifier(learning_rate=0.01, max_depth=5, n_estimators=200, random_state=42,
                                      subsample=0.8).fit(dt_X_train, dt_y_train)
# make predictions using GDC.
ensemble_predictions = ensemble.predict(dt_X_test)
# print the accuracy, an increase of 3% was found and the model is now 0.89% accurate.
print("Accuracy:", metrics.accuracy_score(dt_y_test, ensemble_predictions).round(decimals=2))
