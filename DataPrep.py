#Author: William Frazee
#The purpose of this script is to prepare incel and non incel user data for classification for my project.
from remoteDBconnector import dbconn
import pandas as pd
import argparse, sys
from bson.regex import Regex
from sklearn.feature_extraction.text import TfidfVectorizer
import pathlib
from tqdm import tqdm
import os
from datetime import datetime
#also remember to output a description of the data set. Look at pandas. Number of words, vocabulary size, average size of corpus, etc, Df.describe


#Setup
print("Doing script prepartation...")

current_dir = str(pathlib.Path(__file__).parent.resolve())+'\\'

#Handle command line arguments
#note to self, access specific arguments with args.year as an example
parser=argparse.ArgumentParser()
parser.add_argument("-t", "--targetSubreddit", help="This is the subreddit that we are trying to classify users as members of.")
parser.add_argument("-y", "--year", help="What year are we querying?")
parser.add_argument("-m", "--month", help="What month are we querying?")
args=parser.parse_args()

#make a directory to store the results of this script in
output_dir = "DataprepOutput_"+current_dir+datetime.now().strftime("%d-%m-%Y_%H.%M.%S")+"__"+args.year+"-"+args.month+'\\'
os.mkdir(output_dir)
f = open(output_dir+"output.txt", "a")
listOfSubreddits = pd.read_csv(current_dir+r'SubredditList.csv')['0']
'''
###############################################################################
###############################################################################
START QUERIES
If you are adapting my code to your own database, then you will want to change 
this section. I had access to a large mongodb database of reddit comments
for this project.
###############################################################################
###############################################################################
'''
#Handling mongodb connection
mytunnel=dbconn(current_dir+'mongo_reddit.conf')
database = mytunnel.conn["reddit"]
###############################################################################
'''
This section queries for active users of the target subreddit.
'''
###############################################################################
print("Querying for active users of the subreddit:", args.targetSubreddit)


#Query for all users who have posted in the target subreddit
def queryForUsersOfSubreddit(year,month,db,subreddit):
    collectionName = "comments_{y}-{m}".format(m=month, y=year)
    collection = db[collectionName]
    query = {}
    query["subreddit"] = subreddit
    projection = {}
    projection["author"] = 1.0
    cursor = collection.find(query, projection=projection)
    users = list(cursor)
    listOfUsers = []
    for item in users:
        if(not item['author'] == '[deleted]' or not item['author'] == 'AutoModerator'):
            listOfUsers.append(item['author'])
    listOfUsers = list(set(listOfUsers))
    return listOfUsers

def queryForCommentsByUser(year,month,db,user,subreddit):
    collectionName = "comments_{y}-{m}".format(m=month, y=year)
    collection = db[collectionName]
    query = {}
    query["$and"] = [{u"subreddit": subreddit}, {u"author": user}]
    projection = {}
    projection["body"] = 1.0
    cursor = collection.find(query, projection=projection)
    return(list(cursor))

#execute the query and ensure that every user is unique by making it a set
print("Querying for target subreddit usernames...")
users = set(queryForUsersOfSubreddit(args.year,args.month,database,args.targetSubreddit))
print("Querying for comments by target users, in order to determine who is an active user...")
totalComments = 0
usersToComments = {}
for user in tqdm(users):
    ammountOfComments = 0
    ammountOfComments = ammountOfComments + len(queryForCommentsByUser(args.year,args.month,database,user,args.targetSubreddit))
    usersToComments[user] = ammountOfComments
    totalComments += ammountOfComments
print("Total number of comments by target users in the target subreddit:", totalComments, file=f)
#Identify active Incels users by total number of comments per user, 
##defined as any user who has a comment history in incel of equal to or above the average
usersSorted=dict(sorted(usersToComments.items(),key = lambda x:x[1],reverse=True))

import math
commentThreshold = math.floor(totalComments/len(usersSorted))
print("comment threshold:", commentThreshold, file=f)
targetUsers = []
for user in usersSorted:
    if(usersToComments[user]>=commentThreshold):
        targetUsers.append(user)
dfTargetSubredditUsers = pd.DataFrame(targetUsers)
dfTargetSubredditUsers.to_csv(output_dir+'TargetUsers.csv', index=False)

###############################################################################
'''
In this section, we query for comments made by users who ARE apart of 
the target subreddit, inside our list of generic subreddits.
'''
###############################################################################
print("Querying for comments of people who are apart of the target subreddit")
#This query is meant to grab all commments by a user in a subreddit
def queryForComments(year,month,db,subreddit,user):
    collectionName = "comments_{y}-{m}".format(m=month, y=year)
    collection = db[collectionName]
    query = {}
    query["$and"] = [
        {
            u"author": {
                u"$ne": u"AutoModerator"
            },
            u"author": {
                u"$ne": u"[deleted]"
            }
        },
        {
            u"subreddit": subreddit
        },
        {
            u"$nor": [
                {
                    u"distinguished": u"moderator"
                }
            ]
        },
        {
            u"body": {
                u"$not": Regex(u".*\\[removed].*", "i")
            }
        },
        {
            u"body": {
                u"$not": Regex(u".*\\[deleted].*", "i")
            }
        }
    ]
    if(user!=-1):
        query["author"] = user
    projection = {}
    projection["author"] = 1.0
    projection["subreddit"] = 1.0
    projection["body"] = 1.0
    cursor = collection.find(query, projection=projection)
    return(list(cursor))

targetSubComments_df = pd.DataFrame([[0,0]], columns = ['author', 'body']) 
for user in tqdm(targetUsers):
    for subreddit in listOfSubreddits:
        df = pd.DataFrame((queryForComments(args.year, args.month,database,subreddit,user)))
        if(not df.empty):
            targetSubComments_df = pd.concat([df,targetSubComments_df])
targetSubComments_df.to_json(output_dir+'TargetUserComments.json', default_handler=str, orient="records")
print("Target subreddit users comments description:\n", targetSubComments_df.describe(), file=f)

###############################################################################
'''
In this section, we query for comments made by users who are NOT apart of 
the target subreddit, inside our list of generic subreddits.
'''
###############################################################################
print("Quering for regular user comments.")

regularUserComments_df = pd.DataFrame()

for subreddit in tqdm(listOfSubreddits):
    temp_df = pd.DataFrame((queryForComments(args.year,args.month,database,subreddit,-1)))
    if(not temp_df.empty):
        regularUserComments_df = pd.concat([temp_df,regularUserComments_df])

#exclude comments made by users of the target subreddit
print("Done with queries, you can close the VPN. Removing users who are users of target subreddit...")
for user in tqdm(targetUsers):
    regularUserComments_df = regularUserComments_df[regularUserComments_df.author != user]
print("Regular user dataframe description:\n", regularUserComments_df.describe(), file=f)

#ensure that we get a sampling from regular users comparable to the number of comments made by users in the target subreddit
print("Starting sampling of regular users...")

regularUserCounts = regularUserComments_df.groupby('author').count().reset_index()
regularUserList = []

#Don't be alarmed if the number of authors in targetSubComments is different than the length of targetUsers
#This could just means that some people in targetusers did not post in the subreddits that we quueried
for user in tqdm(targetSubComments_df.author.unique()):
    #figure out the number of comments that the target user has made
    targetCount = len(targetSubComments_df[targetSubComments_df['author'] == user])
    #get one random uiser who matches the comment count of the target user
    if(len(regularUserCounts.loc[regularUserCounts['body']==targetCount].author)==0):
        #no one had the same comment count. I don't expect this to happen too often
        print("no regular user had the same comment count as\n",user, targetCount, file=f)

        #drop the offending user from the dataframe
        #targetSubComments_df = targetSubComments_df[targetSubComments_df.author != user]
        #continue
        #search for the closest count one increment at a time
        while(len(regularUserCounts.loc[regularUserCounts['body']==targetCount].author)==0):
            targetCount = targetCount - 1
            if(targetCount == 0):
                raise Exception("targetCount is 0, something wrong with logic or dataset")
    name = regularUserCounts.loc[regularUserCounts['body']==targetCount].author.sample().iloc[0]
    #drop the user from our collection, then append it to our list
    regularUserCounts = regularUserCounts[regularUserCounts.author != name]
    regularUserList.append(name)

regularUserComments_df = regularUserComments_df[regularUserComments_df['author'].isin(regularUserList)]

print("Sample of regular user dataframe description:\n", regularUserComments_df.describe(), file=f)
regularUserComments_df.to_json(output_dir+'NonTargetUserSample.json', default_handler=str, orient="records")
'''
###############################################################################
###############################################################################
END QUERIES
###############################################################################
###############################################################################
'''
'''
###############################################################################
###############################################################################
START PREPROCESSING
###############################################################################
###############################################################################
'''
###############################################################################
'''
#Turn each user's comments into a document, per user, by concatting them together.
'''
###############################################################################
def concat_comments_per_user(df):
    user_list = df.groupby("author").count().reset_index()['author'].tolist()
    list_of_dicts = []
    for user in tqdm(user_list):
        #i don't know why, but I need to do this due to the idiosyncricies of dataframes
        if(user==0):
            continue
        user_df = df[df['author'] == user]
        corpus = ''
        for row in user_df.itertuples():
            corpus = corpus + "[START]" + str(row.body) + "[END]"
        if(len(corpus)==0):
            print(user)
        new_row = {'author':user, 'corpus':corpus}
        #print("newrow:",new_row)
        list_of_dicts.append(new_row)
    return (pd.DataFrame(list_of_dicts))
print("Combining target user comments into a single document, per user...")
targetSubComments_df_concat = concat_comments_per_user(targetSubComments_df)
print("Combining regular user comments into a single document, per user...")
regularUserComments_df_concat = concat_comments_per_user(regularUserComments_df)
targetSubComments_df_concat.to_json(output_dir+'ConcatTargetUserComments.json', orient="records")
regularUserComments_df_concat.to_json(output_dir+'ConcatNonTargetUserComments.json', orient="records")

###############################################################################
'''
In this section, we will vectorize the documents of each user. Then, we will apply
each user with a label.
Then we will combine the users into one dataframe and seperate into test and train
'''
###############################################################################

def Vectorize(df):
    vectorizer = TfidfVectorizer()
    #perform the tfid operation
    vectors = vectorizer.fit_transform(df['corpus'].to_list())
    feature_names = vectorizer.get_feature_names()
    #convert from vectorizer to dataframe for ease of use
    dense = vectors.todense()
    denselist = dense.tolist()
    wordDocument_df = pd.DataFrame(denselist,columns=feature_names)

    #export dataframe containing the matrix of the word x documents
    return wordDocument_df

def DivideIntoPercentageSet(denominator, df):
    #demoniator should be a decimal <1
    part = df.sample(frac = denominator, replace=False)
    rest = df.drop(part.index)
    return part, rest

print("Vectorizing the documents")
#combine into one dataframe
targetSubComments_df_concat['label'] = args.targetSubreddit
regularUserComments_df_concat['label'] = 'non-'+args.targetSubreddit
combined_df = pd.concat([targetSubComments_df_concat, regularUserComments_df_concat], ignore_index=True)
#vectorize the comments, then apply the labels from the unvectorized df
vectorizedComments = Vectorize(combined_df)
extracted_col = combined_df["label"]
vectorizedComments = pd.merge(vectorizedComments, extracted_col, left_index=True, right_index=True)
vectorizedComments.rename(columns={'label_y': 'label'}, inplace=True)
#split into train/test
train_df, test_df = DivideIntoPercentageSet(.8,vectorizedComments)
train_df.to_json(output_dir+'TrainSet.json', orient="records")
test_df.to_json(output_dir+'TestSet.json', orient="records")

'''
###############################################################################
###############################################################################
END PREPROCESSING
###############################################################################
###############################################################################
'''
f.close()