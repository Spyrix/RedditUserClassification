#Author William Frazee
#The purpose of this script is to imoport a model and use it to classify a reddit user
import pandas as pd
import tqdm
from bson.regex import Regex
import pickle
import pathlib
import argparse, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from remoteDBconnector import dbconn


current_dir = str(pathlib.Path(__file__).parent.resolve())+'\\'

#Handle command line arguments
#note to self, access specific arguments with args.year as an example
parser=argparse.ArgumentParser()
parser.add_argument("-u", "--user", help="This is the user that we are trying to classify.")
parser.add_argument("-f", "--filepath", help="This is the parent folder holding the model we are using to classify.")
parser.add_argument("-y", "--year", help="What year are we querying?")
parser.add_argument("-m", "--month", help="What month are we querying?")
args=parser.parse_args()

#setup database connection
#Handling mongodb connection
subreddit_list = pd.read_csv(current_dir+r'SubredditList.csv')
mytunnel=dbconn(current_dir+r'mongo_reddit.conf')
database = mytunnel.conn["reddit"]

model = pickle.load(open(args.filepath+r"\classifier.sav", 'rb'))

#define query function
def queryForComments(year,month,db,subreddit,user):
    collectionName = "comments_{y}-{m}".format(m=month, y=year)
    collection = db[collectionName]
    query = {}
    query["$and"] = [
        {
            u"author": user
        },
        {
            u"subreddit": subreddit
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


    projection = {}

    projection["author"] = 1.0
    projection["body"] = 1.0
    projection["subreddit"] = 1.0

    cursor = collection.find(query, projection = projection)    
    return(list(cursor))

#querying for all of the comments made by the target user
comment_df = pd.DataFrame()

for subreddit in subreddit_list['0'].tolist():
    temp_df = pd.DataFrame((queryForComments(args.year,args.month,database,subreddit,args.user)))
    if(not temp_df.empty):
        comment_df = pd.concat([temp_df,comment_df])

#This function concats the user's comments together
def concat_comments_per_user(df):
    user_list = df.groupby("author").count().reset_index()['author'].tolist()
    list_of_dicts = []
    for user in user_list:
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

comment_df = concat_comments_per_user(comment_df)

#So what we need to do now is to create a new tdidf vectorizer from the old vocabulary.
regularUserComments_df_concat = pd.read_json(args.filepath+r'\ConcatNonTargetUserComments.json')
targetSubComments_df_concat =  pd.read_json(args.filepath+r'\ConcatTargetUserComments.json')
combined_df = pd.concat([targetSubComments_df_concat, regularUserComments_df_concat], ignore_index=True)

#obtain the old vocabulary
vectorizer1 = TfidfVectorizer()
tf1 = vectorizer1.fit(combined_df['corpus'].to_list())
#use the old vocabulary to create a new vectorizer and vectorize 
vectorizer2 = TfidfVectorizer(vocabulary=tf1.vocabulary_)
vector_matrix = vectorizer2.fit_transform(comment_df['corpus'].to_list())
vector_df = pd.DataFrame(vector_matrix.toarray())
#use loaded model to predict
print(model.predict(vector_df))