import pandas as pd
import pickle

BEER_INDICES_PATH = "models/unique_beer_indices.pkl"


def recommend_beers(beer_df, beer_name, cosine_sim, num_recommendations=5):
    with open(BEER_INDICES_PATH, "rb") as f:
        indices = pickle.load(f)
    print(indices)
    print(len(beer_df))
    if beer_name not in indices:
        print(f"The beer '{beer_name}' is not in the database.")
        return

    idx = indices[beer_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[
        1 : num_recommendations + 1
    ]  # Get the desired number of similar beers
    beer_indices = [i[0] for i in sim_scores]

    print(f"Recommendations for {beer_name}:\n")
    beer_names = []
    for index in beer_indices:
        beer_names.append(beer_df.iloc[index]["beer_name"])
    return beer_indices, beer_names


def get_user_id(username, dataset):
    # Iterate over all users in the dataset
    for user_id, inner_id in dataset.ur.items():
        # Convert inner user ID to original user ID (username)
        original_username = dataset.to_raw_uid(user_id)
        # Check if the username matches the input username
        if original_username == username:
            return user_id
    # Username not found in the dataset
    return None


def get_top_recommendations(model, user_id, top_n=5):
    # Get the top-N recommendations for a user
    print(user_id)
    user_id = get_user_id(user_id, model.trainset)
    print(user_id)
    user_items = set(model.trainset.all_items())
    user_unseen_items = user_items - set(model.trainset.ur[user_id])
    predictions = [model.predict(user_id, item_id) for item_id in user_unseen_items]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
    top_items = [
        (model.trainset.to_raw_iid(pred.iid), pred.est) for pred in top_predictions
    ]
    return top_items


def userbasedrecommendation(model,df):

    from surprise import Dataset, Reader

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(df[['username', 'beer_name', 'score']], reader) 

    from surprise import SVD,KNNBasic,NMF,NormalPredictor
    from surprise.model_selection import cross_validate

    benchmark = []

    for algorithm in [SVD(), NMF(), NormalPredictor(), KNNBasic()]:

        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = pd.concat([tmp, pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm'])])
        benchmark.append(tmp)

    from surprise.model_selection import train_test_split

    # Split the data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    trainset = data.build_full_trainset()
    model.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = model.test(testset)

    from collections import defaultdict
    def get_all_predictions(predictions):
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)

        return top_n



    all_pred = get_all_predictions(predictions)

    n = 5

    for uid, user_ratings in all_pred.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        all_pred[uid] = user_ratings[:n]

    tmp = pd.DataFrame.from_dict(all_pred)
    tmp_transpose = tmp.transpose()

    def get_predictions(username):
        results = tmp_transpose.loc[username]
        return results

    with open(BEER_INDICES_PATH, "rb") as f:
        indices = pickle.load(f)
    username= df.iloc[indices[1]]['username']
    results = get_predictions(username)

    recommended_beers=[]
    for x in range(0, n):
        recommended_beers.append(results[x][0])

    print(recommended_beers)