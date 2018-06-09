
def get_train_data():
    import myenv
    import gym
    import random

    env = gym.make('myenv-v3')
    Episodes = 100

    data_set = []

    for _ in range(Episodes):
        observation = env.reset()
        done = False
        count = 0

        while not done:
            action = random.choice([0, 1, 2, 3])
            next_observation, reward, done, info = env.step(action)

            data_set.append([observation, action, reward])

            count += 1
            observation = next_observation


            if done:
                print('reward: ', reward)
                print('steps: ', count)


    import pickle
    with open('sample2.pickle', mode='wb') as f:
        pickle.dump(data_set, f)

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np
def cllasify_data_by_rf():

    with open('sample.pickle', mode='rb') as f:
        data_set = pickle.load(f)

    X = []
    Y = []
    for d in data_set:
        x = d[0].flatten().tolist()
        x.append(d[1])
        Y.append(d[2])
        X.append(x)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)
    rf = RandomForestClassifier()
    rf.fit(train_x, train_y)

    accuracy = rf.score(test_x, test_y)
    print(accuracy)




if __name__ == '__main__':
    get_train_data()
    #cllasify_data_by_rf()