import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import itertools



data = pd.read_csv('Movie_Record.csv')
print(data.head(10))


# Compute cosine similarity and find the customer with max cosine similarity
idx = np.argmax(cosine_similarity(data.iloc[:9,1:].values, data.iloc[9, 1:].values.reshape(-1,7)))
customer_with_max_sim = data.iloc[idx]
customer = data.iloc[9]


# Find the columns where the values for the above customer is 1
col_to_consider = np.where(customer_with_max_sim ==1)[0]
common_movie = np.where(np.logical_and(customer==1,customer_with_max_sim==1)==True)[0]

lift_of = customer[common_movie]
lift_with = customer_with_max_sim[col_to_consider]

#calculation of lift values
x = list(lift_of.keys())
y = list(lift_with.keys())
total = len(data['Customer'])
movies_lift = {}

class recommendations:
    def singles(x,y):
        for i in range(len(x)):
            for j in range(len(y)):      
                a = sum(data[x[i]])
                b = sum(data[y[j]])
                a_b = sum(data[x[i]] == data[y[j]])
        
                if y[j] not in x:
                    support = a/total
                    confidence = a/a_b
                    lift = confidence/((b/total))
                    mn = x[i] + ' : ' + y[j]
                    movies_lift[mn] = lift
                    
    def doubles(x,y):
        for i in range(len(x)):
            for j in range(len(y)):
                pairs = []
                for pair in itertools.combinations(x, 2):
                    pairs.append(pair)

                for perm in range(len(pairs)):
                    temp = data[pairs[perm][0]] == data[pairs[perm][1]]
                    a = sum(temp)
                    b = sum(data[y[j]])
                    a_b = sum(temp == data[y[j]])
            
                    if (y[j] not in x):
                        support = a/total
                        confidence = a/a_b
                        lift = confidence/((b/total))
                        mn = pairs[perm][0]+pairs[perm][1] + ' : ' + y[j]
                        movies_lift[mn] = lift
                        

    def multiples(x,y):
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(3,(len(x)+1)):
                    pairs = []
                    for pair in itertools.combinations(x, k):
                        pairs.append(pair) 
                        
                    for l in range(len(pairs)):
                        for m in range(k-1):
                            r = data[pairs[l[m]]] & data[pairs[l[m+1]]]
                            a = sum(r)
                            b = sum(data[y[j]])
                            a_b = sum(r == data[y[j]])
                            
                            if (y[j] not in x):
                                support = a/total
                                confidence = a/a_b
                                lift = confidence/((b/total))
                                count = 1
                                mn = count + ' : ' + y[j]
                                movies_lift[mn] = lift
                                count += 1
                            
                    
recommendations.singles(x,y)
recommendations.doubles(x,y)
if len(x)>2:
    recommendations.multiples(x,y)
                             
                
            
        
print(max(movies_lift, key=movies_lift.get))
