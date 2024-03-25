import joblib
import pandas as pd
import json



def vector_count(title, vectors):
  wordcount = []
  for vector in vectors :
    count = title.count(vector)
    wordcount.append(count)

  return wordcount

def predict_price(title: str, stars: float, reviews: int, isBestSeller: bool, boughtInLastMonth: int) -> None:
    if not isinstance(title, str):
        raise ValueError('Title cannot be empty')

    if not isinstance(stars, float):
        raise ValueError('Stars must be a float')
    elif not 0 <= stars <= 5:
        raise ValueError('Stars must be between 0 and 5')

    if not isinstance(reviews, int) or reviews < 0:
        raise ValueError('Reviews must be a non-negative integer')

    if not isinstance(isBestSeller, bool):
        raise ValueError('isBestSeller must be a boolean value')

    if not isinstance(boughtInLastMonth, int) or boughtInLastMonth < 0:
        raise ValueError('boughtInLastMonth must be a non-negative integer')
    
    
    pickled_model = joblib.load(open('model.pkl', 'rb'))

    title_vectors = pickled_model.feature_names_in_
    title_vectors = [value for value in title_vectors if value not in ['stars', 'reviews', 'isBestSeller', 'boughtInLastMonth']]

    count = vector_count(title, title_vectors)
    X = [stars, reviews, isBestSeller, boughtInLastMonth]
    counting = vector_count('premium labor lord mic', title_vectors)
    for val in counting:
      X.append(val)
    X = pd.DataFrame([X], columns=pickled_model.feature_names_in_)

 
    result = pickled_model.predict(X)

    return result



#title = "Carrying Case for Nintendo Switch / New Switch OLED Console & Accessories, Switch Carry Case with Waterproof Soft Lining Protector Hard Shell Home Storage & Travel Box with 10 Games Cartridges Pouch (Red&Blue)"

#x = predict_price(title, 5.0, 4, True, 0 )
#print(x[0])
