import pickle

favorite_color = { "lion": "yellow", "kitty": "red" }

pickle.dump( favorite_color, open( "save.pkl", "wb" ) )
    
favorite_color2 = pickle.load( open( "save.pkl", "rb" ) )
print(favorite_color2)
