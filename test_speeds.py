from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree
import json

import time
import os, time, json
from SAModel import *
from pathlib import Path


if __name__ == '__main__':
    
    

    #14, 37, 58, 97, 99, 107, 107
    # text = ["I like pie very much just like you and everyone else in this world."
    # ,"There was no answer and Edmund noticed that his own voice had a curious sound—not the sound you expect in a cupboard but a kind of open-air sound. He also noticed that he was unexpectedly cold."
    # ,"Now the steps she had heard were those of Edmund; and he came into the room just in time to see Lucy vanishing into the wardrobe. He at once decided to get into it himself—not because he thought it a particularly good place to hide but because he wanted to go on teasing her about her imaginary country. "
    # ,"On offering to help the blind man, the man who then stole his car, had not, at that precise moment, had any evil intention, quite the contrary, what he did was nothing more than obey those feelings of generosity and altruism which, as everyone knows, are the two best traits of human nature and to be found in much more hardened criminals than this one, a simple car-thief without any hope of advancing in his profession, exploited by the real owners of this enterprise, for it is they who take advantage of the needs of the poor."
    # ,"My very photogenic mother died in a freak accident (picnic, lightning) when I was three, and, save for a pocket of warmth in the darkest past, nothing of her subsists within the hollows and dells of memory, over which, if you can still stand my style (I am writing under observation), the sun of my infancy had set: surely, you all know those redolent remnants of day suspended, with the midges, about some hedge in bloom or suddenly entered and traversed by the rambler, at the bottom of a hill, in the summer dusk; a furry warmth, golden midges."
    # ,'The French are certainly misunderstood: — but whether the fault is theirs, in not sufficiently explaining themselves, or speaking with that exact limitation and precision which one would expect on a point of such importance, and which, moreover, is so likely to be contested by us — or whether the fault may not be altogether on our side, in not understanding their language always so critically as to know “what they would be at” — I shall not decide; but ‘tis evident to me, when they affirm, “That they who have seen Paris, have seen every thing,” they must mean to speak of those who have seen it by day-light.'
    # ,"In the loveliest town of all, where the houses were white and high and the elms trees were green and higher than the houses, where the front yards were wide and pleasant and the back yards were bushy and worth finding out about, where the streets sloped down to the stream and the stream flowed quietly under the bridge, where the lawns ended in orchards and the orchards ended in fields and the fields ended in pastures and the pastures climbed the hill and disappeared over the top toward the wonderful wide sky, in this loveliest of all towns Stuart stopped to get a drink of sarsaparilla."
    # ]
    
    text = ["I like pie very much just like you and everyone else in this world."
    ,"There was no answer and Edmund noticed that his own voice had a curious sound—not the sound you expect in a cupboard but a kind of open-air sound. He also noticed that he was unexpectedly cold."
    ]


    
    paths_file = open(Path(os.getcwd()) / "config.txt", "r")
    paths_json = json.load(paths_file)
    paths_file.close()
    root_path = paths_json["ROOT_PATH"]


    model = SAModel(paths_json)
    #vs
    nlp = StanfordCoreNLP('http://localhost:9000')


    timesServer = [0,0,0,0,0,0,0]
    timesNLTK = [0,0,0,0,0,0,0]
    test_runs = 3
    for k in range(test_runs):
        
        for j in range(len(text)):
            #print("Text: {}".format(text[j]))


            start_time = time.time()
            model.preparePipeline(text[j])
            end_time = time.time()

            timesServer[j] += end_time - start_time

        

       
        for j in range(len(text)):
            start_time = time.time()
            model.tokenizeReview(text[j])
                
            model.sent_comp_method = "PARSETREE"
            model.setReview()

            end_time = time.time()
            timesNLTK[j] += end_time - start_time
    
    print(timesServer) #[0.10884690284729004, 0.2878899574279785, 0.7185342311859131, 5.623664855957031, 6.225468873977661, 7.0179970264434814, 5.6672279834747314]
    print(timesNLTK)
    avg_time_ratio = [0,0,0,0,0,0,0]
    for i in range(len(text)):
        avg_time_ratio[i] = (timesNLTK[i]/float(test_runs))/(timesServer[i]/float(test_runs))

    print(avg_time_ratio) 
    #   14,                     37,                 58,                 97,                 99,             107,                107
    #[35.30431343588149, 18.22317226600444, 13.348485061514026, 2.49166855066129, 2.9477028671937107, 2.415487550237234, 2.971909439028416]

