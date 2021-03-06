import SAModel,os, json
from pathlib import Path

model_enum = {
"NONE NONE FLAT" : 0,

"NEGTOOL ANTONYM_LOOKUP PARSETREE" : 1,
"NEGTOOL SYM_INVERT PARSETREE" : 2,
"NEGTOOL AFFIRM_SHIFT PARSETREE" : 3,
"NEGTOOL MEANINGSPEC_FREQ PARSETREE" : 4,
"NEGTOOL MEANINGSPEC_FREQDP PARSETREE" : 5,

"WINDOW ANTONYM_LOOKUP PARSETREE" : 6,
"WINDOW SYM_INVERT PARSETREE" : 7,
"WINDOW AFFIRM_SHIFT PARSETREE" : 8,
"WINDOW MEANINGSPEC_FREQ PARSETREE" : 9,
"WINDOW MEANINGSPEC_FREQDP PARSETREE" : 10,

"PARSETREE SYM_INVERT PARSETREE" : 11,
"PARSETREE AFFIRM_SHIFT PARSETREE" : 12,
"PARSETREE MEANINGSPEC_FREQ PARSETREE" : 13,
"PARSETREE MEANINGSPEC_FREQDP PARSETREE" : 14,

"NEGTOOL ANTONYM_LOOKUP FLAT" : 15,
"NEGTOOL SYM_INVERT FLAT" : 16,
"NEGTOOL AFFIRM_SHIFT FLAT" : 17,
"NEGTOOL MEANINGSPEC_FREQ FLAT" : 18,
"NEGTOOL MEANINGSPEC_FREQDP FLAT" : 19,

"WINDOW ANTONYM_LOOKUP FLAT" : 20,
"WINDOW SYM_INVERT FLAT" : 21,
"WINDOW AFFIRM_SHIFT FLAT": 22,
"WINDOW MEANINGSPEC_FREQ FLAT": 23,
"WINDOW MEANINGSPEC_FREQDP FLAT": 24

}
model_types = {
0: "NONE NONE FLAT",

1 : "NEGTOOL ANTONYM_LOOKUP PARSETREE",
2 : "NEGTOOL SYM_INVERT PARSETREE",
3 : "NEGTOOL AFFIRM_SHIFT PARSETREE",
4 : "NEGTOOL MEANINGSPEC_FREQ PARSETREE",
5 : "NEGTOOL MEANINGSPEC_FREQDP PARSETREE",

6 : "WINDOW ANTONYM_LOOKUP PARSETREE",
7 : "WINDOW SYM_INVERT PARSETREE",
8 : "WINDOW AFFIRM_SHIFT PARSETREE",
9 : "WINDOW MEANINGSPEC_FREQ PARSETREE",
10 : "WINDOW MEANINGSPEC_FREQDP PARSETREE",

11 : "PARSETREE SYM_INVERT PARSETREE",
12 : "PARSETREE AFFIRM_SHIFT PARSETREE",
13 : "PARSETREE MEANINGSPEC_FREQ PARSETREE",
14 : "PARSETREE MEANINGSPEC_FREQDP PARSETREE",

15 : "NEGTOOL ANTONYM_LOOKUP FLAT",
16 : "NEGTOOL SYM_INVERT FLAT",
17 : "NEGTOOL AFFIRM_SHIFT FLAT",
18 : "NEGTOOL MEANINGSPEC_FREQ FLAT",
19 : "NEGTOOL MEANINGSPEC_FREQDP FLAT",

20 : "WINDOW ANTONYM_LOOKUP FLAT",
21 : "WINDOW SYM_INVERT FLAT",
22 : "WINDOW AFFIRM_SHIFT FLAT",
23 : "WINDOW MEANINGSPEC_FREQ FLAT",
24 : "WINDOW MEANINGSPEC_FREQDP FLAT"

}

models = [model_types[i] for i in range(25)]

paths_file = open(Path(os.getcwd()) / "config.txt", "r")
paths_json = json.load(paths_file)
paths_file.close()
root_path = paths_json["ROOT_PATH"]

results_analyzer = SAModel.SAResults(models,paths_json)

with open(str(Path(root_path) / "results" / "review_results6.txt"), "r") as results:
	for line in results:
		line = json.loads(line)
		truth = int(line["t"])
		predictions = line["p"]
		for i in range(len(predictions)):
			model = model_types[i]
			p = predictions[i]
			results_analyzer.categorize_prediction(model, truth, p)

print(json.dumps(results_analyzer.numberOfCorrect, indent = 4))

results_analyzer.write_to_file()





