def get_paths_json():
	from pathlib import Path
	import os, json
	paths_file = open(Path(os.getcwd()) / "config.txt", "r")
	paths_json = json.load(paths_file)
	paths_file.close()
	return paths_json