# Generate Model
import os

def load_model_from_modelpy(model_dir_name, model_name):
    import importlib
    return importlib.import_module(model_dir_name+'.'+model_name)

def generate_dir(dir_list:list):
	result_path = os.path.join(*dir_list)
	try:
		os.makedirs(result_path)
	except FileExistsError:
		pass

	return result_path