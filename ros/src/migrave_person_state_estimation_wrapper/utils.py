import os
import yaml

def parse_yaml_config(config_file):
    if config_file and os.path.isfile(config_file):
        configs = {}
        with open(config_file, 'r') as infile:
            configs = yaml.safe_load(infile)
    
        return configs 

def get_game_performance(person_id, game_id):
    """
    Get game performance from database
    """

    #ToDo:
    #Implement database query to get game performance information
    
    return 0 
