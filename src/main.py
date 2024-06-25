import network_opti_config_randomizer
import sys 
import os 
import utility 
import training_data_extractor
import model_trainer

def run(p: str, sz: int):

    for i in range(sz):
        os.mkdir(os.path.join(p, str(i)))
        model_path  = os.path.join(p, str(i), "model.pt")
        config_path = os.path.join(p, str(i), "config.json")
        report_path = os.path.join(p, str(i), "training_report.txt")
        training_path = os.path.join(p, str(i), "training_data.json")
        trainer_config, extractor_config = network_opti_config_randomizer.randomize_training_config(model_path, report_path, training_path), network_opti_config_randomizer.randomize_extractor_config(training_path)
        utility.dump_to_file(config_path, trainer_config, extractor_config)
        training_data_extractor.extract(extractor_config)
        model_trainer.train(trainer_config)

def main():

    sys.setrecursionlimit(int(1e5))
    run("/home/tommy2tonez/dg_projects/us_market_time_series_state_forcast/src/06-18-2024_random/", 1000)

if __name__ == "__main__":
    main()