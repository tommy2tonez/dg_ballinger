import os 
import network_opti_config_randomizer
import utility
import model_trainer
import glob 

training_path   = "/home/tommy2tonez/dg_projects/us_market_time_series_state_forcast/src/tuning_models/data/data.json"
training_folder = "/home/tommy2tonez/dg_projects/us_market_time_series_state_forcast/src/tuning_models" 
trained_idxes   = [os.path.splitext(os.path.basename(e))[0] for e in glob.glob(os.path.join(training_folder, "*"))]
trained_idxes   = [int(idx) for idx in trained_idxes if idx.isnumeric()]

for i in range(max(trained_idxes) + 1, 1000):
  os.mkdir(os.path.join(training_folder, str(i)))
  model_path    = os.path.join(training_folder, str(i), "model.pt")
  config_path   = os.path.join(training_folder, str(i), "config.json")
  report_path   = os.path.join(training_folder, str(i), "training_report.txt")
  model_config  = network_opti_config_randomizer.randomize_training_config(model_path, report_path, training_path)
  utility.dump_to_file(config_path, model_config)
  model_trainer.train(model_config)