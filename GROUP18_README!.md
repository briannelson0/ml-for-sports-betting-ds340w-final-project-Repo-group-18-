This project is cloned from conor walshes repository for the research paper "Machine learning from sports betting: should model selection be based on accuracy or calibration?"
You can find that papers Github here: https://github.com/conorwalsh99/ml-for-sports-betting. What we have done is cloned that repo and added a third metric for sports betting: Precision.
We updated the pipeline to include precision for feuture selection, hyperparameter optimization, and model selection. The changes to the original code is not overwhelming and can be seen in the history in the repository.
I left all the original files from the original repository including their README, test folder, figures, etc. All of the changes/additions we made are in the SRC file and any other file that says group18 in it, like this one.
To run group 18's code which includes precision in the pipeline, follow the instructions below. For me, it took about 2 hours to run. 


Here is the code to run our updated pipeline with precision-based model selection on your device:

1. Open a terminal
   
2.clone the github:

git clone https://github.com/briannelson0/ml-for-sports-betting-ds340w-final-project-Repo-group-18-

3.change directory:

cd ml-for-sports-betting-ds340w-final-project-Repo-group-18-

4. install dependencies:

poetry install

6. run_pipeline
   
poetry run python src/run_pipeline.py
