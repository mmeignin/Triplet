#!/bin/bash



#SBATCH --mem=7G
#SBATCH --cpus-per-task=6

#SBATCH -w figuier


app="$(pwd)/../"
pythonEnv="${app}distance/"
. ${pythonEnv}"bin/activate"


##install requiremennts for the training
if [ ${VIRTUAL_ENV:(-8)} == "distance" ]; then 
        #pip install -r "$(pwd)/requirements.txt"
        cd ..
        python3 triplet_train.py
                                
else 
        echo "Virtual Environment issue, env name: ${$VIRTUAL_ENV}" 
fi
##Check missing requirements
#pip freeze > virtual_env_requirements.txt
deactivate

exit 0
