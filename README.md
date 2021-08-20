# Project LabmAIte


## 1. Installation Guide
   Note: These commands are written for mac/ubuntu platforms, on windows the paths have to be changed to windows style, i.e. / becomes \ .
   ### Requirements
   The code is developed using mac and ubuntu platforms. Use on windows platforms is possible but may produce errors. For any support requests please file an issue in github's [issue section](https://github.com/d-raith/labmAIte/issues).<br> 
   For the RL part we recommend a Intel i5 CPU or later. Same holds for the vision part which additionally requires a GPU with at least 4 GB of RAM (8GB recommended).<br>
   The project is developed using [IntelliJ's PyCharm](https://www.jetbrains.com/pycharm/) editor. To enable reproducible errors we recommend using this editor for development and execution.<br> 
   
   ### 1.1 Code Download
   The git command line tool or any git GUI tool is required to download this code repository. <br>
   To download the repository with the command line enter the following command in your terminal:<br>
   If you have provided your public SSH key use the following command:
   ```
   git clone  git@github.com:d-raith/labmAIte.git
   ```
   If you want to use your git credentials every time you connect to the github server use the following command:
   ```
   git clone  https://github.com/d-raith/labmAIte.git
   ```
   The repository will be downloaded to your command line's current working directory.
   
   ### 1.2 Anaconda
   1.2.1 Install the anaconda toolkit: https://www.anaconda.com/distribution/ <br>
   1.2.2 Open a terminal and create a new Python 3.7 anaconda environment from the provided .yml file in the `labmAIte` directory:
   ```
   conda env create -f {path_to_labmaite_directory}/labmaite_conda_environment.yml
   ```
   ### 1.3 PyCharm
   1.3.1 Open the PyCharm or desired development IDE and select the cloned `labmAIte` directory as project root.<br> 
   Then configure the IDE to use the created anaconda environment with the name ``labmaite``. <br>
   For PyCharm this setting can be found in the preferences under `preferences -> project: labmAIte -> project interpreter`. <br>
   If the environment is not yet listed in the dropdown menu, press ``show all`` in the dropdown menu and then the `add-symbol` on the lower left.<br>
   In the resulting window, go to your anaconda home directory, open the `envs` folder and search for the `labmaite` folder. In the `labmaite/bin` folder select the `python` executable and press ok.<br>
   The recently created anaconda environment should now be listed in the dropdown list. Select it and save / exit the preferences.
   
   ### 2. Code Structure
   The repository consists of three major modules: ```rl, vision and utils```.<br><br>
   `rl` contains code regarding the ai-driven control part of the project using the [stable-baselines](https://github.com/hill-a/stable-baselines) tensorflow library. RL agents can be trained and executed here on a color-mix simulation or the real experiment.<br><br>
   `vision` contains the code for fluorescence prediction and object detection of individual cells using [PyTorch](https://pytorch.org/). It is currently under development and not yet verified to work without errors.<br><br>
   `utils` contains a variety of utility classes as well as the necessary code to access and control the pumps using the serial interface. <br>
   The color-mix simulation core also resides here. <br>
   For the basic execution of the RL and vision part this module is not required to be changed. 
   
   ### 3. Usage
   An example usage of the RL module is documented in the module's readme. A documentation of the vision part will be added in the future. <br>
   Please note that this project was originally developed as a personal project and therefore currently lacks documentation.<br><br> 
   *For any questions, problems with code execution, configuration or other feature requests, please use the mentioned [issue section](https://github.com/d-raith/labmAIte/issues).*
    
    
   ### 4. Old Tutorial (to be improved)
   
   #### Ibidi Chip:
 The top cover of the chip must be drilled in order to insert the tubing (with its corresponding metal point) for both inlets -syringes- and outlet -draining pump-. The outlet may have the adaptor shown in the first picture but it is not necessary. 
To focus the camera always in the same spot we drew an arrow with black permanent marker on the back part of the chip (marked in red).  
To enhance the contrast with the color, an even white cardboard is attached with scotch tape as shown in the second image.

#### Peristaltic pump:
Our configuration regarding the draining pump was continuous draining.

#### USB Hub:
Try setup_checks.py to see whether and where the pumps and microscope are connected. 
Check the file config.json to configure the ports and assure the correct port names.

### 5. Short Command Overview for Git and Anaconda:
Some command cheatsheet for anaconda:
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

- list all conda environments available:
conda env list

- activate environment:
conda activate "env_name"

- create conda environment
conda create --name "env_name" python=3.7


Git commands:

- Check modified files and the current branch:
git status

- Reset all changes to the last time a commit was made:
git reset --hard

- Pull the latest changes from the remote branch:
git pull

- undo - but store - the changes since the last commit
(good to use if you cant pull because of local changes you dont want to commit yet)
git stash

- redo the stored changes on the current code:
git stash pop

- view local and remote branches:
git branch -a

- change current branch
git checkout "branch_name"

- create new local branch:
git checkout -b "branch_name"

   

