CALL conda create --name labmaite
CALL conda activate labmaite
CALL conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
CALL pip install -r requirements.txt