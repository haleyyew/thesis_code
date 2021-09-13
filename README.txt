sudo apt install python3-pip

sudo apt install python3-virtualenv
virtualenv -p /usr/bin/python3 venv

source python_venv/venv/bin/activate

pip3 install numpy pandas strsim nltk

python3
help('modules')
import nltk
nltk_path = "/media/haleyyew/e2660490-a736-4bb9-b3dd-5c0f3871a2f2/thesis_code/python_venv/nltk_data/"
nltk.download("wordnet", nltk_path)
nltk.download("words", nltk_path)
nltk.data.path.append(nltk_path)
exit()

deactivate
