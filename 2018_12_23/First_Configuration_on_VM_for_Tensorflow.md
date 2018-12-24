## 처음 설정하기 - Jupyter lab과 텐서플로우 

* ### Google Cloud VM에 필요한 패키지 설치하기
  - python2와 3이 미리 설치되어있기 때문에 사용하면서 하나씩 필요한 것들을 그때 그때 설치함

        $ sudo apt-get install python-pip
        $ pip2 install --upgrade pip
        $ sudo apt-get install python3-pip
        $ pip3 install --upgrade pip
        
        * 혹시 문제 생기면 
        $ sudo python3 -m pip uninstall pip && sudo apt install python3-pip --reinstall)
        
        $ pip install --user tensorflow-gpu
        $ pip install --user keras
        $ pip install --user matplotlib
        $ pip install --user pandas
        $ pip install --user jupyterlab
        $ pip install --user jupyter

  - jupyter notebook과 lab 설정 파일 변경
        
        $ jupyter notebook --generate-config
        $ vim .jupyter/jupyter_notebook_config.py


        c.NotebookApp.notebook_dir = '[directory]'
        c.NotebookApp.password = '[password]'
        c.NotebookApp.ip = '0.0.0.0'
        c.NotebookApp.port = 8888
        c.NotebookApp.open_browser = False

  - python2 커널을 jupyter notebook 및 lab에 추가하기 

        $ python2 -m pip install --user ipykernel

* ### 참조
   http://goodtogreate.tistory.com/entry/IPython-Notebook-%EC%84%A4%EC%B9%98%EB%B0%A9%EB%B2%95
