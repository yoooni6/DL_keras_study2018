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
        $ pip install opencv-contrib-python

  - python2 커널을 jupyter notebook 및 lab에 추가하기 

        $ python2 -m pip install --user ipykernel
        
* ### 위 내용처럼 하다가 그냥 Anaconda3를 설치해서 사용하기로 마음을 바꿨음
  - https://repo.continuum.io/archive/ 에서 archive 선택하여 wget
  - bash [download된 archive] 명령어로 설치
  - source .bashrc (PATH 변경 반영) 
  - 기존에 있던 python 2.7과 3.5와의 충돌이 생길지도 모르겠지만 알아서 해결하기로 함
  - jupyter notebook과 lab에 python2.7 커널 추가하기
        
        jupyter kernelspec list
        conda update conda
        conda create -n py27 python=2.7 anaconda

        source activate py27
        ipython kernelspec install-self --user
        source deactivate

  - jupyter notebook과 lab 설정 파일 변경
        
        $ jupyter notebook --generate-config
        $ vim .jupyter/jupyter_notebook_config.py

        c.NotebookApp.notebook_dir = '[directory]'
        c.NotebookApp.password = '[password]'
        c.NotebookApp.ip = '0.0.0.0'
        c.NotebookApp.port = 8888
        c.NotebookApp.open_browser = False

* ### 참조
   http://goodtogreate.tistory.com/entry/IPython-Notebook-%EC%84%A4%EC%B9%98%EB%B0%A9%EB%B2%95
   
   
* ### kaggle Data 다운받기 참조
   https://github.com/Kaggle/kaggle-api
