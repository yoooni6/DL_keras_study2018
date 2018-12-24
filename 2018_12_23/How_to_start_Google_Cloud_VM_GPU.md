## 구글 클라우드 플랫폼에 GPU가 추가된 VM 생성하기 
### 작성일 : 2018.12.23

<hr/>

### **1. 글을 쓰는 이유**
    인터넷에 친절하게 설명된 글들을 그대로 따라했지만 내가 잘못했거나 구글 클라우드가 바뀌었거나 어떤 이유로 GPU 할당에 애를 먹었다.

    Zone을 바꿔가며 4번 이상 신청을 하였지만 실패했고 결국 다른 분의 도움을 얻어서 내가 잘못 생각하고 있던 부분을 찾을 수 있었다.
    
    이런 삽질을 다음에 또 하지 않길 바라며 글을 써본다.

### **2. 작업 순서**
* 구글 클라우드 플랫폼에 가입하기
* 새 프로젝트 만들기
* GPU 할당량 신청하기
* GPU가 포함된 VM 생성하기
* SSH 접속 후 GPU Driver(CUDA, cuDNN) 설치하기

      # CUDA Toolkit 9.0 설치
      curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
      sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
      sudo apt-get update
      sudo apt-get install cuda-9-0 -y --allow-unauthenticated
      
      nvidia-smi  # 설치 확인

      # Nvidia Developer Program에 회원가입 후 cuDNN 다운로드 사이트에서 tensorflow에서 지원하는 버전 중 CUDA와 호환되는 파일 Runtime 버전과 Developer 버전 모두 Download(이후 SSH 창에서 업로드)

      sudo dpkg -i libcudnn7*
      echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
      echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
      echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
      source ~/.bashrc

* (Optional) docker-ce, nvidia-docker2 설치하기
    
      # Docker 설치하기
      #/bin/bash
      # install packages to allow apt to use a repository over HTTPS:
      sudo apt-get -y install \
      apt-transport-https ca-certificates curl software-properties-common
      # add Docker’s official GPG key:
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
      # set up the Docker stable repository.
      sudo add-apt-repository \
         "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
         $(lsb_release -cs) \
         stable"
      # update the apt package index:
      sudo apt-get -y update
      # finally, install docker
      sudo apt-get -y install docker-ce

      # nvidia-docker2 설치하기
      curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
      curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
      sudo apt-get -qq update
      sudo apt-get install -y nvidia-docker2
      sudo pkill -SIGHUP dockerd

      # Tensorflow 가 포함된 Dockerfile 만들고 실행하기
      cat << EOF > Dockerfile.tensorflow_gpu_jupyter
      FROM tensorflow/tensorflow:latest-gpu
      RUN apt-get update && apt-get install -y python-opencv python-skimage git
      RUN pip install requests ipywidgets seaborn
      RUN jupyter nbextension enable --py widgetsnbextension
      CMD ["/run_jupyter.sh", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password='sha1:a6179df3a1ce:383a2f049eb0fdf432e439b3c7170e21a3e07312'"]
      EOF

      sudo docker build -t tensorflow_gpu_jupyter -f Dockerfile.tensorflow_gpu_jupyter .
      sudo nvidia-docker run -dit --restart unless-stopped -p 8888:8888 tensorflow_gpu_jupyter
* 방화벽 오픈하기
* VM Restart

참조 - https://right1203.github.io/tool/2018/09/30/gcp-datalab/
> nvidia-docker2를 사용하지 않을 거라면 Optional 과정은 필요없음

### **3. 주요 사항**
* 위의 참조 링크에 나와있는 내용을 따라하면 대부분 문제없이 진행될 것으로 생각됨
* 하지만 GPU 할당량 증가 요청 시 문제에 봉착했었음
  - 글에는 '계정 업그레이드' 후 사용자가 원하는 위치(Zone)을 선택하고 신청하는 순서로 기록되어 있음
  - 지금은 처음 접속했어도 거의 모든 Zone의 GPU 할당량이 0이 아닌 1이라고 적혀있음
  - 단 하나, 'GPUs (all regions)'의 할당량만 0으로 적혀있는데 이 할당량을 1로 증가시키는 신청을 해야함
    
    + 나의 추측으로는 초기에 모든 지역 별 GPU 할당량은 1로 세팅되지만 글로벌 GPU 할당량을 0으로 맞춰놔서 초기에 사용하지 못하도록 정책을 변경한 것 같음. 
    
    + 나중에 무료 크레딧이 끝나고 GPU를 2개 이상 쓰고 싶을 때는 글로벌 할당량을 증가시키는 것과 특정 지역 할당량을 증가시키는 걸 동시에 고려해야 할 것 같은데 그건 그 때 가서 생각하려함.

* Compute engine 메뉴에서 VM을 생성할 때 '인스턴스 만들기' 화면에서 CPU 설정 시 '맞춤 설정' 버튼을 눌러야 GPU를 추가시키는 버튼이 보임

### **4. 마무리**
    위의 주요 사항을 참조하여 링크에 나온 내용을 적절히 수행하면 무리없이 할 수 있다.
