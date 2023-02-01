# GCP - Ubuntu20.04 Tensorflow설치 문서.

### 아래 사이트에서 cuda 11.0.3, cudnn 11.0-8.0.5.39, tensorflow 2.4.1, tensorflow-gpu2.4.1 설치.
https://gist.github.com/mikaelhg/cae5b7938aa3dfdf3d06a40739f2f3f4

##### cuda 11.0.3
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run

sudo bash cuda_11.0.3_450.51.06_linux.run --no-man-page --override --silent \
  --toolkit --toolkitpath=/usr/local/cuda-11.0.3 --librarypath=/usr/local/cuda-11.0.3

##### cudnn 11.0-8.0.5.39
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.0-linux-x64-v8.0.5.39.tgz

sudo mkdir /usr/local/cudnn-11.0-8.0.5.39
sudo tar -xzf cudnn-11.0-linux-x64-v8.0.5.39.tgz -C /usr/local/cudnn-11.0-8.0.5.39 --strip 1

##### tensorflow 2.4.1, tensorflow-gpu2.4.1 
conda install tensorflow==2.4.1
conda install tensorflow-gpu==2.4.1
conda install typing_extensions  # for albumentation error


##### LD_LIBRARY_PATH는 python부터 에러나서 빼고 추가함.
$ ln -snf /usr/local/cuda-11.0.3/bin bin  # 추가 안함.
$ LD_LIBRARY_PATH=/usr/local/cuda-11.0.3/lib64:/usr/local/cuda-11.0.3/extras/CUPTI/lib64:/usr/local/cudnn-11.0-8.0.5.39/lib64 python -c 'import tensorflow'
$ LD_LIBRARY_PATH=/usr/local/cuda-11.0.3/lib64:/usr/local/cuda-11.0.3/extras/CUPTI/lib64:/usr/local/cudnn-11.0-8.0.5.39/lib64:$LD_LIBRARY_PATH



##### gcp 방화벽 설정.
사용하고자 하는 port에 tag규칙에 https-server, http-server 추가.


##### ssh 연결 끊기지 않게 수정.

- ssh-key expired 되는 문제점
아래와 같이 ssh-key 생성후, meta-data에 추가.

https://cloud.google.com/compute/docs/connect/create-ssh-keys?hl=ko#create_an_ssh_key_pair

https://cloud.google.com/compute/docs/connect/add-ssh-keys?hl=ko

https://cloud.google.com/compute/docs/connect/add-ssh-keys?hl=ko#after-vm-creation


- sshd 연결 옵션.
sudo vi /etc/ssh/sshd_config
   and set TCPKeepAlive yes
sudo service ssh restart

