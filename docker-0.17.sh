WORKING_DIR=/home/phuc-nguyen/workspace/mv-4261/fp8_mqa_logits
CONTAINER_NAME=phucnguyen-dev-tune-paged-fp8-0.17

docker run -d \
  --ipc=host --network=host --group-add render \
  --privileged --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  -v /remote/vast0/share-mv:/remote/vast0/share-mv \
  -v $WORKING_DIR:$WORKING_DIR \
  -w $WORKING_DIR \
  --name $CONTAINER_NAME \
  --entrypoint bash \
  255250787067.dkr.ecr.ap-northeast-2.amazonaws.com/unencrypted/moreh-vllm:0.17.0-260518-3bedd64-hsa-tilelang \
  -lc 'sleep infinity'

docker exec -ti $CONTAINER_NAME bash