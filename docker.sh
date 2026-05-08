WORKING_DIR=/remote/vast0/phuc-nguyen/workspace/tickets/mv-4077/optests/fp8_paged_mqa_logits
CONTAINER_NAME=phucnguyen-dev-tune-paged-fp8

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
  vllm/vllm-openai-rocm:v0.19.0 \
  -lc 'sleep infinity'

docker exec -ti $CONTAINER_NAME bash