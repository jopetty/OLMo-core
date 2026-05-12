beaker session create \
    --remote \
    --bare \
    --cluster ai2/saturn \
    --gpus 1 \
    --min-runtime 2h \
    --mount src=weka,ref=reviz-default,dst=/weka
