docker build . -t linto-diarization-simple:3.0.11 -f simple/Dockerfile || exit 1

echo
echo Go to http://localhost:8080/docs
echo
OPT="-v /home/wghezaiel/.cache:/root/.cache --name linto-diarization-simple"
OPT="$OPT --shm-size=1gb --tmpfs /run/user/0"
for folder in "/data-local" "/data-storage" "/media"; do
    if [ -d $folder ];then
        OPT="$OPT -v $folder:$folder"
    fi
done
docker run -it --rm --runtime=nvidia  -p 8080:80 $OPT -v /tmp:/opt/audio --env-file .envdefault linto-diarization-simple:3.0.11


#-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all