./ss-server -s 0.0.0.0 -p8388 -l1080 -m aes-256-cfb -k zhangsen --fast-open
ss-local -s 119.8.23.14 -p 8388 -l 1080 -k zhangse -m aes-256-cfb -t 60 -v --reuse-port
source ~/pytorch-env/bin/activate
