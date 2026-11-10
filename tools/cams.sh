#!/bin/zsh
# cams.sh start|stop  — camera wall on monitor 2
case "$1" in
  stop) pkill -f "cams_avf.py" && echo stopped || echo "not running";;
  *) pkill -f "cams_avf.py" 2>/dev/null; cd ~/so101 && nohup $HOME/so101/.venv/bin/python $HOME/so101/tools/cams_avf.py > /tmp/so101_cams.log 2>&1 & echo "started (pid $!), log /tmp/so101_cams.log";;
esac
