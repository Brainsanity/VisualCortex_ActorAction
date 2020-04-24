#!/bin/sh
#
# name     : tmuxen， tmux environment made easy
# author   : Xu Xiaodong xxdlhy@gmail.com
# license  : GPL
# created  : 2012 Jul 01
# modified : 2012 Jul 02
#
 
cmd=$(which tmux) # tmux path
session=codefun   # session name
 
if [ -z $cmd ]; then
  echo "You need to install tmux."
  exit 1
fi
 
$cmd has -t $session
 
if [ $? != 0 ]; then
  $cmd new -d -n bash -s $session "bash"
  $cmd splitw -v -p 50 -t $session "bash"
  #$cmd neww -n bash -t $session "bash"
  #$cmd neww -n bash -t $session "bash"
  #$cmd neww -n bash -t $session "bash"
  #$cmd neww -n bash -t $session "bash"
  $cmd splitw -h -p 50 -t $session "bash"
  $cmd selectp -t 0
  $cmd splitw -h -p 50 -t $session "bash"
  #$cmd selectw -t $session:5
fi
 
$cmd att -t $session
 
exit 0
