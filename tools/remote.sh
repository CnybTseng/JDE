#!/bin/sh

sshpass -p $1 ssh $2@$3 -p $4 "cd \"$5\"; ls -l; exit"