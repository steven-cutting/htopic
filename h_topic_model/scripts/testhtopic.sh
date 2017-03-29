#! /usr/bin/env bash


textfiles_dir=$1
randomstate=2017
logfile=log.log

echo $textfiles_dir
echo "random state: ${randomstate}"
echo "log file: ${logfile}"

echo "htopic -R $randomstate -L $logfile mkwc $textfiles_dir"| bash -x
echo "htopic -R $randomstate -L $logfile morfessor" | bash -x
echo "htopic -R $randomstate -L $logfile mkcorpus $textfiles_dir" | bash -x
echo "htopic -R $randomstate -L $logfile mkmodel" | bash -x
