#!/bin/bash
for f in $(grep -rIl $'\t' ../../Source)
do
	echo "Converting tabs to spaces in $f"
	expand -t 4 $f > tmp_file
	mv tmp_file $f
done
