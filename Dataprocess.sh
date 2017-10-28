#!/bin/bash 


mkdir NEW
 
list_alldir(){  
    for file2 in `ls -a $1`  
    do  
        if [ x"$file2" != x"." -a x"$file2" != x".." ];then  
            if [ -d "$1/$file2" ];then  
                echo "$1/$file2"  
                #list_alldir "$1/$file2"  
		for data in `ls -a $1/$file2`
		do
			echo $1,"\t",$file2,"\t",$data
			size=`du -h $1/$file2/$data | awk '{print $1}'`
			echo $size,$size2,$1/$file2/$data
			if [ "$size"x = "0K"x ];then
				continue
			else
				myPath="NEW"/$file2
				if [ ! -d "$myPath" ]; then
 					mkdir -p $myPath
				fi
					python test.py $1/$file2/$data > temp
					mv temp $myPath/$data
					rm -rf temp
			fi
		done
            fi  
        fi  
    done  
}  
  
list_alldir ./new_train
