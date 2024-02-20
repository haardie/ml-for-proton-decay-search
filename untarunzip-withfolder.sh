#!/bin/bash

function untar_folder () {
    for pathFile in `find . -name "*.tar"`;
    do
        origPath="${PWD}"
        myPath="${pathFile%/*}"
        myFile="${pathFile##*/}"
        cd "${myPath}"
        myNewFolder=$(echo "$myFile" | cut -f 1 -d '.')
        mkdir "${myNewFolder}"
        tar -xvf "${myFile}" -C "${myNewFolder}"
        rm "${myFile}"
        cd "${origPath}"
    done
}

function gunzip_folder () {
    for file in `find . -name "*.gz"`;
    do
        gunzip -dv "${file}";
    done
}

while [ `find . -name "*.tar" | wc -l` -ne 0 ]
do
    untar_folder;
done


while [ `find . -name "*.gz" | wc -l` -ne 0 ]
do
    gunzip_folder;
done

