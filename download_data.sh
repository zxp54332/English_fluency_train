#!/bin/bash

set -e

url=www.openslr.org/resources/101
data=./data
chmod +x move_file.sh

[ -d $data ] || mkdir -p $data

corpus_name=speechocean762

if [ -z "$url" ]; then
    echo "$0: empty URL base."
    exit 1;
fi

if [ -f $data/$corpus_name/.complete ]; then
    echo "$0: data part $corpus_name was already successfully extracted, nothing to do."
    exit 0;
fi

full_url=$url/$corpus_name.tar.gz

echo "$0: downloading data from $full_url.  This may take some time, please be patient."
if ! wget -c --no-check-certificate $full_url -O $data/$corpus_name.tar.gz; then
    echo "$0: error executing wget $full_url"
    exit 1;
fi


cd $data
if ! tar -xvzf $corpus_name.tar.gz; then
    echo "$0: error un-tarring archive $data/$corpus_name.tar.gz"
    exit 1;
fi

rm ./speechocean762.tar.gz
touch $corpus_name/.complete
cd -

echo "$0: Successfully downloaded and un-tarred $data/$corpus_name.tar.gz"

./move_file.sh

echo "$0: Successfully split train、test file"