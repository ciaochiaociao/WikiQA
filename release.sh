#!/bin/bash
set -e
MODULE_PATH=fgc_wiki_qa
NEURAL_MODEL_PATH=models/neural_predicate_inference
VER=${1:-}
FNAME="WIKIQA_V$VER"
KEEP_TEMP=${2:-false}

echo "copying $MODULE_PATH to temp directory ..."

mkdir -p temp_dir
cp -r $MODULE_PATH temp_dir/ > /dev/null
cp README.md temp_dir/
cp requirements.txt temp_dir/

echo "testing ..."

cp tests/test_deployed.py temp_dir/ > /dev/null
cd temp_dir/
python3 test_deployed.py

echo "test done."

rm test_deployed.py

echo "test file removed."

echo 'Compressing ...'
pwd
tar -czvf $FNAME.tar.gz $MODULE_PATH README.md requirements.txt

echo 'Moving out to released/ ...'
mv *.tar.gz ../released/

cd ..
pwd

if ! $KEEP_TEMP
then
  rm -r temp_dir
  echo "temp_dir is removed. "
fi

echo 'Done'
