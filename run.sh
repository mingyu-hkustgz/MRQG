#datasets=("gist" "msmarc-small" "tiny5m" "msong" "OpenAI-1536" "word2vec" "OpenAI-3072")
datasets=("OpenAI-3072" "OpenAI-1536")
mkdir results
mkdir results/index-time
mkdir results/index-space
mkdir results/index-space
mkdir results/recall@20
mkdir results/recall@100

for data in "${datasets[@]}"; do
    mkdir data/$data
    mkdir results/recall@20/$data
    mkdir results/recall@100/$data
done

rm -rf build
mkdir build
cd build
cmake ..
make -j 48
cd ..


for data in "${datasets[@]}"; do
  echo "Searching & Indexing - ${data}"

  if [ $data == "tiny5m" ]; then
    bits=256
  elif [ $data == "msong" ]; then
    bits=256
  elif [ $data == "word2vec" ]; then
    bits=256
  elif [ $data == "gist" ]; then
    bits=512
  elif [ $data == "OpenAI-1536" ]; then
    bits=512
  elif [ $data == "OpenAI-3072" ]; then
    bits=512
  elif [ $data == "deep1M" ]; then
    bits=128
  elif [ $data == "msmarc-small" ]; then
    bits=512
  fi

   ./build/test/qg_build $data

  ./build/test/qg_search $data 20

  ./build/test/qg_search $data 100

   ./build/test/mrqg_build $data $bits

  ./build/test/mrqg_search $data $bits 20

  ./build/test/mrqg_search $data $bits 100

  done