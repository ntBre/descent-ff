set -xe

mkdir -p outputs

record=8150601

curl -o outputs/$record.zip https://zenodo.org/api/records/$record/files-archive
unzip outputs/$record.zip -d outputs/$record

cd outputs/$record

for f in *.tar.gz; do 
	tar -zxf "$f" && rm -r "$f"
done
