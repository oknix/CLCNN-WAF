FILE_ID=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE
FILE_NAME=amazon_review_polarity_csv.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}