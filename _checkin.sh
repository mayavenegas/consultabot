#!/bin/bash
if [[ "$1" == "" ]]; then
  echo "Usage: $0 <commit-comment>"
  exit -1
fi
pushd bot-server; rm -rf logs; popd
pushd bot-ui; rm -rf logs; popd
echo "Disk usage:"
du -sh .
gitleaks detect -r .gitleaks
cat .gitleaks
#read -p "Do you wish to continue? " yn
yn='y'
case $yn in
  [Yy]* )
	git add .; git commit -m "$1"; git push origin main
	;;
  * )
	exit
	;;
esac
