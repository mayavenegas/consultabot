#!/bin/bash
TOKEN=`curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
PUB_HOSTNAME=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-hostname)
PUB_IPV4=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)

sudo apt update
sudo apt install -y nginx
pushd mkcert
  ./install-mkcert.sh
  mkcert query.consultabot.com ${PUB_HOSTNAME} ${PUB_IPV4}
  mv *-key.pem ../key.pem
  mv *.pem ../cert.pem
popd
sudo cp nginx.conf /etc/nginx/
sudo mv *.pem /etc/nginx/
sudo nginx -s reload
