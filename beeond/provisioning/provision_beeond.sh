#/bin/bash

if [ ! -z "$(mount | grep beegfs_ondemand)" ]; then
  echo BeeOND already provisioned. Will not continue
  exit 111
fi

beeond_mnt=$1

echo -e "### Installing Prerequisites:\n"

sudo apt-get install -yq jq zip unzip

echo -e "\n\n### Collecting Cluster Metadata:\n"

NODE_IP=$(ifconfig ib0 | grep -oe "inet[^6][adr: ]*[0-9.]*" | cut -d" " -f2)
MASTER_IP=$(head -1 nodefile)

echo "Node IP is: $NODE_IP"
echo "Master IP is: $MASTER_IP"

echo -e "\n\n### Configuring P2P SSH:"

echo "Installing master public key:"
cat masterkey | sudo tee -a /root/.ssh/authorized_keys

echo -e "\n\n### Provisioning BeeOND FS:\n"

echo "Adding BeeOND public key"
wget -q https://www.beegfs.io/release/latest-stable/gpg/DEB-GPG-KEY-beegfs -O- | sudo apt-key add -

echo "Adding BeeOND repo"
wget -q https://www.beegfs.io/release/beegfs_7.2/dists/beegfs-deb9.list -O- | \
  sudo tee /etc/apt/sources.list.d/beegfs-deb9.list &>/dev/null

sudo rm  /etc/apt/sources.list.d/beegfs-deb10.list &>/dev/null

sudo apt-get update -q

sudo apt-get install -y beeond

sudo mkdir -p /root/bgconf
echo ib0 | sudo tee /root/bgconf/interfaces
for serv in client helperd meta mgmtd storage; do
    cat <<EOF | sudo tee /root/bgconf/beegfs-${serv}.conf
connInterfacesFile = /root/bgconf/interfaces
EOF
done
# OFED_INCLUDE_PATH=/usr/src/ofa_kernel-5.1/include
cat <<EOF | sudo tee /etc/beegfs/beegfs-client-autobuild.conf
buildArgs=-j20
buildEnabled=true
EOF

if [ "${MASTER_IP}" == "${NODE_IP}" ]; then
  sleep 30s
  echo "\n\n## Master Node running BeeOND startup:\n"

  sudo beeond start -n nodefile -f /root/bgconf -d /mnt/resource/beeond -c $beeond_mnt -F

fi
